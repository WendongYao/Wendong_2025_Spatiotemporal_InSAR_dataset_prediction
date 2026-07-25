"""Backfill persisted direct-point predictions for manuscript LASSO runs.

The historical runs saved the fitted state, dense prediction, and metrics but
not the direct test predictions used by the primary endpoint.  This utility
reconstructs those predictions from the frozen state without refitting, checks
the reconstructed RMSE against the recorded metric, and writes a forensic
manifest for the backfill operation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from raw_holdout_data import (  # noqa: E402
    RawHoldoutSpec,
    build_raw_holdout_task,
    load_forecast_columns,
)
from raw_point_supervision import _predict_torch_l1_regressor  # noqa: E402
from run_controlled_buffer_suite import _make_task  # noqa: E402


RESULT_ROOTS = (
    "R069_saqr_frozen_seed42/lasso_raw_supervised",
    "R070_saqr_multiregion_seed42/*/lasso_raw_supervised",
    "R081_external_seeds43_46/*/seed_*/lasso_raw_supervised",
    "R082_controlled_buffer_E32N34_seed42/*/lasso_raw_supervised",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_manifest(directory: Path) -> Path:
    for parent in (directory, *directory.parents):
        candidate = parent / "run_manifest.json"
        if candidate.exists():
            return candidate
        if parent == PROJECT_ROOT:
            break
    raise FileNotFoundError(f"No run_manifest.json found above {directory}")


def _load_state(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normal_task(manifest: dict[str, object], cache_dir: Path):
    spec = RawHoldoutSpec(
        csv_path=Path(str(manifest["csv_path"])),
        tile=str(manifest["tile"]),
        grid_size=int(manifest.get("grid_size", 256)),
        split_seed=int(manifest["seed"]),
        block_side=int(manifest.get("block_side", 8)),
        buffer_blocks=int(manifest.get("buffer_blocks", 0)),
    )
    return spec, build_raw_holdout_task(spec, cache_dir=cache_dir)


def _controlled_task(
    manifest: dict[str, object],
    condition: str,
    cache_dir: Path,
):
    spec = RawHoldoutSpec(
        csv_path=Path(str(manifest["csv_path"])),
        tile=str(manifest["tile"]),
        grid_size=256,
        split_seed=int(manifest["seed"]),
        block_side=8,
        buffer_blocks=0,
    )
    base = build_raw_holdout_task(spec, cache_dir=cache_dir)
    task, _, _ = _make_task(
        base,
        axis=0 if manifest["axis"] == "easting" else 1,
        test_side=str(manifest["test_side"]),
        tail_fraction=float(manifest["tail_fraction"]),
        buffer_meters=float(manifest["buffer_meters"]),
        condition=condition,
        seed=int(manifest["seed"]),
    )
    return spec, task


def main() -> None:
    results_root = PROJECT_ROOT / "results"
    cache_dir = results_root / "_raw_task_cache"
    output_root = results_root / "R084_lasso_prediction_backfill"
    output_root.mkdir(parents=True, exist_ok=True)

    lasso_dirs: list[Path] = []
    for pattern in RESULT_ROOTS:
        lasso_dirs.extend(sorted(results_root.glob(pattern)))
    lasso_dirs = sorted(set(path.resolve() for path in lasso_dirs))
    if len(lasso_dirs) != 18:
        raise AssertionError(f"Expected 18 manuscript LASSO directories, found {len(lasso_dirs)}")

    raw_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    records: list[dict[str, object]] = []
    for lasso_dir in lasso_dirs:
        manifest_path = _nearest_manifest(lasso_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "R082_controlled_buffer" in str(lasso_dir):
            condition = lasso_dir.parent.name
            spec, task = _controlled_task(manifest, condition, cache_dir)
        else:
            spec, task = _normal_task(manifest, cache_dir)

        csv_key = str(Path(str(manifest["csv_path"])).resolve())
        if csv_key not in raw_cache:
            raw_cache[csv_key] = load_forecast_columns(spec)
        points, raw_history, raw_target = raw_cache[csv_key]
        if not np.allclose(points, task.raw_points) or not np.allclose(raw_target, task.raw_target):
            raise AssertionError(f"Raw data order differs from task for {lasso_dir}")

        state_path = lasso_dir / "lasso_state.pth"
        metrics_path = lasso_dir / "metrics.json"
        state = _load_state(state_path)
        indices = task.test_target_indices.astype(np.int64)
        prediction = _predict_torch_l1_regressor(
            raw_history[indices], state, device_str="cpu"
        ).astype(np.float32)
        truth = task.raw_target[indices].astype(np.float32)
        reconstructed_rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        recorded_rmse = float(metrics["direct_raw_rmse"])
        if not np.isclose(reconstructed_rmse, recorded_rmse, rtol=0.0, atol=2e-6):
            raise AssertionError(
                f"RMSE mismatch for {lasso_dir}: {reconstructed_rmse} vs {recorded_rmse}"
            )

        prediction_path = lasso_dir / "direct_raw_test_predictions.npz"
        np.savez_compressed(
            prediction_path,
            indices=indices,
            points=task.raw_points[indices].astype(np.float64),
            truth=truth,
            prediction=prediction,
            residual=(prediction - truth).astype(np.float32),
        )
        record = {
            "lasso_dir": str(lasso_dir),
            "source_manifest": str(manifest_path.resolve()),
            "tile": str(manifest["tile"]),
            "seed": int(manifest["seed"]),
            "condition": lasso_dir.parent.name if "R082_controlled_buffer" in str(lasso_dir) else "spatial_block",
            "point_count": int(len(indices)),
            "recorded_direct_raw_rmse": recorded_rmse,
            "reconstructed_direct_raw_rmse": reconstructed_rmse,
            "state_sha256": _sha256(state_path),
            "metrics_sha256": _sha256(metrics_path),
            "prediction_sha256": _sha256(prediction_path),
        }
        (lasso_dir / "prediction_backfill.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records.append(record)
        print(json.dumps({"backfilled": str(lasso_dir), "rmse": reconstructed_rmse}), flush=True)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "prediction_backfill_without_refitting",
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "run_count": len(records),
        "records": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
