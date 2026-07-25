"""Known-truth diagnostic for interpolated future-target self-consistency.

This experiment intentionally recreates the non-deployable gridding-first
protocol questioned by the reviewers.  Sparse analytic future observations
are interpolated to a pseudo-dense supervision map, a dense model is fitted to
that map, and the resulting prediction is scored twice on the same held-out
grid cells: once against the interpolated pseudo-target and once against the
independent analytic field.  The gap quantifies how an apparently favourable
gridded score can reflect the target operator rather than the known truth.

The pseudo-target uses all sparse future observations by design.  This is an
explicit leakage/confounding diagnostic, not a valid forecasting protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_config import RevisionConfig  # noqa: E402

from raw_holdout_data import RawHoldoutTask, idw_interpolate  # noqa: E402
from raw_holdout_models import run_lasso, run_patch_model  # noqa: E402
from synthetic_truth_data import (  # noqa: E402
    SyntheticTruthSpec,
    analytic_field,
    build_synthetic_truth_task,
    sample_irregular_support,
)


SOURCE_COMMIT = "ffc1d4e8eb09c86ac81faa09ff662868b7494162"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate_target(
    support: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    method: str,
    *,
    neighbors: int,
    power: float,
) -> np.ndarray:
    if method == "idw":
        return idw_interpolate(support, values, queries, neighbors=neighbors, power=power)
    if method == "nearest":
        indices = cKDTree(support).query(queries, k=1)[1]
        return np.asarray(values[indices], dtype=np.float32)
    if method == "linear":
        result = np.asarray(LinearNDInterpolator(support, values, fill_value=np.nan)(queries))
        missing = ~np.isfinite(result)
        if missing.any():
            indices = cKDTree(support).query(queries[missing], k=1)[1]
            result[missing] = values[indices]
        return result.astype(np.float32)
    raise ValueError(method)


def build_pseudo_target_task(
    spec: SyntheticTruthSpec,
    target_interpolation: str,
) -> tuple[RawHoldoutTask, np.ndarray, np.ndarray]:
    analytic_task = build_synthetic_truth_task(spec)
    support = sample_irregular_support(spec)
    support_target = analytic_field(
        support[:, 0],
        support[:, 1],
        float(spec.target_step),
        spec.scenario,
    )
    axis = analytic_task.easting_axis
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    queries = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    pseudo_target = interpolate_target(
        support,
        support_target,
        queries,
        target_interpolation,
        neighbors=spec.idw_neighbors,
        power=spec.idw_power,
    ).reshape(spec.grid_size, spec.grid_size)
    analytic_target = analytic_task.dense_task.target_map.astype(np.float32)
    dense_task = replace(
        analytic_task.dense_task,
        target_map=pseudo_target.astype(np.float32),
        interpolation_method=(
            f"{spec.input_interpolation}_inputs_"
            f"{target_interpolation}_full_support_future_target"
        ),
    )
    metadata = {
        **analytic_task.metadata,
        "protocol": "analytic-interpolated-target-confound-v1",
        "target_interpolation": target_interpolation,
        "target_source": "interpolated_full_support_future_observations",
        "test_target_used_in_any_target_grid": True,
        "deployable_forecast": False,
        "diagnostic_only": True,
    }
    raw_task = replace(
        analytic_task,
        dense_task=dense_task,
        raw_target=analytic_target.ravel(),
        metadata=metadata,
    )
    return raw_task, analytic_target, pseudo_target.astype(np.float32)


def add_dual_truth_metrics(
    payload: dict[str, object],
    prediction: np.ndarray,
    analytic_target: np.ndarray,
    pseudo_target: np.ndarray,
    raw_task: RawHoldoutTask,
) -> dict[str, object]:
    test = raw_task.dense_task.test_mask
    analytic_error = prediction[test] - analytic_target[test]
    pseudo_error = prediction[test] - pseudo_target[test]
    target_distortion = pseudo_target[test] - analytic_target[test]
    payload.update(
        {
            "analytic_test_rmse": float(np.sqrt(np.mean(analytic_error**2))),
            "pseudo_target_test_rmse": float(np.sqrt(np.mean(pseudo_error**2))),
            "pseudo_target_distortion_rmse": float(np.sqrt(np.mean(target_distortion**2))),
            "optimism_gap_mm": float(
                np.sqrt(np.mean(analytic_error**2)) - np.sqrt(np.mean(pseudo_error**2))
            ),
            "analytic_test_mae": float(np.mean(np.abs(analytic_error))),
            "pseudo_target_test_mae": float(np.mean(np.abs(pseudo_error))),
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenario", default="composite", choices=[
        "seasonal_trend", "localized_acceleration", "moving_front", "composite"
    ])
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--support-points", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-interpolation", default="idw", choices=["idw", "linear", "nearest"])
    parser.add_argument(
        "--target-interpolations",
        nargs="+",
        default=["idw", "linear", "nearest"],
        choices=["idw", "linear", "nearest"],
    )
    parser.add_argument("--models", nargs="+", default=["lasso", "cnn_lstm_hybrid"], choices=["lasso", "cnn_lstm_hybrid"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spec = SyntheticTruthSpec(
        scenario=args.scenario,
        grid_size=args.grid_size,
        support_points=args.support_points,
        observation_noise_std=args.noise,
        seed=args.seed,
        split_seed=args.seed,
        tile_size=max(8, args.grid_size // 8),
        input_interpolation=args.input_interpolation,
    )
    config = RevisionConfig(
        csv_path="synthetic://analytic-interpolated-target-confound",
        grid_size=args.grid_size,
        split_seed=args.seed,
        split_strategy="spatial_tile",
        interpolation_method=args.input_interpolation,
        cnn_epochs=args.epochs,
        cnn_patience=args.patience,
        cnn_learning_rate=3e-4,
        cnn_weight_decay=1e-5,
        patch_size=16,
        patch_stride=8,
        patch_min_valid_pixels=8,
        patch_batch_size=16,
        nontransformer_hybrid_hidden_channels=64,
        convlstm_hidden_dim=64,
        convlstm_num_layers=1,
        lasso_epochs=300,
        lasso_patience=40,
        lasso_learning_rate=2e-2,
        output_root=args.output_root,
        use_task_cache=False,
    )
    code_files = [
        PROJECT_ROOT / "experiments_ext" / "synthetic_truth_data.py",
        PROJECT_ROOT / "experiments_ext" / "raw_holdout_models.py",
        Path(__file__).resolve(),
    ]
    manifest: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "source_git_commit": SOURCE_COMMIT,
        "extension_code_sha256": {path.name: sha256(path) for path in code_files},
        "spec": spec.__dict__,
        "models": args.models,
        "target_interpolations": args.target_interpolations,
        "warning": (
            "Diagnostic only: pseudo-target maps use all sparse future observations, "
            "including observations in held-out spatial regions."
        ),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for target_interpolation in args.target_interpolations:
        raw_task, analytic_target, pseudo_target = build_pseudo_target_task(spec, target_interpolation)
        condition_dir = args.output_root / f"target_{target_interpolation}"
        condition_dir.mkdir(parents=True, exist_ok=True)
        np.save(condition_dir / "analytic_target.npy", analytic_target)
        np.save(condition_dir / "pseudo_target.npy", pseudo_target)
        for model_name in args.models:
            model_dir = condition_dir / model_name
            if model_name == "lasso":
                payload = run_lasso(raw_task, config, model_dir)
            else:
                payload = run_patch_model(
                    raw_task,
                    config,
                    model_dir,
                    model_name=model_name,
                    model_kind="cnn_lstm_hybrid",
                    use_warm_start=True,
                )
            prediction = np.load(model_dir / "prediction_grid.npy")
            payload = add_dual_truth_metrics(
                payload,
                prediction,
                analytic_target,
                pseudo_target,
                raw_task,
            )
            payload.update(
                {
                    "input_interpolation": args.input_interpolation,
                    "target_interpolation": target_interpolation,
                    "operator_match": bool(args.input_interpolation == target_interpolation),
                }
            )
            (model_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
            )
            row = {
                "model": model_name,
                "input_interpolation": args.input_interpolation,
                "target_interpolation": target_interpolation,
                "operator_match": bool(args.input_interpolation == target_interpolation),
                "analytic_test_rmse": payload["analytic_test_rmse"],
                "pseudo_target_test_rmse": payload["pseudo_target_test_rmse"],
                "pseudo_target_distortion_rmse": payload["pseudo_target_distortion_rmse"],
                "optimism_gap_mm": payload["optimism_gap_mm"],
                "training_seconds": payload["training_seconds"],
                "inference_seconds": payload["inference_seconds"],
                "best_epoch": payload["best_epoch"],
                "parameter_count": payload["parameter_count"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    import csv

    with (args.output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    manifest.update(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": float(time.perf_counter() - started),
            "summary_sha256": sha256(args.output_root / "summary.json"),
        }
    )
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
