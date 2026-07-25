"""Controlled target-supervision buffer experiment for the frozen CAGEO task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_config import RevisionConfig  # noqa: E402

from raw_holdout_data import (  # noqa: E402
    RawHoldoutSpec,
    RawHoldoutTask,
    build_raw_holdout_task,
    load_forecast_columns,
)
from raw_point_supervision import run_raw_point_supervised_model, run_raw_supervised_lasso  # noqa: E402
from run_raw_holdout_pilot import SOURCE_COMMIT, _current_source_commit, _sha256  # noqa: E402


def _minimum_distance(source: np.ndarray, target: np.ndarray) -> float:
    if len(source) == 0 or len(target) == 0:
        return float("nan")
    distances, _ = cKDTree(target).query(source, k=1)
    return float(np.min(distances))


def _make_task(
    base: RawHoldoutTask,
    *,
    axis: int,
    test_side: str,
    tail_fraction: float,
    buffer_meters: float,
    condition: str,
    seed: int,
) -> tuple[RawHoldoutTask, dict[str, object], np.ndarray]:
    points = base.raw_points
    coordinate = points[:, axis]
    low, high = np.quantile(coordinate, [tail_fraction, 1.0 - tail_fraction])
    if test_side == "high":
        test_mask = coordinate >= high
        val_mask = coordinate <= low
    else:
        test_mask = coordinate <= low
        val_mask = coordinate >= high
    candidate_mask = ~(test_mask | val_mask)
    candidate_indices = np.flatnonzero(candidate_mask)
    protected_indices = np.flatnonzero(test_mask | val_mask)
    candidate_distances, _ = cKDTree(points[protected_indices]).query(
        points[candidate_indices],
        k=1,
    )
    buffered_indices = candidate_indices[candidate_distances >= buffer_meters]
    if len(buffered_indices) == 0:
        raise ValueError("The requested buffer removes every candidate training point.")
    if condition == "buffered":
        train_indices = buffered_indices
    elif condition == "matched_unbuffered":
        rng = np.random.default_rng(seed)
        train_indices = np.sort(
            rng.choice(candidate_indices, size=len(buffered_indices), replace=False)
        )
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    val_indices = np.flatnonzero(val_mask)
    test_indices = np.flatnonzero(test_mask)
    split_codes = np.full(len(points), -1, dtype=np.int8)
    split_codes[train_indices] = 0
    split_codes[val_indices] = 1
    split_codes[test_indices] = 2
    if (
        np.intersect1d(train_indices, val_indices).size
        or np.intersect1d(train_indices, test_indices).size
        or np.intersect1d(val_indices, test_indices).size
    ):
        raise AssertionError("Controlled split overlap detected.")

    train_protected_distances, _ = cKDTree(points[protected_indices]).query(
        points[train_indices],
        k=1,
    )
    axis_name = "easting" if axis == 0 else "northing"
    split_metadata: dict[str, object] = {
        "protocol": "contiguous-tail-target-buffer-v1",
        "controlled_condition": condition,
        "controlled_split_seed": int(seed),
        "controlled_axis": axis_name,
        "controlled_test_side": test_side,
        "controlled_tail_fraction": float(tail_fraction),
        "controlled_low_threshold": float(low),
        "controlled_high_threshold": float(high),
        "controlled_buffer_meters": float(buffer_meters),
        "controlled_candidate_train_points": int(len(candidate_indices)),
        "controlled_buffered_train_target_count": int(len(buffered_indices)),
        "controlled_removed_by_buffer": int(len(candidate_indices) - len(buffered_indices)),
        "controlled_train_to_protected_min_m": float(np.min(train_protected_distances)),
        "controlled_train_to_protected_median_m": float(np.median(train_protected_distances)),
        "controlled_train_to_test_min_m": _minimum_distance(points[train_indices], points[test_indices]),
        "controlled_train_to_val_min_m": _minimum_distance(points[train_indices], points[val_indices]),
        "controlled_target_supervision_only": True,
        "controlled_blind_site_forecasting": False,
        "raw_train_points": int(len(train_indices)),
        "raw_val_points": int(len(val_indices)),
        "raw_test_points": int(len(test_indices)),
        "raw_buffer_points": int(np.sum(split_codes == -1)),
        "grid_split_note": (
            "Dense input histories are retained as forecast-time covariates; "
            "primary training and evaluation use only the controlled raw target split."
        ),
    }
    metadata = {**base.metadata, **split_metadata}
    task = RawHoldoutTask(
        dense_task=base.dense_task,
        raw_points=base.raw_points,
        raw_target=base.raw_target,
        raw_split_codes=split_codes,
        raw_block_ids=base.raw_block_ids,
        easting_axis=base.easting_axis,
        northing_axis=base.northing_axis,
        train_target_source_indices=train_indices.astype(np.int64),
        val_target_source_indices=val_indices.astype(np.int64),
        test_target_indices=test_indices.astype(np.int64),
        metadata=metadata,
    )
    return task, split_metadata, candidate_distances.astype(np.float32)


def _config(
    *,
    csv_path: Path,
    output_root: Path,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> RevisionConfig:
    return RevisionConfig(
        csv_path=str(csv_path.resolve()),
        grid_size=256,
        split_seed=seed,
        split_strategy="spatial_tile",
        interpolation_method="idw",
        cnn_epochs=epochs,
        cnn_patience=patience,
        cnn_learning_rate=3e-4,
        cnn_weight_decay=1e-5,
        patch_size=16,
        patch_stride=8,
        patch_min_valid_pixels=24,
        patch_batch_size=batch_size,
        nontransformer_hybrid_hidden_channels=64,
        convlstm_hidden_dim=64,
        convlstm_num_layers=1,
        lasso_epochs=600,
        lasso_patience=60,
        lasso_learning_rate=2e-2,
        output_root=output_root,
        use_task_cache=False,
    )


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--tile", default="E32N34")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--axis", choices=["easting", "northing"], default="easting")
    parser.add_argument("--test-side", choices=["high", "low"], default="high")
    parser.add_argument("--tail-fraction", type=float, default=0.15)
    parser.add_argument("--buffer-meters", type=float, default=2000.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0.05 <= args.tail_fraction <= 0.30:
        raise ValueError("tail-fraction must be between 0.05 and 0.30.")
    if args.buffer_meters <= 0:
        raise ValueError("buffer-meters must be positive.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    base_spec = RawHoldoutSpec(
        csv_path=args.csv_path,
        tile=args.tile,
        grid_size=256,
        split_seed=args.seed,
        block_side=8,
        buffer_blocks=0,
    )
    base = build_raw_holdout_task(base_spec, cache_dir=args.cache_dir)
    loaded_points, raw_history, loaded_target = load_forecast_columns(base_spec)
    if not np.allclose(loaded_points, base.raw_points) or not np.allclose(loaded_target, base.raw_target):
        raise AssertionError("Reloaded raw arrays differ from the cached task.")

    axis = 0 if args.axis == "easting" else 1
    tasks: dict[str, RawHoldoutTask] = {}
    split_metadata: dict[str, dict[str, object]] = {}
    candidate_distances = None
    for condition in ("buffered", "matched_unbuffered"):
        task, metadata, distances = _make_task(
            base,
            axis=axis,
            test_side=args.test_side,
            tail_fraction=args.tail_fraction,
            buffer_meters=args.buffer_meters,
            condition=condition,
            seed=args.seed,
        )
        tasks[condition] = task
        split_metadata[condition] = metadata
        candidate_distances = distances
    if len(tasks["buffered"].train_target_source_indices) != len(
        tasks["matched_unbuffered"].train_target_source_indices
    ):
        raise AssertionError("Controlled conditions do not have equal training-target counts.")
    if not np.array_equal(
        tasks["buffered"].test_target_indices,
        tasks["matched_unbuffered"].test_target_indices,
    ):
        raise AssertionError("Controlled conditions do not share the same test region.")
    if not np.array_equal(
        tasks["buffered"].val_target_source_indices,
        tasks["matched_unbuffered"].val_target_source_indices,
    ):
        raise AssertionError("Controlled conditions do not share the same validation region.")

    np.savez_compressed(
        args.output_root / "controlled_split_indices.npz",
        buffered_codes=tasks["buffered"].raw_split_codes,
        matched_unbuffered_codes=tasks["matched_unbuffered"].raw_split_codes,
        candidate_protected_distances_m=candidate_distances,
    )
    manifest: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "source_git_commit": _current_source_commit(),
        "clean_baseline_commit": SOURCE_COMMIT,
        "extension_code_sha256": {
            name: _sha256(PROJECT_ROOT / "experiments_ext" / name)
            for name in [
                "support_aware_model.py",
                "raw_point_supervision.py",
                "raw_holdout_data.py",
                "run_controlled_buffer_suite.py",
            ]
        },
        "csv_path": str(args.csv_path.resolve()),
        "tile": args.tile,
        "seed": args.seed,
        "axis": args.axis,
        "test_side": args.test_side,
        "tail_fraction": args.tail_fraction,
        "buffer_meters": args.buffer_meters,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "conditions": split_metadata,
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    summary: dict[str, object] = {}
    for condition, task in tasks.items():
        condition_dir = args.output_root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        (condition_dir / "raw_task_metadata.json").write_text(
            json.dumps(task.metadata, indent=2),
            encoding="utf-8",
        )
        config = _config(
            csv_path=args.csv_path,
            output_root=condition_dir,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        lasso = run_raw_supervised_lasso(
            task,
            raw_history,
            config,
            condition_dir / "lasso_raw_supervised",
        )
        spar = run_raw_point_supervised_model(
            task,
            raw_history,
            config,
            condition_dir / "saqr_point_query",
            model_name="saqr_point_query",
            model_kind="support_aware_point_query",
            use_warm_start=True,
            formulation="normalized_residual",
            support_use_spatial_context=False,
            support_use_global_coordinates=False,
            support_use_local_coordinates=False,
            support_history_source="direct_raw_point",
        )
        summary[condition] = {
            "lasso_direct_raw_rmse": float(lasso["direct_raw_rmse"]),
            "spar_direct_raw_rmse": float(spar["direct_raw_rmse"]),
            "spar_reduction_percent": float(
                100.0
                * (float(lasso["direct_raw_rmse"]) - float(spar["direct_raw_rmse"]))
                / float(lasso["direct_raw_rmse"])
            ),
            "train_target_count": int(task.metadata["raw_train_points"]),
            "train_to_test_min_m": float(task.metadata["controlled_train_to_test_min_m"]),
            "train_to_val_min_m": float(task.metadata["controlled_train_to_val_min_m"]),
        }
        print(json.dumps({"completed_condition": condition, **summary[condition]}), flush=True)

    manifest.update(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": float(time.perf_counter() - started),
            "summary_file": str((args.output_root / "summary.json").resolve()),
        }
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
