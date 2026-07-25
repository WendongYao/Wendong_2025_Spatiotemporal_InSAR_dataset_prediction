"""Stratify raw-point forecast errors by InSAR support and measurement regime."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from raw_holdout_data import (
    RawHoldoutSpec,
    build_raw_holdout_task,
    load_quality_rmse,
    sample_grid_at_raw_points,
)


def metric_row(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = prediction - truth
    return {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "p95_absolute_error": float(np.quantile(np.abs(residual), 0.95)),
    }


def quantile_groups(values: np.ndarray, quantiles: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)):
    edges = np.quantile(values, quantiles)
    edges = np.maximum.accumulate(edges)
    for index in range(len(edges) - 1):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == len(edges) - 2:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values >= lower) & (values < upper)
        if mask.any():
            yield f"q{int(quantiles[index] * 100):02d}_q{int(quantiles[index + 1] * 100):02d}", lower, upper, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--buffer-blocks", type=int, default=0)
    args = parser.parse_args()
    if len(args.prediction) != len(args.label):
        raise ValueError("--prediction and --label counts differ")

    spec = RawHoldoutSpec(
        csv_path=args.csv_path,
        tile=args.tile,
        grid_size=256,
        split_seed=args.seed,
        block_side=8,
        buffer_blocks=args.buffer_blocks,
    )
    task = build_raw_holdout_task(spec, cache_dir=args.cache_dir)
    test = task.test_target_indices
    truth = task.raw_target[test]
    quality = load_quality_rmse(spec)[test]
    distance_to_train_target = cKDTree(task.raw_points[task.train_target_source_indices]).query(
        task.raw_points[test],
        k=1,
    )[0]
    grid_rows = np.clip(
        np.rint(
            (task.raw_points[:, 0] - task.easting_axis[0])
            / (task.easting_axis[-1] - task.easting_axis[0])
            * (len(task.easting_axis) - 1)
        ).astype(int),
        0,
        len(task.easting_axis) - 1,
    )
    grid_cols = np.clip(
        np.rint(
            (task.raw_points[:, 1] - task.northing_axis[0])
            / (task.northing_axis[-1] - task.northing_axis[0])
            * (len(task.northing_axis) - 1)
        ).astype(int),
        0,
        len(task.northing_axis) - 1,
    )
    cell_ids = grid_rows * len(task.northing_axis) + grid_cols
    _, inverse, counts = np.unique(cell_ids[test], return_inverse=True, return_counts=True)
    cell_occupancy = counts[inverse].astype(np.float64)
    features = {
        "measurement_rmse": quality.astype(np.float64),
        "absolute_target": np.abs(truth).astype(np.float64),
        "distance_to_train_target": distance_to_train_target.astype(np.float64),
        "cell_occupancy": cell_occupancy,
    }

    rows: list[dict[str, object]] = []
    for label, prediction_path in zip(args.label, args.prediction, strict=True):
        prediction_grid = np.load(prediction_path)
        prediction = sample_grid_at_raw_points(prediction_grid, task, test)
        rows.append({"model": label, "feature": "all", "bin": "all", **metric_row(truth, prediction)})
        for feature_name, values in features.items():
            for bin_name, lower, upper, mask in quantile_groups(values):
                rows.append(
                    {
                        "model": label,
                        "feature": feature_name,
                        "bin": bin_name,
                        "lower": lower,
                        "upper": upper,
                        **metric_row(truth[mask], prediction[mask]),
                    }
                )
        extreme_threshold = float(np.quantile(np.abs(truth), 0.90))
        extreme = np.abs(truth) >= extreme_threshold
        rows.append(
            {
                "model": label,
                "feature": "absolute_target",
                "bin": "top_10pct",
                "lower": extreme_threshold,
                "upper": float(np.max(np.abs(truth))),
                **metric_row(truth[extreme], prediction[extreme]),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_root / "stratified_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "stratified_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
