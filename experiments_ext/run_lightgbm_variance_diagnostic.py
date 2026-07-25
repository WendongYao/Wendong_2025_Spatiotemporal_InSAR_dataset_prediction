"""Reproduce and explain LightGBM split variance on the public IDW task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_utils import _build_spatial_tile_split_masks  # noqa: E402


def locate_idw_cache(cache_dir: Path) -> Path:
    matches: list[Path] = []
    for path in cache_dir.glob("dense_task_*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            method = str(payload["interpolation_method"].reshape(-1)[0])
            if method == "idw" and payload["target_map"].shape == (256, 256):
                matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one 256x256 IDW cache in {cache_dir}, found {matches}")
    return matches[0]


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - truth
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan"),
    }


def tail_diagnostics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    train_min: float,
    train_max: float,
) -> dict[str, float | int]:
    residual = prediction - truth
    squared = residual**2
    order = np.argsort(np.abs(residual))[::-1]
    top_count = max(1, int(np.ceil(0.01 * len(residual))))
    outside = (truth < train_min) | (truth > train_max)
    return {
        "outside_train_range_count": int(outside.sum()),
        "outside_train_range_fraction": float(outside.mean()),
        "outside_train_range_rmse": float(np.sqrt(np.mean(squared[outside]))) if outside.any() else float("nan"),
        "inside_train_range_rmse": float(np.sqrt(np.mean(squared[~outside]))) if (~outside).any() else float("nan"),
        "top_1pct_absolute_errors_mse_fraction": float(squared[order[:top_count]].sum() / squared.sum()),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "p99_absolute_error": float(np.quantile(np.abs(residual), 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "source" / "experiments" / "revision_outputs" / "_task_cache",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--fit-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--device-type", choices=["cpu", "gpu", "cuda"], default="cpu")
    args = parser.parse_args()

    import lightgbm as lgb

    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_path = locate_idw_cache(args.cache_dir)
    with np.load(cache_path, allow_pickle=False) as payload:
        input_maps = payload["input_maps"].astype(np.float32)
        target_map = payload["target_map"].astype(np.float32)
        eligible = payload["eligible_mask"].astype(bool)
    X_all = input_maps[:, eligible].T
    y_all = target_map[eligible]
    flat_eligible = np.flatnonzero(eligible.ravel())
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        train_mask, val_mask, test_mask = _build_spatial_tile_split_masks(
            eligible_mask=eligible,
            seed=seed,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            tile_size=32,
        )
        flat_train = train_mask.ravel()[flat_eligible]
        flat_val = val_mask.ravel()[flat_eligible]
        flat_test = test_mask.ravel()[flat_eligible]
        y_train = y_all[flat_train]
        y_test = y_all[flat_test]
        row: dict[str, object] = {
            "seed": seed,
            "train_count": int(flat_train.sum()),
            "val_count": int(flat_val.sum()),
            "test_count": int(flat_test.sum()),
            "train_min": float(y_train.min()),
            "train_max": float(y_train.max()),
            "train_mean": float(y_train.mean()),
            "train_std": float(y_train.std()),
            "test_min": float(y_test.min()),
            "test_max": float(y_test.max()),
            "test_mean": float(y_test.mean()),
            "test_std": float(y_test.std()),
            "test_p01": float(np.quantile(y_test, 0.01)),
            "test_p99": float(np.quantile(y_test, 0.99)),
        }
        if seed in args.fit_seeds:
            dtrain = lgb.Dataset(X_all[flat_train], label=y_train)
            dvalid = lgb.Dataset(X_all[flat_val], label=y_all[flat_val], reference=dtrain)
            params = {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "seed": seed,
                "verbosity": -1,
                "device_type": args.device_type,
            }
            started = time.perf_counter()
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=300,
                valid_sets=[dvalid],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)],
            )
            prediction = model.predict(X_all[flat_test], num_iteration=model.best_iteration).astype(np.float32)
            row.update(regression_metrics(y_test, prediction))
            row.update(
                tail_diagnostics(
                    y_test,
                    prediction,
                    train_min=float(y_train.min()),
                    train_max=float(y_train.max()),
                )
            )
            row.update(
                {
                    "best_iteration": int(model.best_iteration),
                    "training_and_inference_seconds": float(time.perf_counter() - started),
                    "device_type": args.device_type,
                }
            )
            np.savez_compressed(
                args.output_root / f"seed_{seed}_test_predictions.npz",
                flat_indices=flat_eligible[flat_test],
                truth=y_test,
                prediction=prediction,
                residual=prediction - y_test,
            )
            model.save_model(str(args.output_root / f"seed_{seed}_lightgbm_model.txt"))
        rows.append(row)
        print(json.dumps(row, allow_nan=True), flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_root / "lightgbm_variance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "lightgbm_variance.json").write_text(
        json.dumps({"cache_path": str(cache_path.resolve()), "rows": rows}, indent=2, allow_nan=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
