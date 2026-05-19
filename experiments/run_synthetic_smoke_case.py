"""
Run a compact synthetic smoke case for the CAGEO package.

This script is a functional test, not a paper-result reproduction. It uses the
bundled synthetic CSV to validate task construction and a small one-seed subset
of the classical and deep experiment paths.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cg_additional_experiments import run_cnn_lstm_hybrid_experiment, run_persistence_baseline
from revision_config import PROJECT_ROOT, RevisionConfig
from revision_experiments import run_lasso_experiment, run_lightgbm_experiment
from revision_utils import build_dense_forecast_task


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact synthetic smoke case for the CAGEO package.")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional explicit path to a synthetic CSV.")
    parser.add_argument("--output-root", type=str, default="synthetic_smoke_outputs", help="Relative or absolute output directory.")
    parser.add_argument("--interpolation", type=str, default="idw", help="Interpolation method for the smoke case.")
    parser.add_argument("--grid-size", type=int, default=64, help="Dense-grid size for the smoke case.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed for the smoke case.")
    parser.add_argument("--epochs", type=int, default=2, help="Epoch budget for the Hybrid CNN-LSTM smoke run.")
    parser.add_argument("--patience", type=int, default=2, help="Patience for the Hybrid CNN-LSTM smoke run.")
    parser.add_argument("--patch-size", type=int, default=8, help="Patch size for the Hybrid CNN-LSTM smoke run.")
    parser.add_argument("--patch-stride", type=int, default=4, help="Patch stride for the Hybrid CNN-LSTM smoke run.")
    parser.add_argument("--patch-batch-size", type=int, default=4, help="Patch batch size for the Hybrid CNN-LSTM smoke run.")
    parser.add_argument("--lasso-epochs", type=int, default=40, help="Optimization epochs for the LASSO smoke run.")
    parser.add_argument("--num-boost-round", type=int, default=20, help="Boosting rounds for the LightGBM smoke run.")
    parser.add_argument("--early-stopping-rounds", type=int, default=5, help="Early stopping rounds for the LightGBM smoke run.")
    args = parser.parse_args()

    default_csv = PROJECT_ROOT / "examples" / "synthetic_egms_small.csv"
    csv_path = args.csv_path or str(default_csv)

    output_root_arg = Path(args.output_root)
    output_root = output_root_arg if output_root_arg.is_absolute() else PROJECT_ROOT / output_root_arg
    config = RevisionConfig(
        csv_path=csv_path,
        interpolation_method=args.interpolation,
        grid_size=args.grid_size,
        split_seed=args.seed,
        tile_size=max(args.grid_size // 4, 8),
        cnn_epochs=args.epochs,
        cnn_patience=args.patience,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_batch_size=args.patch_batch_size,
        patch_min_valid_pixels=8,
        lasso_epochs=args.lasso_epochs,
        lightgbm_num_boost_round=args.num_boost_round,
        lightgbm_early_stopping_rounds=args.early_stopping_rounds,
        output_root=output_root,
    )

    task = build_dense_forecast_task(config, interpolation_method=args.interpolation)
    run_plan = [
        ("persistence", lambda: run_persistence_baseline(config, interpolation_method=args.interpolation)),
        ("lasso", lambda: run_lasso_experiment(config, interpolation_method=args.interpolation)),
        ("lightgbm", lambda: run_lightgbm_experiment(config, interpolation_method=args.interpolation)),
        ("cnn_lstm_hybrid", lambda: run_cnn_lstm_hybrid_experiment(config)),
    ]

    rows: list[dict[str, object]] = []
    for model_name, fn in run_plan:
        payload = fn()
        rows.append(
            {
                "model": model_name,
                "interpolation_method": args.interpolation,
                "split_seed": args.seed,
                "rmse": payload["rmse"],
                "mae": payload["mae"],
                "mse": payload["mse"],
                "r2": payload["r2"],
                "runtime_seconds": payload.get("runtime_seconds"),
                "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
                "best_epoch": payload.get("best_epoch"),
            }
        )

    summary = {
        "csv_path": str(config.resolve_csv_path()),
        "output_root": str(output_root),
        "grid_size": config.grid_size,
        "interpolation_method": args.interpolation,
        "task_shape": list(task.input_maps.shape),
        "target_shape": list(task.target_map.shape),
        "train_pixels": int(task.train_mask.sum()),
        "val_pixels": int(task.val_mask.sum()),
        "test_pixels": int(task.test_mask.sum()),
        "models": rows,
    }

    _write_rows(rows, output_root / "smoke_summary.csv")
    (output_root / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
