"""
Rerun the repaired deep-learning backends after fixing the sample construction.

Revision skeleton alignment:
- Deep backend correction after verifying the previous loader/architecture mismatch
- Multi-seed rerun for fair comparison against existing baselines
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cg_additional_experiments import run_cnn_tcn_experiment, run_cnnlstm_maskaware_experiment
from revision_config import RevisionConfig


DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
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
    parser = argparse.ArgumentParser(description="Rerun repaired deep-learning models.")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    parser.add_argument("--split-strategy", type=str, default="spatial_tile")
    parser.add_argument("--output-root", type=str, default="revision_outputs/deep_model_repair/primary_multiseed")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_root_arg = Path(args.output_root)
    output_root = output_root_arg if output_root_arg.is_absolute() else project_root / output_root_arg

    rows: list[dict] = []
    summary_rows: list[dict] = []
    model_fns = {
        "cnn_lstm_maskaware": run_cnnlstm_maskaware_experiment,
        "cnn_tcn": run_cnn_tcn_experiment,
    }

    for seed in args.seeds:
        config = RevisionConfig(
            csv_path=args.csv_path,
            split_seed=seed,
            split_strategy=args.split_strategy,
            cnn_epochs=args.epochs,
            cnn_patience=args.patience,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            patch_batch_size=args.patch_batch_size,
            output_root=output_root,
        )
        for model_name, fn in model_fns.items():
            payload = fn(config)
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "rmse": payload["rmse"],
                    "mae": payload["mae"],
                    "mse": payload["mse"],
                    "r2": payload["r2"],
                    "runtime_seconds": payload.get("runtime_seconds"),
                    "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
                    "best_epoch": payload.get("best_epoch"),
                    "train_patch_count": payload.get("train_patch_count"),
                    "val_patch_count": payload.get("val_patch_count"),
                    "metrics_path": str(output_root / model_name / "linear" / f"split_seed_{seed}" / "metrics.json"),
                }
            )

    for model_name in model_fns:
        model_rows = [row for row in rows if row["model"] == model_name]
        n = len(model_rows)
        summary_rows.append(
            {
                "model": model_name,
                "n_seeds": n,
                "rmse_mean": sum(row["rmse"] for row in model_rows) / n,
                "rmse_std": (sum((row["rmse"] - (sum(r["rmse"] for r in model_rows) / n)) ** 2 for row in model_rows) / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0,
                "mae_mean": sum(row["mae"] for row in model_rows) / n,
                "r2_mean": sum(row["r2"] for row in model_rows) / n,
                "runtime_seconds_mean": sum(row["runtime_seconds"] for row in model_rows) / n,
                "peak_gpu_memory_mb_mean": sum(row["peak_gpu_memory_mb"] for row in model_rows) / n,
            }
        )

    _write_rows(rows, output_root / "deep_repair_seed_level.csv")
    _write_rows(summary_rows, output_root / "deep_repair_summary.csv")
    print(json.dumps({"output_root": str(output_root), "rows": len(rows), "summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
