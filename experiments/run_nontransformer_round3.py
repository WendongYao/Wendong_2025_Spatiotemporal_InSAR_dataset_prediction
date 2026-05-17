"""
Round-3 exploration focused on non-Transformer deep routes.

Revision skeleton alignment:
- Follow-up deep experiments after repairing the original sample construction
- Targeted exploration of CNN-LSTM / TCN-style architectures under the same fair split
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cg_additional_experiments import (
    run_cnn_lstm_hybrid_experiment,
    run_cnn_tcn_hybrid_experiment,
)
from revision_config import RevisionConfig


DEFAULT_MODELS = ["cnn_lstm_hybrid", "cnn_tcn_hybrid"]
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


def _summary_rows(rows: list[dict], model_names: list[str]) -> list[dict]:
    summary = []
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue
        n = len(model_rows)
        rmse_mean = sum(row["rmse"] for row in model_rows) / n
        summary.append(
            {
                "model": model_name,
                "n_seeds": n,
                "rmse_mean": rmse_mean,
                "rmse_std": (
                    sum((row["rmse"] - rmse_mean) ** 2 for row in model_rows) / max(n - 1, 1)
                )
                ** 0.5
                if n > 1
                else 0.0,
                "mae_mean": sum(row["mae"] for row in model_rows) / n,
                "r2_mean": sum(row["r2"] for row in model_rows) / n,
                "runtime_seconds_mean": sum(row["runtime_seconds"] for row in model_rows) / n,
                "peak_gpu_memory_mb_mean": sum(row["peak_gpu_memory_mb"] for row in model_rows) / n,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-3 non-Transformer deep exploration.")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--patch-min-valid-pixels", type=int, default=24)
    parser.add_argument("--patch-batch-size", type=int, default=16)
    parser.add_argument("--nontransformer-hidden-channels", type=int, default=64)
    parser.add_argument("--convlstm-num-layers", type=int, default=1)
    parser.add_argument("--split-strategy", type=str, default="spatial_tile")
    parser.add_argument("--output-root", type=str, default="revision_outputs/nontransformer_round3")
    parser.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_root_arg = Path(args.output_root)
    output_root = output_root_arg if output_root_arg.is_absolute() else project_root / output_root_arg

    model_fns = {
        "cnn_lstm_hybrid": run_cnn_lstm_hybrid_experiment,
        "cnn_tcn_hybrid": run_cnn_tcn_hybrid_experiment,
    }

    rows: list[dict] = []
    for seed in args.seeds:
        config = RevisionConfig(
            csv_path=args.csv_path,
            split_seed=seed,
            split_strategy=args.split_strategy,
            cnn_epochs=args.epochs,
            cnn_patience=args.patience,
            cnn_learning_rate=args.learning_rate,
            cnn_weight_decay=args.weight_decay,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            patch_min_valid_pixels=args.patch_min_valid_pixels,
            patch_batch_size=args.patch_batch_size,
            nontransformer_hybrid_hidden_channels=args.nontransformer_hidden_channels,
            convlstm_num_layers=args.convlstm_num_layers,
            output_root=output_root,
        )
        for model_name in args.models:
            payload = model_fns[model_name](config)
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

    summary_rows = _summary_rows(rows, list(args.models))
    _write_rows(rows, output_root / "round3_seed_level.csv")
    _write_rows(summary_rows, output_root / "round3_summary.csv")
    print(json.dumps({"output_root": str(output_root), "rows": len(rows), "summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
