"""
Collect model metrics into a paper-ready summary table.

Revision skeleton alignment:
- Section 4 / main regression table
- Section 4 / reproducibility-oriented result aggregation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect revision outputs into one summary CSV.")
    parser.add_argument("--output-root", type=str, default="revision_outputs")
    parser.add_argument("--interpolation", type=str, default="linear")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_root_arg = Path(args.output_root)
    output_root = output_root_arg if output_root_arg.is_absolute() else project_root / output_root_arg
    seed_dir = f"split_seed_{args.split_seed}"
    rows = []
    for model_name in ["cnn_lstm", "goodmodel_aligned", "lasso", "lightgbm"]:
        metrics_path = output_root / model_name / args.interpolation / seed_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_metrics(metrics_path)
        rows.append(
            {
                "model": metrics["model"],
                "interpolation_method": metrics["interpolation_method"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "test_pixels": metrics["test_pixels"],
                "device": metrics.get("device", metrics.get("device_type", "n/a")),
                "metrics_path": str(metrics_path),
            }
        )

    if not rows:
        raise FileNotFoundError("No metrics.json files were found for the requested interpolation and split seed.")

    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"main_model_summary_{args.interpolation}_seed_{args.split_seed}.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary table to {summary_path}")


if __name__ == "__main__":
    main()
