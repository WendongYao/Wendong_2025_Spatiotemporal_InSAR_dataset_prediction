"""Apply the preregistered R094 sampler rule and freeze the R095 configuration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "R094_v23_sampler_ablation"
            / "E32N34"
            / "seed_42"
        ),
    )
    args = parser.parse_args()
    root = args.result_root.resolve()
    variants = (
        "legacy_capped_selection",
        "all_cells_uniform",
        "all_cells_density_balanced",
    )
    rows: list[dict[str, object]] = []
    for variant in variants:
        metrics_path = root / variant / f"spar_{variant}" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "sampler": variant,
                "native_cell_rmse": float(metrics["native_cell_rmse"]),
                "native_cell_mae": float(metrics["native_cell_mae"]),
                "native_cell_bias": float(metrics["native_cell_bias"]),
                "train_available_cell_count": int(
                    metrics["train_available_cell_count"]
                ),
                "train_used_cell_count": int(metrics["train_used_cell_count"]),
                "train_cell_coverage": float(metrics["train_cell_coverage"]),
                "best_epoch": int(metrics["best_epoch"]),
                "training_seconds": float(metrics["training_seconds"]),
                "inference_seconds": float(metrics["inference_seconds"]),
                "parameter_count": int(metrics["parameter_count"]),
                "metrics_sha256": sha256(metrics_path),
            }
        )
    best_rmse = min(float(row["native_cell_rmse"]) for row in rows)
    lookup = {str(row["sampler"]): row for row in rows}
    if (
        float(lookup["all_cells_uniform"]["native_cell_rmse"])
        <= 1.01 * best_rmse
    ):
        selected = "all_cells_uniform"
        reason = "uniform all-cells RMSE is within 1% of the development-best sampler"
    elif (
        float(lookup["all_cells_density_balanced"]["native_cell_rmse"])
        <= 1.01 * best_rmse
    ):
        selected = "all_cells_density_balanced"
        reason = "density-balanced all-cells RMSE is within 1% of the development-best sampler"
    else:
        selected = "legacy_capped_selection"
        reason = "neither all-cells sampler is within 1% of the development-best sampler"

    with (root / "sampler_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "sampler_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    tcn_metrics_path = root / "tcn_development" / "tcn" / "metrics.json"
    tcn_metrics = json.loads(tcn_metrics_path.read_text(encoding="utf-8"))
    model_code = PROJECT_ROOT / "experiments_ext" / "native_pointwise_v23.py"
    driver_code = PROJECT_ROOT / "experiments_ext" / "run_v23_native_suite.py"
    plan_path = (
        PROJECT_ROOT
        / "refine-logs"
        / "EXPERIMENT_PLAN_20260729_235334.md"
    )
    locked = {
        "lock_version": "cageo-v23-r095",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "decision_was_made_before_r095": True,
        "development_partition": 42,
        "locked_confirmation_partitions": [47, 48, 49, 50],
        "selected_spar_sampler": selected,
        "selection_rule": (
            "choose all_cells_uniform if RMSE <= 1.01*best; otherwise choose "
            "all_cells_density_balanced if RMSE <= 1.01*best; otherwise retain "
            "legacy_capped_selection"
        ),
        "selection_reason": reason,
        "development_sampler_results": rows,
        "spar": {
            "architecture": "history_length->96->24->64->1",
            "anchor": "fixed native-support LASSO",
            "correction_head_initialization": "zero",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "loss": "Smooth-L1 on standardized future increment",
            "maximum_epochs": 60,
            "patience": 12,
            "batch_size": 1024,
        },
        "tcn": {
            "architecture": "four causal residual blocks",
            "channels": 32,
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8],
            "dropout": 0.1,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "loss": "MSE on standardized absolute target",
            "maximum_epochs": 60,
            "patience": 12,
            "batch_size": 1024,
            "development_native_cell_rmse": float(
                tcn_metrics["native_cell_rmse"]
            ),
            "development_metrics_sha256": sha256(tcn_metrics_path),
        },
        "primary_task": {
            "tile": "E32N34",
            "history_start_col": 11,
            "history_length": 300,
            "target_col": 312,
            "grid_size_for_legacy_assignment_only": 256,
            "block_side": 8,
        },
        "code_sha256": {
            "experiments_ext/native_pointwise_v23.py": sha256(model_code),
            "experiments_ext/run_v23_native_suite.py": sha256(driver_code),
        },
        "frozen_plan_sha256": sha256(plan_path),
    }
    lock_path = root / "LOCKED_CONFIG.json"
    lock_path.write_text(json.dumps(locked, indent=2), encoding="utf-8")
    lock_digest = sha256(lock_path)
    (root / "LOCKED_CONFIG.sha256").write_text(
        f"{lock_digest}  LOCKED_CONFIG.json\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "selected_sampler": selected,
                "best_development_rmse": best_rmse,
                "locked_config_sha256": lock_digest,
            }
        )
    )


if __name__ == "__main__":
    main()
