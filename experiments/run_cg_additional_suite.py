"""
Run the additional Computers & Geosciences experiment suite described in the DOCX plan.

Revision skeleton alignment:
- E0 to E7, E10, and E11 automation entry point
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cg_additional_experiments import (
    CG_OUTPUT_ROOT,
    CG_SUITE_ROOT,
    MANDATORY_SEEDS,
    PRIMARY_MODELS,
    run_interpretability_suite,
    run_mask_ablation_suite,
    run_metric_sanity_audit,
    run_paired_model_stats,
    run_primary_model_suite,
    run_reproducibility_pack,
    run_resolution_scaling_suite,
    run_split_comparison_suite,
    run_interpolation_sensitivity_suite,
    summarize_seed_table,
)
from revision_config import PROJECT_ROOT, RevisionConfig


def _base_config_from_args(args) -> RevisionConfig:
    return RevisionConfig(
        csv_path=args.csv_path,
        interpolation_method=args.interpolation,
        grid_size=args.grid_size,
        cnn_epochs=args.epochs,
        cnn_patience=args.patience,
        lasso_epochs=args.lasso_epochs,
        lightgbm_device_type=args.lightgbm_device_type,
        lightgbm_num_boost_round=args.num_boost_round,
        lightgbm_early_stopping_rounds=args.early_stopping_rounds,
        random_forest_n_estimators=args.random_forest_n_estimators,
        tile_size=args.tile_size,
    )


def _known_experiment_roots() -> list[Path]:
    if not CG_SUITE_ROOT.exists():
        return []
    return [path for path in CG_SUITE_ROOT.iterdir() if path.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DOCX-aligned additional experiment suite.")
    parser.add_argument("--phase", type=str, default="all", help="all, primary, mask, interpolation, split, scaling, interpretability, audit, repro")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional explicit path to the EGMS CSV.")
    parser.add_argument("--interpolation", type=str, default="linear", help="Default interpolation method for non-sensitivity runs.")
    parser.add_argument("--grid-size", type=int, default=256, help="Default grid size.")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for torch backends.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience for torch backends.")
    parser.add_argument("--lasso-epochs", type=int, default=600, help="Optimization epochs for the torch LASSO baseline.")
    parser.add_argument("--num-boost-round", type=int, default=150, help="Maximum LightGBM boosting rounds.")
    parser.add_argument("--early-stopping-rounds", type=int, default=20, help="Validation patience for LightGBM.")
    parser.add_argument("--random-forest-n-estimators", type=int, default=100, help="Tree count for the RandomForest baseline.")
    parser.add_argument("--lightgbm-device-type", type=str, default="auto", help="auto, cpu, gpu, cuda")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile size for spatial tile split.")
    args = parser.parse_args()

    base_config = _base_config_from_args(args)
    phase = args.phase.lower()
    completed = {}

    if phase in {"all", "primary"}:
        root = run_primary_model_suite(base_config, seeds=MANDATORY_SEEDS, model_names=PRIMARY_MODELS)
        summary = summarize_seed_table(
            seed_level_csv=root / "seed_level_results.csv",
            group_fields=["model", "split_strategy"],
            metric_fields=["rmse", "mae", "mse", "r2", "runtime_seconds", "peak_gpu_memory_mb"],
            output_csv=root / "primary_multiseed_summary.csv",
        )
        stats_payload = run_paired_model_stats(
            seed_level_csv=root / "seed_level_results.csv",
            candidate_models=PRIMARY_MODELS,
            proposed_model="cnn_lstm_maskaware",
            output_json=root / "paired_model_stats.json",
        )
        completed["primary"] = {
            "root": str(root),
            "summary_rows": len(summary),
            "strongest_baseline": stats_payload["strongest_baseline"],
        }

    if phase in {"all", "mask"}:
        root = run_mask_ablation_suite(base_config, seeds=MANDATORY_SEEDS)
        summary = summarize_seed_table(
            seed_level_csv=root / "seed_level_mask_ablation.csv",
            group_fields=["variant"],
            metric_fields=["rmse", "mae", "mse", "r2", "full_grid_rmse", "runtime_seconds"],
            output_csv=root / "mask_ablation_summary.csv",
        )
        completed["mask"] = {"root": str(root), "summary_rows": len(summary)}

    if phase in {"all", "interpolation"}:
        root = run_interpolation_sensitivity_suite(base_config, seeds=MANDATORY_SEEDS)
        forecast_summary = summarize_seed_table(
            seed_level_csv=root / "forecast_metric_deltas.csv",
            group_fields=["method", "model"],
            metric_fields=["rmse", "mae", "mse", "r2", "runtime_seconds"],
            output_csv=root / "forecast_metric_summary.csv",
        )
        holdout_summary = summarize_seed_table(
            seed_level_csv=root / "point_holdout_interpolation_error.csv",
            group_fields=["method"],
            metric_fields=["point_holdout_rmse", "point_holdout_mae", "point_holdout_mse"],
            output_csv=root / "point_holdout_interpolation_summary.csv",
        )
        completed["interpolation"] = {
            "root": str(root),
            "forecast_summary_rows": len(forecast_summary),
            "holdout_summary_rows": len(holdout_summary),
        }

    if phase in {"all", "split"}:
        root = run_split_comparison_suite(base_config, seeds=MANDATORY_SEEDS)
        summary = summarize_seed_table(
            seed_level_csv=root / "split_comparison_seed_level.csv",
            group_fields=["model", "split_strategy"],
            metric_fields=["rmse", "mae", "mse", "r2", "inflation_optimism_pct"],
            output_csv=root / "split_comparison_summary.csv",
        )
        completed["split"] = {"root": str(root), "summary_rows": len(summary)}

    if phase in {"all", "scaling"}:
        root = run_resolution_scaling_suite(base_config, seeds=MANDATORY_SEEDS)
        summary = summarize_seed_table(
            seed_level_csv=root / "resolution_scaling_seed_level.csv",
            group_fields=["grid_size", "model"],
            metric_fields=["rmse", "mae", "mse", "r2", "runtime_seconds", "peak_gpu_memory_mb"],
            output_csv=root / "resolution_scaling_summary.csv",
        )
        completed["scaling"] = {"root": str(root), "summary_rows": len(summary)}

    if phase in {"all", "interpretability"}:
        root = run_interpretability_suite(base_config, seed=42)
        completed["interpretability"] = {"root": str(root)}

    if phase in {"all", "audit"}:
        audit_root = CG_OUTPUT_ROOT
        audit_csv = run_metric_sanity_audit(_known_experiment_roots(), audit_root)
        completed["audit"] = {"csv": str(audit_csv)}

    if phase in {"all", "repro"}:
        repro_root = run_reproducibility_pack(base_config, _known_experiment_roots())
        completed["repro"] = {"root": str(repro_root)}

    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
