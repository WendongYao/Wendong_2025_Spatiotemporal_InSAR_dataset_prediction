"""Run models against analytic (non-interpolated) dense deformation truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_config import RevisionConfig  # noqa: E402

from raw_holdout_models import run_lasso, run_patch_model, run_persistence
from synthetic_truth_data import (
    SyntheticTruthSpec,
    analytic_truth_diagnostics,
    build_synthetic_truth_task,
    sample_irregular_support,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenario", choices=["seasonal_trend", "localized_acceleration", "moving_front", "composite"], default="composite")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--support-points", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--input-interpolation", choices=["idw", "linear", "nearest"], default="idw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--models", nargs="+", choices=["persistence", "lasso", "cnn_lstm_hybrid", "conv_lstm_residual", "simvp_style_residual"], default=["persistence", "lasso", "cnn_lstm_hybrid"])
    args = parser.parse_args()

    tile_size = max(8, args.grid_size // 8)
    spec = SyntheticTruthSpec(
        scenario=args.scenario,
        grid_size=args.grid_size,
        support_points=args.support_points,
        observation_noise_std=args.noise,
        seed=args.seed,
        split_seed=args.split_seed,
        tile_size=tile_size,
        input_interpolation=args.input_interpolation,
    )
    raw_task = build_synthetic_truth_task(spec)
    support = sample_irregular_support(spec)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "synthetic_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(raw_task.metadata, handle, indent=2)
    np.save(args.output_root / "analytic_target.npy", raw_task.dense_task.target_map.astype(np.float32))
    np.save(args.output_root / "last_input_idw.npy", raw_task.dense_task.input_maps[-1].astype(np.float32))
    np.save(args.output_root / "support_points.npy", support.astype(np.float64))

    config = RevisionConfig(
        csv_path=str(PROJECT_ROOT / "source" / "experiments" / "examples" / "synthetic_egms_small.csv"),
        grid_size=args.grid_size,
        split_seed=args.split_seed,
        split_strategy="spatial_tile",
        cnn_epochs=args.epochs,
        cnn_patience=args.patience,
        cnn_learning_rate=3e-4,
        cnn_weight_decay=1e-5,
        patch_size=16,
        patch_stride=8,
        patch_min_valid_pixels=24,
        patch_batch_size=16,
        nontransformer_hybrid_hidden_channels=64,
        convlstm_num_layers=1,
        lasso_epochs=300,
        lasso_patience=40,
        lasso_learning_rate=2e-2,
        output_root=args.output_root,
        use_task_cache=False,
    )
    results: list[dict[str, object]] = []
    for model in args.models:
        model_dir = args.output_root / model
        if model == "persistence":
            payload = run_persistence(raw_task, model_dir)
        elif model == "lasso":
            payload = run_lasso(raw_task, config, model_dir)
        elif model == "cnn_lstm_hybrid":
            payload = run_patch_model(raw_task, config, model_dir, model_name=model, model_kind="cnn_lstm_hybrid", use_warm_start=True)
        elif model == "conv_lstm_residual":
            payload = run_patch_model(raw_task, config, model_dir, model_name=model, model_kind="conv_lstm_residual", use_warm_start=False)
        elif model == "simvp_style_residual":
            payload = run_patch_model(raw_task, config, model_dir, model_name=model, model_kind="simvp_style_residual", use_warm_start=False)
        else:
            raise AssertionError(model)
        prediction = np.load(model_dir / "prediction_grid.npy")
        payload.update(analytic_truth_diagnostics(prediction, raw_task, support_points=support))
        with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=True)
        results.append(payload)
        print(json.dumps({"completed": model, "truth_rmse": payload["rmse"], "gradient_rmse": payload["gradient_vector_rmse"]}), flush=True)
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()
