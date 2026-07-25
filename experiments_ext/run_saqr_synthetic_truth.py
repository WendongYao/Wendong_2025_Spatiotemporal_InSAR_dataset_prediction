"""Evaluate frozen SAQR-Net on analytic truth at irregular measurement support."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_config import RevisionConfig  # noqa: E402

from raw_point_supervision import run_raw_point_supervised_model, run_raw_supervised_lasso
from synthetic_truth_data import (
    SyntheticTruthSpec,
    analytic_truth_diagnostics,
    build_synthetic_support_holdout,
)


SOURCE_COMMIT = "ffc1d4e8eb09c86ac81faa09ff662868b7494162"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=["seasonal_trend", "localized_acceleration", "moving_front", "composite"],
        default="composite",
    )
    parser.add_argument("--input-interpolation", choices=["idw", "linear", "nearest"], default="idw")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--support-points", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    started = time.perf_counter()
    spec = SyntheticTruthSpec(
        scenario=args.scenario,
        grid_size=args.grid_size,
        support_points=args.support_points,
        observation_noise_std=args.noise,
        seed=args.seed,
        split_seed=args.seed,
        tile_size=max(8, args.grid_size // 8),
        input_interpolation=args.input_interpolation,
    )
    raw_task, raw_history, support = build_synthetic_support_holdout(spec)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = RevisionConfig(
        csv_path="synthetic://analytic-irregular-support",
        grid_size=args.grid_size,
        split_seed=args.seed,
        split_strategy="spatial_tile",
        interpolation_method=args.input_interpolation,
        cnn_epochs=args.epochs,
        cnn_patience=args.patience,
        cnn_learning_rate=3e-4,
        cnn_weight_decay=1e-5,
        patch_size=16,
        patch_stride=8,
        patch_min_valid_pixels=8,
        patch_batch_size=16,
        nontransformer_hybrid_hidden_channels=64,
        convlstm_hidden_dim=64,
        convlstm_num_layers=1,
        lasso_epochs=600,
        lasso_patience=60,
        lasso_learning_rate=2e-2,
        output_root=args.output_root,
        use_task_cache=False,
    )
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "source_git_commit": SOURCE_COMMIT,
        "extension_code_sha256": {
            name: _sha256(PROJECT_ROOT / "experiments_ext" / name)
            for name in [
                "support_aware_model.py",
                "raw_point_supervision.py",
                "synthetic_truth_data.py",
                "run_saqr_synthetic_truth.py",
            ]
        },
        **raw_task.metadata,
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.save(args.output_root / "analytic_target.npy", raw_task.dense_task.target_map.astype(np.float32))
    np.save(args.output_root / "support_points.npy", support)

    results = []
    for name in ["lasso_raw_supervised", "saqr_point_query"]:
        model_dir = args.output_root / name
        if name == "lasso_raw_supervised":
            payload = run_raw_supervised_lasso(raw_task, raw_history, config, model_dir)
        else:
            payload = run_raw_point_supervised_model(
                raw_task,
                raw_history,
                config,
                model_dir,
                model_name=name,
                model_kind="support_aware_point_query",
                use_warm_start=True,
                formulation="normalized_residual",
                support_use_spatial_context=False,
                support_use_global_coordinates=False,
                support_use_local_coordinates=False,
                support_history_source="direct_raw_point",
            )
        prediction = np.load(model_dir / "prediction_grid.npy")
        dense_test = raw_task.dense_task.test_mask
        dense_truth = raw_task.dense_task.target_map
        payload.update(analytic_truth_diagnostics(prediction, raw_task, support_points=support))
        payload["dense_analytic_test_rmse"] = float(
            np.sqrt(np.mean((prediction[dense_test] - dense_truth[dense_test]) ** 2))
        )
        (model_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
        )
        results.append(payload)
        print(
            json.dumps(
                {
                    "completed": name,
                    "direct_raw_rmse": payload["direct_raw_rmse"],
                    "dense_analytic_test_rmse": payload["dense_analytic_test_rmse"],
                }
            ),
            flush=True,
        )
    (args.output_root / "summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=True), encoding="utf-8"
    )
    manifest.update(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": float(time.perf_counter() - started),
        }
    )
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
