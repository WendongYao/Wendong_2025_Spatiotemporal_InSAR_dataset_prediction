"""Run the first measurement-grounded CAGEO pilot on one tile and seed."""

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
SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_config import RevisionConfig  # noqa: E402

from raw_holdout_data import (
    RawHoldoutSpec,
    build_raw_holdout_task,
    load_forecast_columns,
    load_quality_rmse,
    sample_grid_at_raw_points,
)
from raw_holdout_models import run_lasso, run_patch_model, run_persistence, run_target_idw_diagnostic
from raw_point_supervision import run_raw_point_supervised_model, run_raw_supervised_lasso


SOURCE_COMMIT = "ffc1d4e8eb09c86ac81faa09ff662868b7494162"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--block-side", type=int, default=8)
    parser.add_argument("--buffer-blocks", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--convlstm-num-layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--hybrid-no-warm-start", action="store_true")
    parser.add_argument("--hybrid-disable-recent-gate", action="store_true")
    parser.add_argument("--hybrid-disable-spatial-correction", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="+", choices=["persistence", "target_idw_diagnostic", "lasso", "lasso_raw_supervised", "cnn_lstm_hybrid", "cnn_lstm_raw_supervised", "cnn_lstm_raw_quality_weighted", "conv_lstm_raw_supervised", "conv_lstm_raw_residual", "conv_lstm_absolute_supervised", "simvp_raw_supervised", "conv_lstm_residual", "simvp_style_residual", "saqr_point_query", "saqr_with_context", "saqr_with_global_coord", "saqr_grid_history", "saqr_no_anchor"], default=["persistence", "lasso", "cnn_lstm_hybrid"])
    args = parser.parse_args()

    spec = RawHoldoutSpec(
        csv_path=args.csv_path,
        tile=args.tile,
        grid_size=args.grid_size,
        split_seed=args.seed,
        block_side=args.block_side,
        buffer_blocks=args.buffer_blocks,
    )
    raw_task = build_raw_holdout_task(spec, cache_dir=args.cache_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_manifest: dict[str, object] = {
        "started_utc": started_utc,
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "source_git_commit": SOURCE_COMMIT,
        "extension_code_sha256": {
            name: _sha256(PROJECT_ROOT / "experiments_ext" / name)
            for name in [
                "support_aware_model.py",
                "raw_point_supervision.py",
                "run_raw_holdout_pilot.py",
            ]
        },
        "tile": args.tile,
        "seed": args.seed,
        "csv_path": str(args.csv_path.resolve()),
        "output_root": str(args.output_root.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "models": list(args.models),
        "grid_size": args.grid_size,
        "block_side": args.block_side,
        "buffer_blocks": args.buffer_blocks,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "convlstm_num_layers": args.convlstm_num_layers,
        "hybrid_no_warm_start": args.hybrid_no_warm_start,
        "hybrid_disable_recent_gate": args.hybrid_disable_recent_gate,
        "hybrid_disable_spatial_correction": args.hybrid_disable_spatial_correction,
        "resume": args.resume,
    }
    with (args.output_root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)
    with (args.output_root / "raw_task_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(raw_task.metadata, handle, indent=2)

    config = RevisionConfig(
        csv_path=str(args.csv_path.resolve()),
        grid_size=args.grid_size,
        split_seed=args.seed,
        split_strategy="spatial_tile",
        interpolation_method="idw",
        cnn_epochs=args.epochs,
        cnn_patience=args.patience,
        cnn_learning_rate=3e-4,
        cnn_weight_decay=1e-5,
        patch_size=16,
        patch_stride=8,
        patch_min_valid_pixels=24,
        patch_batch_size=args.batch_size,
        nontransformer_hybrid_hidden_channels=64,
        convlstm_hidden_dim=64,
        convlstm_num_layers=args.convlstm_num_layers,
        lasso_epochs=600,
        lasso_patience=60,
        lasso_learning_rate=2e-2,
        output_root=args.output_root,
        use_task_cache=False,
    )
    results: list[dict[str, object]] = []
    raw_history = None
    grid_sampled_history = None
    raw_quality = None
    for model in args.models:
        model_dir = args.output_root / model
        metrics_path = model_dir / "metrics.json"
        if args.resume and metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            results.append(payload)
            print(json.dumps({"resumed": model, "raw_test_rmse": payload["rmse"]}), flush=True)
            continue
        if model == "persistence":
            payload = run_persistence(raw_task, model_dir)
        elif model == "target_idw_diagnostic":
            payload = run_target_idw_diagnostic(raw_task, model_dir)
        elif model == "lasso":
            payload = run_lasso(raw_task, config, model_dir)
        elif model == "lasso_raw_supervised":
            if raw_history is None:
                loaded_points, raw_history, loaded_target = load_forecast_columns(spec)
                if not np.allclose(loaded_points, raw_task.raw_points) or not np.allclose(loaded_target, raw_task.raw_target):
                    raise AssertionError("Reloaded raw arrays differ from cached task.")
            payload = run_raw_supervised_lasso(raw_task, raw_history, config, model_dir)
        elif model == "cnn_lstm_hybrid":
            payload = run_patch_model(
                raw_task,
                config,
                model_dir,
                model_name=model,
                model_kind="cnn_lstm_hybrid",
                use_warm_start=not args.hybrid_no_warm_start,
                disable_recent_gate=args.hybrid_disable_recent_gate,
                disable_spatial_correction=args.hybrid_disable_spatial_correction,
            )
        elif model in {"cnn_lstm_raw_supervised", "cnn_lstm_raw_quality_weighted", "conv_lstm_raw_supervised", "conv_lstm_raw_residual", "conv_lstm_absolute_supervised", "simvp_raw_supervised", "saqr_point_query", "saqr_with_context", "saqr_with_global_coord", "saqr_grid_history", "saqr_no_anchor"}:
            if raw_history is None:
                loaded_points, raw_history, loaded_target = load_forecast_columns(spec)
                if not np.allclose(loaded_points, raw_task.raw_points):
                    raise AssertionError("Reloaded raw point order differs from the cached task.")
                if not np.allclose(loaded_target, raw_task.raw_target):
                    raise AssertionError("Reloaded raw target differs from the cached task.")
            if model == "cnn_lstm_raw_quality_weighted" and raw_quality is None:
                raw_quality = load_quality_rmse(spec)
            support_variant = model.startswith("saqr_")
            model_history = raw_history
            if model == "saqr_grid_history":
                if grid_sampled_history is None:
                    all_indices = np.arange(len(raw_task.raw_points), dtype=np.int64)
                    grid_sampled_history = np.stack(
                        [
                            sample_grid_at_raw_points(frame, raw_task, all_indices)
                            for frame in raw_task.dense_task.input_maps
                        ],
                        axis=1,
                    ).astype(np.float32)
                model_history = grid_sampled_history
            payload = run_raw_point_supervised_model(
                raw_task,
                model_history,
                config,
                model_dir,
                model_name=model,
                model_kind=(
                    "support_aware_point_query"
                    if support_variant
                    else "cnn_lstm_hybrid"
                    if model in {"cnn_lstm_raw_supervised", "cnn_lstm_raw_quality_weighted"}
                    else "conv_lstm_residual"
                    if model in {"conv_lstm_raw_supervised", "conv_lstm_raw_residual", "conv_lstm_absolute_supervised"}
                    else "simvp_style_residual"
                ),
                use_warm_start=(
                    model in {"cnn_lstm_raw_supervised", "cnn_lstm_raw_quality_weighted", "saqr_point_query", "saqr_with_context", "saqr_with_global_coord", "saqr_grid_history"}
                    and not args.hybrid_no_warm_start
                ),
                formulation=(
                    "raw_residual"
                    if model == "conv_lstm_raw_residual"
                    else "normalized_absolute"
                    if model == "conv_lstm_absolute_supervised"
                    else "normalized_residual"
                ),
                raw_quality_rmse=(
                    raw_quality
                    if model == "cnn_lstm_raw_quality_weighted"
                    else None
                ),
                disable_recent_gate=args.hybrid_disable_recent_gate,
                disable_spatial_correction=args.hybrid_disable_spatial_correction,
                support_use_spatial_context=model == "saqr_with_context",
                support_use_global_coordinates=model in {"saqr_with_context", "saqr_with_global_coord"},
                support_use_local_coordinates=model == "saqr_with_context",
                support_history_source=(
                    "idw_grid_sampled_at_query" if model == "saqr_grid_history" else "direct_raw_point"
                ),
            )
        elif model == "conv_lstm_residual":
            payload = run_patch_model(
                raw_task,
                config,
                model_dir,
                model_name=model,
                model_kind="conv_lstm_residual",
                use_warm_start=False,
            )
        elif model == "simvp_style_residual":
            payload = run_patch_model(
                raw_task,
                config,
                model_dir,
                model_name=model,
                model_kind="simvp_style_residual",
                use_warm_start=False,
            )
        else:
            raise AssertionError(model)
        results.append(payload)
        print(json.dumps({"completed": model, "raw_test_rmse": payload["rmse"]}), flush=True)
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)
    run_manifest.update(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": float(time.perf_counter() - run_started),
            "result_files": [str((args.output_root / model / "metrics.json").resolve()) for model in args.models],
            "summary_file": str((args.output_root / "summary.json").resolve()),
        }
    )
    with (args.output_root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)


if __name__ == "__main__":
    main()
