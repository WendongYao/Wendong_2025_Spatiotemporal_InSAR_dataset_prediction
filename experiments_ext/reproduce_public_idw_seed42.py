"""Reproduce the public CAGEO IDW seed-42 LASSO/Hybrid reference run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from cg_additional_experiments import run_cnn_lstm_hybrid_experiment  # noqa: E402
from revision_config import RevisionConfig  # noqa: E402
from revision_experiments import run_lasso_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-cache-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["lasso", "cnn_lstm_hybrid"], default=["lasso", "cnn_lstm_hybrid"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    config = RevisionConfig(
        csv_path=str(args.csv_path.resolve()),
        interpolation_method="idw",
        grid_size=256,
        split_strategy="spatial_tile",
        split_seed=42,
        tile_size=32,
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
        lasso_epochs=600,
        lasso_patience=60,
        lasso_learning_rate=2e-2,
        output_root=args.output_root.resolve(),
        task_cache_root=args.task_cache_root.resolve(),
        use_task_cache=True,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    runners = {
        "lasso": lambda: run_lasso_experiment(config, interpolation_method="idw"),
        "cnn_lstm_hybrid": lambda: run_cnn_lstm_hybrid_experiment(config, interpolation_method="idw"),
    }
    for model in args.models:
        started = time.perf_counter()
        payload = runners[model]()
        payload["wrapper_wall_seconds"] = float(time.perf_counter() - started)
        results.append(payload)
        print(json.dumps({"completed": model, "rmse": payload.get("rmse"), "wall_seconds": payload["wrapper_wall_seconds"]}), flush=True)
    with (args.output_root / "R004_reference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
