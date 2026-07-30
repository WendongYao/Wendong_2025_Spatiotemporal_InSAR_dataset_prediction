"""Run locked native-support CAGEO v2.3 pointwise experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
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

from native_pointwise_v23 import (  # noqa: E402
    SAMPLERS,
    run_direct_spar,
    run_native_tcn,
)
from native_support_baselines import (  # noqa: E402
    load_forecast_dates,
    run_native_dlinear,
    run_native_persistence,
)
from raw_holdout_data import (  # noqa: E402
    RawHoldoutSpec,
    build_raw_holdout_task,
    load_forecast_columns,
)
from raw_point_supervision import run_raw_supervised_lasso  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT / "source",
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--tile", default="E32N34")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--history-start-col", type=int, default=11)
    parser.add_argument("--history-length", type=int, default=300)
    parser.add_argument("--target-col", type=int, default=312)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--block-side", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dlinear-batch-size", type=int, default=4096)
    parser.add_argument("--spar-sampler", choices=SAMPLERS, default="all_cells_uniform")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["persistence", "dlinear", "lasso", "tcn", "spar"],
        default=["persistence", "dlinear", "lasso", "tcn", "spar"],
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    code_files = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "experiments_ext" / "native_pointwise_v23.py",
        PROJECT_ROOT / "experiments_ext" / "native_support_baselines.py",
        PROJECT_ROOT / "experiments_ext" / "raw_holdout_data.py",
        PROJECT_ROOT / "experiments_ext" / "raw_point_supervision.py",
        PROJECT_ROOT / "experiments_ext" / "support_aware_model.py",
    ]
    manifest: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "source_git_commit": source_commit(),
        "code_sha256_at_launch": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
            for path in code_files
        },
        "input": {
            "path": str(args.csv_path.resolve()),
            "size_bytes": args.csv_path.stat().st_size,
            "sha256": sha256(args.csv_path),
        },
        "tile": args.tile,
        "seed": args.seed,
        "history_start_col": args.history_start_col,
        "history_length": args.history_length,
        "target_col": args.target_col,
        "grid_size": args.grid_size,
        "block_side": args.block_side,
        "models": list(args.models),
        "spar_sampler": args.spar_sampler,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "dlinear_batch_size": args.dlinear_batch_size,
        "output_root": str(args.output_root.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "protocol": "cageo-native-pointwise-v23-locked",
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    spec = RawHoldoutSpec(
        csv_path=args.csv_path,
        tile=args.tile,
        grid_size=args.grid_size,
        history_start_col=args.history_start_col,
        history_length=args.history_length,
        target_col=args.target_col,
        split_seed=args.seed,
        block_side=args.block_side,
    )
    task = build_raw_holdout_task(spec, cache_dir=args.cache_dir)
    points, history, target = load_forecast_columns(spec)
    if not np.array_equal(points, task.raw_points):
        raise AssertionError("Native v2.3 point order differs from the cached task.")
    if not np.array_equal(target, task.raw_target):
        raise AssertionError("Native v2.3 target differs from the cached task.")
    _, _, date_metadata = load_forecast_dates(spec)
    (args.output_root / "task_metadata.json").write_text(
        json.dumps({**task.metadata, **date_metadata}, indent=2),
        encoding="utf-8",
    )
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

    results: list[dict[str, object]] = []
    for model_name in args.models:
        output_name = (
            f"spar_{args.spar_sampler}" if model_name == "spar" else model_name
        )
        output_dir = args.output_root / output_name
        metrics_path = output_dir / "metrics.json"
        if args.resume and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        elif model_name == "persistence":
            metrics = run_native_persistence(task, history, output_dir)
        elif model_name == "dlinear":
            metrics = run_native_dlinear(
                task,
                history,
                output_dir,
                seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.dlinear_batch_size,
                moving_average=25,
            )
        elif model_name == "lasso":
            metrics = run_raw_supervised_lasso(task, history, config, output_dir)
        elif model_name == "tcn":
            metrics = run_native_tcn(
                task,
                history,
                output_dir,
                seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
        elif model_name == "spar":
            metrics = run_direct_spar(
                task,
                history,
                config,
                output_dir,
                seed=args.seed,
                sampler=args.spar_sampler,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
        else:
            raise AssertionError(model_name)
        metrics.update(
            {
                **date_metadata,
                "seed": int(args.seed),
                "history_start_col": int(args.history_start_col),
                "target_col": int(args.target_col),
            }
        )
        metrics_path.write_text(
            json.dumps(metrics, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        results.append(metrics)
        print(
            json.dumps(
                {
                    "completed": output_name,
                    "seed": args.seed,
                    "native_cell_rmse": metrics.get(
                        "native_cell_rmse",
                        metrics.get("direct_raw_rmse"),
                    ),
                }
            ),
            flush=True,
        )

    (args.output_root / "summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    manifest.update(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": float(time.perf_counter() - started),
            "output_sha256": {
                str(path.relative_to(args.output_root)).replace("\\", "/"): sha256(path)
                for path in sorted(args.output_root.rglob("*"))
                if path.is_file() and path.name != "run_manifest.json"
            },
        }
    )
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
