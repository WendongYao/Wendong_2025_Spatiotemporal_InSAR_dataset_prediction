"""Run reviewer-priority native-product-cell baselines with full provenance."""

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

from native_support_baselines import (
    load_forecast_dates,
    run_native_dlinear,
    run_native_linear_trend,
    run_native_persistence,
)
from raw_holdout_data import RawHoldoutSpec, build_raw_holdout_task, load_forecast_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--tile", default="E32N34")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["persistence", "linear_trend", "dlinear"],
        default=["persistence", "linear_trend", "dlinear"],
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--moving-average", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    code_files = [
        PROJECT_ROOT / "experiments_ext" / "native_support_baselines.py",
        PROJECT_ROOT / "experiments_ext" / "run_v22_native_baselines.py",
        PROJECT_ROOT / "experiments_ext" / "raw_holdout_data.py",
        PROJECT_ROOT / "experiments_ext" / "raw_point_supervision.py",
    ]
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except ImportError:
        torch_version = None
        cuda_version = None
        device_name = "unavailable"
    manifest: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "device": device_name,
        "source_git_commit": git_commit(),
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
        "models": args.models,
        "output_root": str(args.output_root.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "moving_average": args.moving_average,
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    spec = RawHoldoutSpec(
        csv_path=args.csv_path,
        tile=args.tile,
        grid_size=256,
        split_seed=args.seed,
        block_side=8,
    )
    task = build_raw_holdout_task(spec, cache_dir=args.cache_dir)
    points, history, target = load_forecast_columns(spec)
    if not np.array_equal(points, task.raw_points) or not np.allclose(
        target, task.raw_target
    ):
        raise AssertionError("Native baseline data order differs from cached split.")
    history_days, target_day, date_metadata = load_forecast_dates(spec)
    results: list[dict[str, object]] = []
    for model_name in args.models:
        output_dir = args.output_root / model_name
        metrics_path = output_dir / "metrics.json"
        if args.resume and metrics_path.exists():
            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        if model_name == "persistence":
            metrics = run_native_persistence(task, history, output_dir)
        elif model_name == "linear_trend":
            metrics = run_native_linear_trend(
                task,
                history,
                history_days,
                target_day,
                output_dir,
            )
        elif model_name == "dlinear":
            metrics = run_native_dlinear(
                task,
                history,
                output_dir,
                seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                moving_average=args.moving_average,
            )
        else:
            raise AssertionError(model_name)
        metrics.update(date_metadata)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        results.append(metrics)
        print(
            json.dumps(
                {
                    "completed": model_name,
                    "seed": args.seed,
                    "native_cell_rmse": metrics["native_cell_rmse"],
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
            "wall_seconds": time.perf_counter() - started,
            "result_files": [
                str((args.output_root / name / "metrics.json").resolve())
                for name in args.models
            ],
            "output_sha256": {
                str(path.relative_to(args.output_root)).replace("\\", "/"): sha256(
                    path
                )
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
