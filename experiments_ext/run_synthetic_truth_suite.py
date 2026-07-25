"""Resumable factorial suite for the analytic known-truth benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


SOURCE_COMMIT = "ffc1d4e8eb09c86ac81faa09ff662868b7494162"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["seasonal_trend", "localized_acceleration", "moving_front", "composite"],
        required=True,
    )
    parser.add_argument("--support-points", nargs="+", type=int, required=True)
    parser.add_argument("--noise-levels", nargs="+", type=float, required=True)
    parser.add_argument(
        "--input-interpolations",
        nargs="+",
        choices=["idw", "linear", "nearest"],
        default=["idw"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["persistence", "lasso", "cnn_lstm_hybrid", "conv_lstm_residual", "simvp_style_residual"],
        default=["persistence", "lasso", "cnn_lstm_hybrid", "conv_lstm_residual", "simvp_style_residual"],
    )
    args = parser.parse_args()

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    completed_configs = 0

    for scenario in args.scenarios:
        for support_points in args.support_points:
            for noise in args.noise_levels:
                for input_interpolation in args.input_interpolations:
                    for seed in args.seeds:
                        label = f"{scenario}_{input_interpolation}_n{support_points}_noise{noise:g}_seed{seed}"
                        run_root = args.output_root / label
                        summary_path = run_root / "summary.json"
                        if not summary_path.exists():
                            command = [
                                sys.executable,
                                str(Path(__file__).with_name("run_synthetic_truth_benchmark.py")),
                                "--output-root",
                                str(run_root),
                                "--scenario",
                                scenario,
                                "--grid-size",
                                str(args.grid_size),
                                "--support-points",
                                str(support_points),
                                "--noise",
                                str(noise),
                                "--input-interpolation",
                                input_interpolation,
                                "--seed",
                                str(seed),
                                "--split-seed",
                                str(seed),
                                "--epochs",
                                str(args.epochs),
                                "--patience",
                                str(args.patience),
                                "--models",
                                *args.models,
                            ]
                            completed = subprocess.run(
                                command,
                                cwd=Path(__file__).resolve().parents[1],
                                text=True,
                                capture_output=True,
                            )
                            (log_dir / f"{label}.out.log").write_text(completed.stdout, encoding="utf-8")
                            (log_dir / f"{label}.err.log").write_text(completed.stderr, encoding="utf-8")
                            if completed.returncode != 0:
                                raise RuntimeError(f"Synthetic run failed for {label}; see {log_dir}")
                        payload = json.loads(summary_path.read_text(encoding="utf-8"))
                        records.extend(payload)
                        completed_configs += 1
                        print(json.dumps({"completed_config": label, "models": args.models}), flush=True)

    combined_path = args.output_root / "combined_results.json"
    combined_path.write_text(json.dumps(records, indent=2, allow_nan=True), encoding="utf-8")
    manifest = {
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "source_git_commit": SOURCE_COMMIT,
        "grid_size": args.grid_size,
        "scenarios": args.scenarios,
        "support_points": args.support_points,
        "noise_levels": args.noise_levels,
        "input_interpolations": args.input_interpolations,
        "seeds": args.seeds,
        "models": args.models,
        "completed_configs": completed_configs,
        "wall_seconds": float(time.perf_counter() - started),
        "combined_results": str(combined_path.resolve()),
    }
    (args.output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
