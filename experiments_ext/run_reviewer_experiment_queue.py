"""Sequential, resumable queue for the remaining must-run reviewer evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
E32_CSV = PROJECT_ROOT.parent / "EGMS_L3_E32N34_100km_U_2018_2022_1.csv"


def run_stage(name: str, command: list[str], sentinel: Path, log_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    if sentinel.exists():
        record = {"stage": name, "status": "resumed", "sentinel": str(sentinel.resolve()), "wall_seconds": 0.0}
        print(json.dumps(record), flush=True)
        return record
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    (log_dir / f"{name}.out.log").write_text(completed.stdout, encoding="utf-8")
    (log_dir / f"{name}.err.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Stage {name} failed; see {log_dir}")
    if not sentinel.exists():
        raise RuntimeError(f"Stage {name} exited successfully but did not create {sentinel}")
    record = {
        "stage": name,
        "status": "completed",
        "sentinel": str(sentinel.resolve()),
        "wall_seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--logs-root", type=Path, default=PROJECT_ROOT / "logs" / "reviewer_queue")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-lightgbm", action="store_true")
    args = parser.parse_args()
    args.logs_root.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    common_runner = str(PROJECT_ROOT / "experiments_ext" / "run_raw_holdout_pilot.py")
    multi_runner = str(PROJECT_ROOT / "experiments_ext" / "run_multiregion_suite.py")
    manifest = str(PROJECT_ROOT / "data_manifests" / "CAGEO_DATA_BOUNDARY.csv")
    cache = str(args.results_root / "_raw_task_cache")
    records: list[dict[str, object]] = []

    dense_baseline_root = args.results_root / "R043_dense_baselines_E32N34_seed42"
    records.append(
        run_stage(
            "R043_dense_baselines_E32N34_seed42",
            [
                PYTHON,
                common_runner,
                "--csv-path",
                str(E32_CSV),
                "--tile",
                "E32N34",
                "--output-root",
                str(dense_baseline_root),
                "--cache-dir",
                cache,
                "--seed",
                "42",
                "--epochs",
                "60",
                "--patience",
                "12",
                "--batch-size",
                "16",
                "--models",
                "conv_lstm_residual",
                "simvp_style_residual",
                "--resume",
            ],
            dense_baseline_root / "summary.json",
            args.logs_root,
        )
    )

    buffer_root = args.results_root / "R012_buffer1_E32N34_seed42"
    records.append(
        run_stage(
            "R012_buffer1_E32N34_seed42",
            [
                PYTHON,
                common_runner,
                "--csv-path",
                str(E32_CSV),
                "--tile",
                "E32N34",
                "--output-root",
                str(buffer_root),
                "--cache-dir",
                cache,
                "--seed",
                "42",
                "--buffer-blocks",
                "1",
                "--epochs",
                "60",
                "--patience",
                "12",
                "--batch-size",
                "16",
                "--models",
                "persistence",
                "lasso",
                "lasso_raw_supervised",
                "cnn_lstm_hybrid",
                "cnn_lstm_raw_supervised",
                "--resume",
            ],
            buffer_root / "summary.json",
            args.logs_root,
        )
    )

    depth_root = args.results_root / "R042_two_layer_E32N34_seeds42_44"
    records.append(
        run_stage(
            "R042_two_layer_E32N34_seeds42_44",
            [
                PYTHON,
                multi_runner,
                "--manifest",
                manifest,
                "--tiles",
                "E32N34",
                "--seeds",
                "42",
                "43",
                "44",
                "--models",
                "cnn_lstm_hybrid",
                "cnn_lstm_raw_supervised",
                "--output-root",
                str(depth_root),
                "--cache-dir",
                cache,
                "--epochs",
                "60",
                "--patience",
                "12",
                "--batch-size",
                "16",
                "--convlstm-num-layers",
                "2",
            ],
            depth_root / "combined_results.json",
            args.logs_root,
        )
    )

    formulation_root = args.results_root / "R040_formulation_E32N34_seeds43_44"
    records.append(
        run_stage(
            "R040_formulation_E32N34_seeds43_44",
            [
                PYTHON,
                multi_runner,
                "--manifest",
                manifest,
                "--tiles",
                "E32N34",
                "--seeds",
                "43",
                "44",
                "--models",
                "conv_lstm_raw_supervised",
                "conv_lstm_raw_residual",
                "conv_lstm_absolute_supervised",
                "--output-root",
                str(formulation_root),
                "--cache-dir",
                cache,
                "--epochs",
                "60",
                "--patience",
                "12",
                "--batch-size",
                "16",
            ],
            formulation_root / "combined_results.json",
            args.logs_root,
        )
    )

    component_specs = [
        ("R041_no_warm_seed42", "--hybrid-no-warm-start"),
        ("R041_no_recent_gate_seed42", "--hybrid-disable-recent-gate"),
        ("R041_no_spatial_correction_seed42", "--hybrid-disable-spatial-correction"),
    ]
    for stage_name, flag in component_specs:
        output_root = args.results_root / stage_name
        records.append(
            run_stage(
                stage_name,
                [
                    PYTHON,
                    common_runner,
                    "--csv-path",
                    str(E32_CSV),
                    "--tile",
                    "E32N34",
                    "--output-root",
                    str(output_root),
                    "--cache-dir",
                    cache,
                    "--seed",
                    "42",
                    "--epochs",
                    "60",
                    "--patience",
                    "12",
                    "--batch-size",
                    "16",
                    "--models",
                    "cnn_lstm_hybrid",
                    "cnn_lstm_raw_supervised",
                    flag,
                    "--resume",
                ],
                output_root / "summary.json",
                args.logs_root,
            )
        )

    if not args.skip_synthetic:
        synthetic_main = args.results_root / "R021_synthetic_scenarios_grid128"
        records.append(
            run_stage(
                "R021_synthetic_scenarios_grid128",
                [
                    PYTHON,
                    str(PROJECT_ROOT / "experiments_ext" / "run_synthetic_truth_suite.py"),
                    "--output-root",
                    str(synthetic_main),
                    "--scenarios",
                    "seasonal_trend",
                    "localized_acceleration",
                    "moving_front",
                    "composite",
                    "--support-points",
                    "512",
                    "--noise-levels",
                    "0.35",
                    "--input-interpolations",
                    "idw",
                    "--seeds",
                    "42",
                    "43",
                    "44",
                    "--grid-size",
                    "128",
                    "--epochs",
                    "40",
                    "--patience",
                    "8",
                ],
                synthetic_main / "combined_results.json",
                args.logs_root,
            )
        )
        synthetic_stress = args.results_root / "R021_synthetic_stress_grid64"
        records.append(
            run_stage(
                "R021_synthetic_stress_grid64",
                [
                    PYTHON,
                    str(PROJECT_ROOT / "experiments_ext" / "run_synthetic_truth_suite.py"),
                    "--output-root",
                    str(synthetic_stress),
                    "--scenarios",
                    "composite",
                    "--support-points",
                    "128",
                    "256",
                    "512",
                    "--noise-levels",
                    "0.10",
                    "0.35",
                    "0.70",
                    "--input-interpolations",
                    "idw",
                    "--seeds",
                    "42",
                    "43",
                    "44",
                    "--grid-size",
                    "64",
                    "--epochs",
                    "30",
                    "--patience",
                    "6",
                ],
                synthetic_stress / "combined_results.json",
                args.logs_root,
            )
        )
        operator_root = args.results_root / "R022_synthetic_operator_shift_grid64"
        records.append(
            run_stage(
                "R022_synthetic_operator_shift_grid64",
                [
                    PYTHON,
                    str(PROJECT_ROOT / "experiments_ext" / "run_synthetic_truth_suite.py"),
                    "--output-root",
                    str(operator_root),
                    "--scenarios",
                    "composite",
                    "--support-points",
                    "256",
                    "--noise-levels",
                    "0.35",
                    "--input-interpolations",
                    "idw",
                    "linear",
                    "nearest",
                    "--seeds",
                    "42",
                    "43",
                    "44",
                    "--grid-size",
                    "64",
                    "--epochs",
                    "30",
                    "--patience",
                    "6",
                ],
                operator_root / "combined_results.json",
                args.logs_root,
            )
        )

    if not args.skip_lightgbm:
        lightgbm_root = args.results_root / "R051_lightgbm_variance"
        records.append(
            run_stage(
                "R051_lightgbm_variance",
                [
                    PYTHON,
                    str(PROJECT_ROOT / "experiments_ext" / "run_lightgbm_variance_diagnostic.py"),
                    "--output-root",
                    str(lightgbm_root),
                    "--seeds",
                    "42",
                    "43",
                    "44",
                    "45",
                    "46",
                    "--fit-seeds",
                    "42",
                    "43",
                    "44",
                    "45",
                    "46",
                    "--device-type",
                    "cpu",
                ],
                lightgbm_root / "lightgbm_variance.json",
                args.logs_root,
            )
        )

    manifest_payload = {
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": [PYTHON, *sys.argv],
        "records": records,
    }
    (args.logs_root / "queue_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
