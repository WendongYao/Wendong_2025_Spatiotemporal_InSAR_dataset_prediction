"""Wait for the reviewer queue, then run the informative component follow-ups.

The queue is deliberately CAGEO-only.  It supplements the seed-42 deletion
pilots with seeds 43--44 under raw-observation supervision and rebuilds the
derived reviewer-evidence tables after the runs finish.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def wait_for_windows_pid(pid: int) -> None:
    """Wait on the exact process handle, avoiding PID-reuse races."""
    synchronize = 0x00100000
    infinite = 0xFFFFFFFF
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return
    try:
        kernel32.WaitForSingleObject(handle, infinite)
    finally:
        kernel32.CloseHandle(handle)


def run_stage(
    name: str,
    command: list[str],
    sentinel: Path,
    log_root: Path,
) -> dict[str, object]:
    if sentinel.exists():
        record = {
            "stage": name,
            "status": "skipped_existing",
            "sentinel": str(sentinel.resolve()),
            "command": command,
        }
        print(json.dumps(record), flush=True)
        return record

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    (log_root / f"{name}.out.log").write_text(completed.stdout, encoding="utf-8")
    (log_root / f"{name}.err.log").write_text(completed.stderr, encoding="utf-8")
    record = {
        "stage": name,
        "status": "completed" if completed.returncode == 0 and sentinel.exists() else "failed",
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "sentinel": str(sentinel.resolve()),
        "command": command,
    }
    print(json.dumps(record), flush=True)
    if record["status"] != "completed":
        raise RuntimeError(f"{name} failed; inspect {log_root}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--logs-root", type=Path, default=PROJECT_ROOT / "logs" / "component_followup_queue")
    args = parser.parse_args()
    args.logs_root.mkdir(parents=True, exist_ok=True)

    started_utc = datetime.now(timezone.utc).isoformat()
    if args.wait_for_pid is not None:
        print(json.dumps({"waiting_for_pid": args.wait_for_pid}), flush=True)
        wait_for_windows_pid(args.wait_for_pid)
        print(json.dumps({"wait_completed_for_pid": args.wait_for_pid}), flush=True)

    common = [
        PYTHON,
        str(PROJECT_ROOT / "experiments_ext" / "run_multiregion_suite.py"),
        "--manifest",
        str(PROJECT_ROOT / "data_manifests" / "CAGEO_DATA_BOUNDARY.csv"),
        "--tiles",
        "E32N34",
        "--seeds",
        "43",
        "44",
        "--models",
        "cnn_lstm_raw_supervised",
        "--cache-dir",
        str(args.results_root / "_raw_task_cache"),
        "--epochs",
        "60",
        "--patience",
        "12",
        "--batch-size",
        "16",
    ]
    stages = [
        (
            "R041_no_recent_gate_seeds43_44",
            args.results_root / "R041_no_recent_gate_seeds43_44",
            "--hybrid-disable-recent-gate",
        ),
        (
            "R041_no_spatial_correction_seeds43_44",
            args.results_root / "R041_no_spatial_correction_seeds43_44",
            "--hybrid-disable-spatial-correction",
        ),
    ]
    records: list[dict[str, object]] = []
    for name, output_root, flag in stages:
        records.append(
            run_stage(
                name,
                [*common, "--output-root", str(output_root), flag],
                output_root / "combined_results.json",
                args.logs_root,
            )
        )

    evidence_root = args.results_root / "R052_reviewer_evidence"
    records.append(
        run_stage(
            "R052_finalize_reviewer_evidence",
            [
                PYTHON,
                str(PROJECT_ROOT / "experiments_ext" / "finalize_reviewer_evidence.py"),
                "--results-root",
                str(args.results_root),
                "--output-root",
                str(evidence_root),
            ],
            evidence_root / "evidence_summary.json",
            args.logs_root,
        )
    )
    manifest = {
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": [PYTHON, *sys.argv],
        "records": records,
    }
    (args.logs_root / "queue_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
