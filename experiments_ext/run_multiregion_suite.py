"""Sequential, resumable multi-region runner for the CAGEO-only workspace."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def load_allowed_paths(manifest: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"].strip().lower() == "allowed" and row["absolute_path"].strip():
                paths[row["tile"].strip()] = Path(row["absolute_path"].strip())
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiles", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--convlstm-num-layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--hybrid-no-warm-start", action="store_true")
    parser.add_argument("--hybrid-disable-recent-gate", action="store_true")
    parser.add_argument("--hybrid-disable-spatial-correction", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    allowed = load_allowed_paths(args.manifest)
    missing = set(args.tiles) - set(allowed)
    if missing:
        raise ValueError(f"Tiles absent or excluded by manifest: {sorted(missing)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for tile in args.tiles:
        for seed in args.seeds:
            run_root = args.output_root / tile / f"seed_{seed}"
            summary = run_root / "summary.json"
            if summary.exists() and not args.force:
                payload = json.loads(summary.read_text(encoding="utf-8"))
                records.extend(payload)
                continue
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_raw_holdout_pilot.py")),
                "--csv-path",
                str(allowed[tile]),
                "--tile",
                tile,
                "--output-root",
                str(run_root),
                "--cache-dir",
                str(args.cache_dir),
                "--seed",
                str(seed),
                "--grid-size",
                "256",
                "--block-side",
                "8",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--convlstm-num-layers",
                str(args.convlstm_num_layers),
                "--models",
                *args.models,
                "--resume",
            ]
            if args.hybrid_no_warm_start:
                command.append("--hybrid-no-warm-start")
            if args.hybrid_disable_recent_gate:
                command.append("--hybrid-disable-recent-gate")
            if args.hybrid_disable_spatial_correction:
                command.append("--hybrid-disable-spatial-correction")
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
            (log_dir / f"{tile}_seed_{seed}.out.log").write_text(completed.stdout, encoding="utf-8")
            (log_dir / f"{tile}_seed_{seed}.err.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"Run failed for {tile} seed {seed}; see {log_dir}")
            payload = json.loads(summary.read_text(encoding="utf-8"))
            records.extend(payload)
            print(json.dumps({"completed_tile": tile, "seed": seed, "models": args.models}), flush=True)

    with (args.output_root / "combined_results.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()
