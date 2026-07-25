"""Add cell-support measurement metrics to completed raw-holdout predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from raw_holdout_data import RawHoldoutSpec, build_raw_holdout_task, cell_aggregated_metrics, raw_point_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--block-side", type=int, default=8)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.prediction) != len(args.label):
        raise ValueError("--prediction and --label counts differ")
    task = build_raw_holdout_task(
        RawHoldoutSpec(
            csv_path=args.csv_path,
            tile=args.tile,
            grid_size=args.grid_size,
            split_seed=args.seed,
            block_side=args.block_side,
        ),
        cache_dir=args.cache_dir,
    )
    rows = []
    for label, path in zip(args.label, args.prediction, strict=True):
        prediction = np.load(path)
        rows.append(
            {
                "model": label,
                **raw_point_metrics(prediction, task),
                **cell_aggregated_metrics(prediction, task),
                "prediction_path": str(path.resolve()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
