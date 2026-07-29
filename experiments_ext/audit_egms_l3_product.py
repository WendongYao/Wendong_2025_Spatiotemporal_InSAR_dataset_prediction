"""Audit the empirical EGMS Level-3 Ortho product support and task dates."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_gcd(values: np.ndarray) -> int:
    unique = np.unique(values.astype(np.int64))
    differences = np.diff(unique)
    positive = differences[differences > 0]
    if not len(positive):
        return 0
    result = int(positive[0])
    for value in positive[1:]:
        result = gcd(result, int(value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--tile", default="E32N34")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--history-start-col", type=int, default=11)
    parser.add_argument("--history-length", type=int, default=300)
    parser.add_argument("--target-col", type=int, default=312)
    args = parser.parse_args()

    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    frame = pd.read_csv(
        args.csv_path,
        usecols=["easting", "northing", "rmse"],
        dtype={"easting": np.int64, "northing": np.int64, "rmse": np.float32},
    )
    east = frame["easting"].to_numpy(dtype=np.int64)
    north = frame["northing"].to_numpy(dtype=np.int64)
    quality = frame["rmse"].to_numpy(dtype=np.float64)
    east_unique = np.unique(east)
    north_unique = np.unique(north)
    coordinate_pairs = np.column_stack((east, north))
    unique_pair_count = len(np.unique(coordinate_pairs, axis=0))
    east_extent_count = int((east_unique.max() - east_unique.min()) // 100 + 1)
    north_extent_count = int((north_unique.max() - north_unique.min()) // 100 + 1)
    extent_cells = int(east_extent_count * north_extent_count)
    history_labels = header[
        args.history_start_col : args.history_start_col + args.history_length
    ]
    skipped = header[
        args.history_start_col + args.history_length : args.target_col
    ]
    date_columns = [
        label for label in header if len(label) == 8 and label.isdigit()
    ]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tile": args.tile,
        "source": {
            "path": str(args.csv_path.resolve()),
            "size_bytes": args.csv_path.stat().st_size,
            "sha256": sha256(args.csv_path),
        },
        "product_interpretation": (
            "EGMS Level-3 Ortho valid product cells on a partially populated "
            "native 100-m grid; rows are not raw persistent scatterers."
        ),
        "rows": int(len(frame)),
        "unique_coordinate_pairs": int(unique_pair_count),
        "duplicate_coordinate_pairs": int(len(frame) - unique_pair_count),
        "native_grid": {
            "easting_unique_count": int(len(east_unique)),
            "northing_unique_count": int(len(north_unique)),
            "represented_axis_counts": [
                int(len(east_unique)),
                int(len(north_unique)),
            ],
            "full_100m_extent_shape": [east_extent_count, north_extent_count],
            "possible_extent_cells": extent_cells,
            "valid_cell_occupancy_percent": float(100.0 * len(frame) / extent_cells),
            "easting_min": int(east_unique.min()),
            "easting_max": int(east_unique.max()),
            "northing_min": int(north_unique.min()),
            "northing_max": int(north_unique.max()),
            "easting_spacing_gcd_m": positive_gcd(east_unique),
            "northing_spacing_gcd_m": positive_gcd(north_unique),
            "easting_mod_100_values": sorted(
                np.unique(np.mod(east_unique, 100)).astype(int).tolist()
            ),
            "northing_mod_100_values": sorted(
                np.unique(np.mod(north_unique, 100)).astype(int).tolist()
            ),
        },
        "temporal_task": {
            "date_column_count": len(date_columns),
            "date_start": date_columns[0],
            "date_end": date_columns[-1],
            "history_count": len(history_labels),
            "history_start": history_labels[0],
            "history_end": history_labels[-1],
            "skipped_dates": skipped,
            "target_date": header[args.target_col],
            "forecast_horizon_days": int(
                (
                    np.datetime64(
                        datetime.strptime(
                            header[args.target_col], "%Y%m%d"
                        ).date(),
                        "D",
                    )
                    - np.datetime64(
                        datetime.strptime(history_labels[-1], "%Y%m%d").date(),
                        "D",
                    )
                ).astype(np.int64)
            ),
        },
        "csv_rmse_field": {
            "semantic_note": (
                "Product quality attribute used only for stratification; it is "
                "not an independent ground-truth displacement error and is not "
                "equated with the official product-level displacement STD."
            ),
            "min": float(np.min(quality)),
            "q25": float(np.quantile(quality, 0.25)),
            "median": float(np.median(quality)),
            "mean": float(np.mean(quality)),
            "q75": float(np.quantile(quality, 0.75)),
            "max": float(np.max(quality)),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "egms_l3_product_audit.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
