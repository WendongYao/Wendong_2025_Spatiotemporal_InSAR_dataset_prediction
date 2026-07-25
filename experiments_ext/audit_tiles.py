"""Streaming schema and data-quality audit for CAGEO EGMS tiles.

This module is deliberately independent of the workspace-root RSASE code.  It
reads only paths marked ``allowed`` in ``CAGEO_DATA_BOUNDARY.csv`` and never
writes beside the source CSV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


HISTORY_START_COL = 11
HISTORY_LENGTH = 300
TARGET_COL = 312
SKIPPED_COL = 311
EXCLUDED_NEW_TILES = {"E32N34", "E30N34", "E33N36", "E35N33", "E34N38"}


@dataclass(frozen=True)
class TileEntry:
    tile: str
    role: str
    status: str
    path: Path
    boundary_note: str


def _load_manifest(path: Path) -> list[TileEntry]:
    entries: list[TileEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_path = row["absolute_path"].strip()
            entries.append(
                TileEntry(
                    tile=row["tile"].strip(),
                    role=row["role"].strip(),
                    status=row["status"].strip().lower(),
                    path=Path(raw_path) if raw_path else Path(),
                    boundary_note=row["boundary_note"].strip(),
                )
            )
    return entries


def _assert_allowed(entry: TileEntry) -> None:
    if entry.status != "allowed":
        raise ValueError(f"Tile {entry.tile} is not allowed by the CAGEO boundary manifest.")
    if entry.tile in EXCLUDED_NEW_TILES and entry.role != "original_cageo_primary":
        raise ValueError(f"Tile {entry.tile} is reserved for RSASE future work.")
    if not entry.path.is_file():
        raise FileNotFoundError(entry.path)


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_finite": 0,
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p01": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p99": float("nan"),
        }
    return {
        f"{prefix}_finite": int(finite.size),
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std": float(finite.std()),
        f"{prefix}_p01": float(np.quantile(finite, 0.01)),
        f"{prefix}_p50": float(np.quantile(finite, 0.50)),
        f"{prefix}_p99": float(np.quantile(finite, 0.99)),
    }


def audit_tile(entry: TileEntry, *, chunksize: int, full_hash: bool) -> dict[str, object]:
    _assert_allowed(entry)
    stat = entry.path.stat()
    header = pd.read_csv(entry.path, nrows=0).columns.tolist()
    if len(header) <= TARGET_COL:
        raise ValueError(f"{entry.tile}: expected at least {TARGET_COL + 1} columns, got {len(header)}")

    history_columns = header[HISTORY_START_COL : HISTORY_START_COL + HISTORY_LENGTH]
    target_name = header[TARGET_COL]
    skipped_name = header[SKIPPED_COL]
    last_input_name = history_columns[-1]
    expected_names = ["pid", "easting", "northing", "rmse"]
    for name in expected_names:
        if name not in header:
            raise ValueError(f"{entry.tile}: missing required column {name!r}")

    usecols = list(dict.fromkeys(expected_names + [last_input_name, skipped_name, target_name]))
    coordinate_hashes: set[int] = set()
    row_count = 0
    values: dict[str, list[np.ndarray]] = {
        "easting": [],
        "northing": [],
        "rmse": [],
        "last_input": [],
        "skipped": [],
        "target": [],
        "target_delta_from_last": [],
        "target_delta_from_skipped": [],
    }
    missing = {key: 0 for key in usecols}

    for chunk in pd.read_csv(entry.path, usecols=usecols, chunksize=chunksize):
        row_count += len(chunk)
        for name in usecols:
            missing[name] += int(chunk[name].isna().sum())

        east = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy(dtype=np.float64)
        north = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy(dtype=np.float64)
        rmse = pd.to_numeric(chunk["rmse"], errors="coerce").to_numpy(dtype=np.float32)
        last_input = pd.to_numeric(chunk[last_input_name], errors="coerce").to_numpy(dtype=np.float32)
        skipped = pd.to_numeric(chunk[skipped_name], errors="coerce").to_numpy(dtype=np.float32)
        target = pd.to_numeric(chunk[target_name], errors="coerce").to_numpy(dtype=np.float32)

        coordinate_frame = pd.DataFrame({"easting": east, "northing": north})
        coordinate_hashes.update(
            int(value) for value in pd.util.hash_pandas_object(coordinate_frame, index=False).to_numpy(dtype=np.uint64)
        )
        values["easting"].append(east)
        values["northing"].append(north)
        values["rmse"].append(rmse)
        values["last_input"].append(last_input)
        values["skipped"].append(skipped)
        values["target"].append(target)
        values["target_delta_from_last"].append(target - last_input)
        values["target_delta_from_skipped"].append(target - skipped)

    combined = {key: np.concatenate(parts) for key, parts in values.items()}
    payload: dict[str, object] = {
        "tile": entry.tile,
        "role": entry.role,
        "path": str(entry.path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(entry.path) if full_hash else None,
        "row_count": int(row_count),
        "column_count": int(len(header)),
        "coordinate_unique_count": int(len(coordinate_hashes)),
        "coordinate_duplicate_count": int(row_count - len(coordinate_hashes)),
        "history_start": history_columns[0],
        "history_end": history_columns[-1],
        "history_length": int(len(history_columns)),
        "skipped_date": skipped_name,
        "target_date": target_name,
        "missing_counts": missing,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key, array in combined.items():
        payload.update(_safe_stats(array, key))
    return payload


def _flatten_for_csv(payload: dict[str, object]) -> dict[str, object]:
    flat = {key: value for key, value in payload.items() if not isinstance(value, dict)}
    flat["missing_counts_json"] = json.dumps(payload["missing_counts"], sort_keys=True)
    return flat


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit CAGEO EGMS tiles without loading full 300-frame tables.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tiles", nargs="*", default=None)
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    entries = _load_manifest(args.manifest)
    selected = [entry for entry in entries if entry.status == "allowed"]
    if args.tiles:
        wanted = set(args.tiles)
        selected = [entry for entry in selected if entry.tile in wanted]
        missing_tiles = wanted - {entry.tile for entry in selected}
        if missing_tiles:
            raise ValueError(f"Requested tiles are absent or excluded: {sorted(missing_tiles)}")
    if not selected:
        raise ValueError("No allowed tiles selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [audit_tile(entry, chunksize=args.chunksize, full_hash=args.full_hash) for entry in selected]
    with (args.output_dir / "tile_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, allow_nan=True)
    pd.DataFrame([_flatten_for_csv(report) for report in reports]).to_csv(
        args.output_dir / "tile_audit.csv", index=False
    )
    print(json.dumps({"tiles": [report["tile"] for report in reports], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
