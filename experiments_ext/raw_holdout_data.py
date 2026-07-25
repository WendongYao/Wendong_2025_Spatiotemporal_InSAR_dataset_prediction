"""Measurement-grounded CAGEO task construction.

The test target observations are separated at raw-point level before any target
gridding.  Input histories may still use every point because those histories
are available at forecast issue time.  Train and validation dense targets are
constructed from disjoint raw target subsets and are applied only inside their
own spatial blocks.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_utils import DenseForecastTask  # noqa: E402


SplitName = Literal["train", "val", "test", "buffer"]
SPLIT_TO_CODE: dict[SplitName, int] = {"buffer": -1, "train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class RawHoldoutSpec:
    csv_path: Path
    tile: str
    grid_size: int = 256
    history_start_col: int = 11
    history_length: int = 300
    target_col: int = 312
    split_seed: int = 42
    block_side: int = 8
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    buffer_blocks: int = 0
    idw_neighbors: int = 8
    idw_power: float = 2.0


@dataclass
class RawHoldoutTask:
    dense_task: DenseForecastTask
    raw_points: np.ndarray
    raw_target: np.ndarray
    raw_split_codes: np.ndarray
    raw_block_ids: np.ndarray
    easting_axis: np.ndarray
    northing_axis: np.ndarray
    train_target_source_indices: np.ndarray
    val_target_source_indices: np.ndarray
    test_target_indices: np.ndarray
    metadata: dict[str, object]


def _path_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    with path.open("rb") as handle:
        head = handle.read(1024 * 1024)
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "header_sha256": hashlib.sha256(head).hexdigest(),
    }


def _spec_cache_key(spec: RawHoldoutSpec) -> str:
    payload = {
        "source": _path_fingerprint(spec.csv_path),
        "tile": spec.tile,
        "grid_size": spec.grid_size,
        "history_start_col": spec.history_start_col,
        "history_length": spec.history_length,
        "target_col": spec.target_col,
        "split_seed": spec.split_seed,
        "block_side": spec.block_side,
        "ratios": [spec.train_ratio, spec.val_ratio, spec.test_ratio],
        "buffer_blocks": spec.buffer_blocks,
        "idw_neighbors": spec.idw_neighbors,
        "idw_power": spec.idw_power,
        "protocol": "raw-target-pregrid-holdout-v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def load_forecast_columns(spec: RawHoldoutSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load coordinates, 300-frame history, and target as float32 only."""
    columns = [1, 2] + list(range(spec.history_start_col, spec.history_start_col + spec.history_length)) + [spec.target_col]
    frame = pd.read_csv(spec.csv_path, usecols=columns, dtype=np.float32)
    if frame.shape[1] != spec.history_length + 3:
        raise ValueError(f"Unexpected selected column count: {frame.shape[1]}")
    points = frame.iloc[:, :2].to_numpy(dtype=np.float64, copy=True)
    history = frame.iloc[:, 2 : 2 + spec.history_length].to_numpy(dtype=np.float32, copy=True)
    target = frame.iloc[:, -1].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(points).all():
        raise ValueError("Coordinates contain non-finite values.")
    if not np.isfinite(target).all():
        raise ValueError("Target contains non-finite values; explicit missing-target handling is required.")
    return points, history, target


def load_quality_rmse(spec: RawHoldoutSpec, quality_col: int = 4) -> np.ndarray:
    quality = pd.read_csv(spec.csv_path, usecols=[quality_col], dtype=np.float32).iloc[:, 0].to_numpy(dtype=np.float32)
    if len(quality) == 0 or not np.isfinite(quality).all():
        raise ValueError("Quality RMSE column is empty or contains non-finite values.")
    return quality


def _axis_block_indices(values: np.ndarray, minimum: float, maximum: float, block_side: int) -> np.ndarray:
    span = max(float(maximum - minimum), np.finfo(np.float64).eps)
    scaled = (values - minimum) / span
    return np.clip(np.floor(scaled * block_side).astype(np.int16), 0, block_side - 1)


def assign_spatial_blocks(points: np.ndarray, block_side: int) -> tuple[np.ndarray, np.ndarray]:
    east_block = _axis_block_indices(points[:, 0], float(points[:, 0].min()), float(points[:, 0].max()), block_side)
    north_block = _axis_block_indices(points[:, 1], float(points[:, 1].min()), float(points[:, 1].max()), block_side)
    block_ids = east_block.astype(np.int32) * block_side + north_block.astype(np.int32)
    block_coords = np.column_stack((east_block, north_block)).astype(np.int16)
    return block_ids, block_coords


def split_raw_blocks(spec: RawHoldoutSpec, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    block_ids, block_coords = assign_spatial_blocks(points, spec.block_side)
    unique_blocks, counts = np.unique(block_ids, return_counts=True)
    if unique_blocks.size < 3:
        raise ValueError("At least three occupied spatial blocks are required.")

    records = [(int(block), int(count)) for block, count in zip(unique_blocks, counts, strict=True)]
    rng = np.random.default_rng(spec.split_seed)
    rng.shuffle(records)
    total = int(len(points))
    targets = {
        "train": spec.train_ratio * total,
        "val": spec.val_ratio * total,
        "test": spec.test_ratio * total,
    }
    current = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for block, count in records:
        split_name = max(current, key=lambda key: targets[key] - current[key])
        assignments[split_name].append(block)
        current[split_name] += count

    if spec.buffer_blocks > 0:
        protected = set(assignments["val"]) | set(assignments["test"])
        buffered_train: list[int] = []
        kept_train: list[int] = []
        for block in assignments["train"]:
            row, col = divmod(block, spec.block_side)
            too_close = False
            for protected_block in protected:
                prow, pcol = divmod(protected_block, spec.block_side)
                if max(abs(row - prow), abs(col - pcol)) <= spec.buffer_blocks:
                    too_close = True
                    break
            (buffered_train if too_close else kept_train).append(block)
        assignments["train"] = kept_train
        assignments["buffer"] = buffered_train
    else:
        assignments["buffer"] = []

    codes = np.full(len(points), SPLIT_TO_CODE["buffer"], dtype=np.int8)
    for split_name in ("train", "val", "test"):
        split_blocks = np.asarray(assignments[split_name], dtype=np.int32)
        codes[np.isin(block_ids, split_blocks)] = SPLIT_TO_CODE[split_name]  # type: ignore[index]
    if min(np.sum(codes == code) for code in (0, 1, 2)) <= 0:
        raise ValueError("Raw spatial-block split produced an empty train/val/test partition.")
    return codes, block_ids, assignments


def _idw_stencil(points: np.ndarray, queries: np.ndarray, *, neighbors: int, power: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("IDW source point set is empty.")
    k = max(1, min(int(neighbors), len(points)))
    distances, indices = cKDTree(points).query(queries, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    exact = distances <= 1e-12
    weights = np.zeros_like(distances, dtype=np.float64)
    nonexact_rows = ~np.any(exact, axis=1)
    weights[nonexact_rows] = 1.0 / np.maximum(distances[nonexact_rows], 1e-12) ** float(power)
    if np.any(~nonexact_rows):
        first_exact = np.argmax(exact[~nonexact_rows], axis=1)
        weights[~nonexact_rows, :] = 0.0
        weights[np.flatnonzero(~nonexact_rows), first_exact] = 1.0
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    return indices, weights.astype(np.float32)


def idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    *,
    neighbors: int,
    power: float,
    time_chunk: int = 16,
) -> np.ndarray:
    """Apply one spatial stencil to one or many value columns."""
    matrix = np.asarray(values, dtype=np.float32)
    was_vector = matrix.ndim == 1
    if was_vector:
        matrix = matrix[:, None]
    if matrix.shape[0] != len(points):
        raise ValueError("Point and value counts differ.")
    indices, weights = _idw_stencil(points, queries, neighbors=neighbors, power=power)
    output = np.empty((len(queries), matrix.shape[1]), dtype=np.float32)
    for start in range(0, matrix.shape[1], time_chunk):
        end = min(start + time_chunk, matrix.shape[1])
        neighbor_values = matrix[indices, start:end]
        finite = np.isfinite(neighbor_values)
        weighted = np.where(finite, neighbor_values, 0.0) * weights[:, :, None]
        denominator = (finite * weights[:, :, None]).sum(axis=1)
        chunk = weighted.sum(axis=1) / np.clip(denominator, 1e-12, None)
        chunk[denominator <= 0] = np.nan
        output[:, start:end] = chunk.astype(np.float32)
    return output[:, 0] if was_vector else output


def _grid_split_masks(
    spec: RawHoldoutSpec,
    easting_axis: np.ndarray,
    northing_axis: np.ndarray,
    assignments: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    grid_east, grid_north = np.meshgrid(easting_axis, northing_axis, indexing="ij")
    points = np.column_stack((grid_east.ravel(), grid_north.ravel()))
    grid_block_ids, _ = assign_spatial_blocks(points, spec.block_side)
    masks: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test", "buffer"):
        masks[split_name] = np.isin(grid_block_ids, assignments[split_name]).reshape(spec.grid_size, spec.grid_size)
    return masks


def _save_raw_task_cache(raw_task: RawHoldoutTask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    task = raw_task.dense_task
    np.savez_compressed(
        path,
        input_maps=task.input_maps.astype(np.float32),
        target_map=task.target_map.astype(np.float32),
        input_valid_mask=task.input_valid_mask.astype(np.uint8),
        target_valid_mask=task.target_valid_mask.astype(np.uint8),
        history_coverage=task.history_coverage.astype(np.float32),
        eligible_mask=task.eligible_mask.astype(np.uint8),
        train_mask=task.train_mask.astype(np.uint8),
        val_mask=task.val_mask.astype(np.uint8),
        test_mask=task.test_mask.astype(np.uint8),
        raw_points=raw_task.raw_points.astype(np.float64),
        raw_target=raw_task.raw_target.astype(np.float32),
        raw_split_codes=raw_task.raw_split_codes.astype(np.int8),
        raw_block_ids=raw_task.raw_block_ids.astype(np.int32),
        easting_axis=raw_task.easting_axis.astype(np.float64),
        northing_axis=raw_task.northing_axis.astype(np.float64),
        train_target_source_indices=raw_task.train_target_source_indices.astype(np.int64),
        val_target_source_indices=raw_task.val_target_source_indices.astype(np.int64),
        test_target_indices=raw_task.test_target_indices.astype(np.int64),
        metadata_json=np.asarray([json.dumps(raw_task.metadata, sort_keys=True)]),
    )


def _load_raw_task_cache(path: Path, csv_path: Path) -> RawHoldoutTask:
    cached = np.load(path, allow_pickle=False)
    dense_task = DenseForecastTask(
        input_maps=cached["input_maps"].astype(np.float32),
        target_map=cached["target_map"].astype(np.float32),
        input_valid_mask=cached["input_valid_mask"].astype(bool),
        target_valid_mask=cached["target_valid_mask"].astype(bool),
        history_coverage=cached["history_coverage"].astype(np.float32),
        eligible_mask=cached["eligible_mask"].astype(bool),
        train_mask=cached["train_mask"].astype(bool),
        val_mask=cached["val_mask"].astype(bool),
        test_mask=cached["test_mask"].astype(bool),
        interpolation_method="idw_raw_holdout",
        csv_path=csv_path,
    )
    return RawHoldoutTask(
        dense_task=dense_task,
        raw_points=cached["raw_points"].astype(np.float64),
        raw_target=cached["raw_target"].astype(np.float32),
        raw_split_codes=cached["raw_split_codes"].astype(np.int8),
        raw_block_ids=cached["raw_block_ids"].astype(np.int32),
        easting_axis=cached["easting_axis"].astype(np.float64),
        northing_axis=cached["northing_axis"].astype(np.float64),
        train_target_source_indices=cached["train_target_source_indices"].astype(np.int64),
        val_target_source_indices=cached["val_target_source_indices"].astype(np.int64),
        test_target_indices=cached["test_target_indices"].astype(np.int64),
        metadata=json.loads(str(cached["metadata_json"][0])),
    )


def build_raw_holdout_task(spec: RawHoldoutSpec, *, cache_dir: Path | None = None) -> RawHoldoutTask:
    spec = RawHoldoutSpec(**{**spec.__dict__, "csv_path": spec.csv_path.resolve()})
    if abs(spec.train_ratio + spec.val_ratio + spec.test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to one.")
    cache_path = None if cache_dir is None else cache_dir / f"raw_holdout_{_spec_cache_key(spec)}.npz"
    if cache_path is not None and cache_path.exists():
        return _load_raw_task_cache(cache_path, spec.csv_path)

    started = time.perf_counter()
    points, history, target = load_forecast_columns(spec)
    load_seconds = time.perf_counter() - started
    split_codes, block_ids, assignments = split_raw_blocks(spec, points)
    train_indices = np.flatnonzero(split_codes == SPLIT_TO_CODE["train"])
    val_indices = np.flatnonzero(split_codes == SPLIT_TO_CODE["val"])
    test_indices = np.flatnonzero(split_codes == SPLIT_TO_CODE["test"])
    if np.intersect1d(test_indices, train_indices).size or np.intersect1d(test_indices, val_indices).size:
        raise AssertionError("Raw target split overlap detected.")

    easting_axis = np.linspace(points[:, 0].min(), points[:, 0].max(), spec.grid_size, dtype=np.float64)
    northing_axis = np.linspace(points[:, 1].min(), points[:, 1].max(), spec.grid_size, dtype=np.float64)
    grid_east, grid_north = np.meshgrid(easting_axis, northing_axis, indexing="ij")
    grid_queries = np.column_stack((grid_east.ravel(), grid_north.ravel()))

    interpolation_started = time.perf_counter()
    input_flat = idw_interpolate(
        points,
        history,
        grid_queries,
        neighbors=spec.idw_neighbors,
        power=spec.idw_power,
    )
    input_maps = input_flat.T.reshape(spec.history_length, spec.grid_size, spec.grid_size)
    train_target = idw_interpolate(
        points[train_indices], target[train_indices], grid_queries, neighbors=spec.idw_neighbors, power=spec.idw_power
    ).reshape(spec.grid_size, spec.grid_size)
    val_target = idw_interpolate(
        points[val_indices], target[val_indices], grid_queries, neighbors=spec.idw_neighbors, power=spec.idw_power
    ).reshape(spec.grid_size, spec.grid_size)
    interpolation_seconds = time.perf_counter() - interpolation_started

    grid_masks = _grid_split_masks(spec, easting_axis, northing_axis, assignments)
    target_map = input_maps[-1].copy()
    target_map[grid_masks["train"]] = train_target[grid_masks["train"]]
    target_map[grid_masks["val"]] = val_target[grid_masks["val"]]
    input_valid_mask = np.isfinite(input_maps)
    target_valid_mask = grid_masks["train"] | grid_masks["val"]
    eligible_mask = grid_masks["train"] | grid_masks["val"] | grid_masks["test"]
    dense_task = DenseForecastTask(
        input_maps=np.nan_to_num(input_maps, nan=0.0).astype(np.float32),
        target_map=np.nan_to_num(target_map, nan=0.0).astype(np.float32),
        input_valid_mask=input_valid_mask,
        target_valid_mask=target_valid_mask,
        history_coverage=input_valid_mask.mean(axis=0).astype(np.float32),
        eligible_mask=eligible_mask,
        train_mask=grid_masks["train"],
        val_mask=grid_masks["val"],
        test_mask=grid_masks["test"],
        interpolation_method="idw_raw_holdout",
        csv_path=spec.csv_path,
    )
    metadata: dict[str, object] = {
        "protocol": "raw-target-pregrid-holdout-v1",
        "tile": spec.tile,
        "source_fingerprint": _path_fingerprint(spec.csv_path),
        "split_seed": spec.split_seed,
        "block_side": spec.block_side,
        "buffer_blocks": spec.buffer_blocks,
        "idw_neighbors": spec.idw_neighbors,
        "idw_power": spec.idw_power,
        "raw_train_points": int(len(train_indices)),
        "raw_val_points": int(len(val_indices)),
        "raw_test_points": int(len(test_indices)),
        "raw_buffer_points": int(np.sum(split_codes == SPLIT_TO_CODE["buffer"])),
        "grid_train_cells": int(grid_masks["train"].sum()),
        "grid_val_cells": int(grid_masks["val"].sum()),
        "grid_test_cells": int(grid_masks["test"].sum()),
        "load_seconds": float(load_seconds),
        "interpolation_seconds": float(interpolation_seconds),
        "test_target_used_in_any_target_grid": False,
    }
    raw_task = RawHoldoutTask(
        dense_task=dense_task,
        raw_points=points,
        raw_target=target,
        raw_split_codes=split_codes,
        raw_block_ids=block_ids,
        easting_axis=easting_axis,
        northing_axis=northing_axis,
        train_target_source_indices=train_indices,
        val_target_source_indices=val_indices,
        test_target_indices=test_indices,
        metadata=metadata,
    )
    if cache_path is not None:
        _save_raw_task_cache(raw_task, cache_path)
    return raw_task


def sample_grid_at_raw_points(grid: np.ndarray, raw_task: RawHoldoutTask, indices: np.ndarray) -> np.ndarray:
    sampler = RegularGridInterpolator(
        (raw_task.easting_axis, raw_task.northing_axis),
        np.asarray(grid, dtype=np.float32),
        method="linear",
        bounds_error=True,
    )
    return np.asarray(sampler(raw_task.raw_points[indices]), dtype=np.float32)


def raw_point_metrics(prediction_grid: np.ndarray, raw_task: RawHoldoutTask, split: Literal["val", "test"] = "test") -> dict[str, float | int]:
    code = SPLIT_TO_CODE[split]
    indices = np.flatnonzero(raw_task.raw_split_codes == code)
    prediction = sample_grid_at_raw_points(prediction_grid, raw_task, indices)
    truth = raw_task.raw_target[indices]
    residual = prediction - truth
    mse = float(np.mean(residual**2))
    return {
        "n_points": int(len(indices)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "p95_absolute_error": float(np.quantile(np.abs(residual), 0.95)),
        "r2": float(1.0 - np.sum(residual**2) / np.clip(np.sum((truth - truth.mean()) ** 2), 1e-12, None)),
    }


def cell_aggregated_metrics(
    prediction_grid: np.ndarray,
    raw_task: RawHoldoutTask,
    split: Literal["val", "test"] = "test",
) -> dict[str, float | int]:
    """Evaluate held-out raw observations at the support scale of the output grid.

    Raw observations are grouped by their nearest output cell.  The returned
    decomposition uses one constant prediction per cell, so point-level MSE is
    exactly the sum of between-cell prediction MSE and within-cell target
    variance (up to floating-point precision).
    """
    indices = np.flatnonzero(raw_task.raw_split_codes == SPLIT_TO_CODE[split])
    points = raw_task.raw_points[indices]
    truth = raw_task.raw_target[indices].astype(np.float64)
    east_index = np.rint(
        (points[:, 0] - raw_task.easting_axis[0])
        / (raw_task.easting_axis[-1] - raw_task.easting_axis[0])
        * (len(raw_task.easting_axis) - 1)
    ).astype(np.int64)
    north_index = np.rint(
        (points[:, 1] - raw_task.northing_axis[0])
        / (raw_task.northing_axis[-1] - raw_task.northing_axis[0])
        * (len(raw_task.northing_axis) - 1)
    ).astype(np.int64)
    east_index = np.clip(east_index, 0, len(raw_task.easting_axis) - 1)
    north_index = np.clip(north_index, 0, len(raw_task.northing_axis) - 1)
    cell_ids = east_index * len(raw_task.northing_axis) + north_index
    unique_cells, inverse, counts = np.unique(cell_ids, return_inverse=True, return_counts=True)
    target_sum = np.bincount(inverse, weights=truth)
    target_mean = target_sum / counts
    cell_east = unique_cells // len(raw_task.northing_axis)
    cell_north = unique_cells % len(raw_task.northing_axis)
    prediction = np.asarray(prediction_grid, dtype=np.float64)[cell_east, cell_north]
    cell_residual = prediction - target_mean
    within_residual = truth - target_mean[inverse]
    point_residual_nearest = prediction[inverse] - truth
    between_mse_point_weighted = float(np.average(cell_residual**2, weights=counts))
    within_mse = float(np.mean(within_residual**2))
    point_mse_nearest = float(np.mean(point_residual_nearest**2))
    return {
        "cell_count": int(len(unique_cells)),
        "cell_mean_rmse_equal_cell": float(np.sqrt(np.mean(cell_residual**2))),
        "cell_mean_mae_equal_cell": float(np.mean(np.abs(cell_residual))),
        "cell_mean_rmse_point_weighted": float(np.sqrt(between_mse_point_weighted)),
        "within_cell_target_rmse": float(np.sqrt(within_mse)),
        "nearest_cell_point_rmse": float(np.sqrt(point_mse_nearest)),
        "mse_decomposition_error": float(abs(point_mse_nearest - (between_mse_point_weighted + within_mse))),
        "points_per_cell_mean": float(counts.mean()),
        "points_per_cell_p95": float(np.quantile(counts, 0.95)),
    }
