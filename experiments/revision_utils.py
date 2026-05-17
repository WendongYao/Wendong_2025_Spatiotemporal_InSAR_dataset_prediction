"""
Shared data, split, metric, and plotting utilities for revision-aligned runs.

Revision skeleton alignment:
- Section 3.1 / data preprocessing, gridding, and masks
- Section 3.2 / task definition and split protocol
- Section 3.6 / evaluation metrics and spatial diagnostics
- Section 3.8 / interpolation-bias sensitivity
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from revision_config import RevisionConfig


@dataclass
class DenseForecastTask:
    input_maps: np.ndarray
    target_map: np.ndarray
    input_valid_mask: np.ndarray
    target_valid_mask: np.ndarray
    history_coverage: np.ndarray
    eligible_mask: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    interpolation_method: str
    csv_path: Path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config_snapshot(config: RevisionConfig, output_dir: Path) -> None:
    ensure_dir(output_dir)
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as fh:
        json.dump(config.as_dict(), fh, indent=2)


def load_revision_dataframe(config: RevisionConfig) -> pd.DataFrame:
    return pd.read_csv(config.resolve_csv_path())


def resolve_grid_coordinates(
    easting: np.ndarray,
    northing: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = np.mgrid[
        easting.min() : easting.max() : complex(0, grid_size),
        northing.min() : northing.max() : complex(0, grid_size),
    ]
    return grid_x, grid_y


def _task_cache_key(
    config: RevisionConfig,
    csv_path: Path,
    interpolation_method: str,
) -> str:
    payload = {
        "csv_path": str(csv_path.resolve()),
        "grid_size": config.grid_size,
        "history_start_col": config.history_start_col,
        "history_length": config.history_length,
        "target_col": config.target_col,
        "interpolation_method": interpolation_method,
        "min_history_coverage": config.min_history_coverage,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _task_cache_path(
    config: RevisionConfig,
    csv_path: Path,
    interpolation_method: str,
) -> Path:
    ensure_dir(config.task_cache_root)
    key = _task_cache_key(config, csv_path, interpolation_method)
    return config.task_cache_root / f"dense_task_{key}.npz"


def _load_cached_dense_forecast_task(
    cache_path: Path,
    csv_path: Path,
    config: RevisionConfig,
) -> DenseForecastTask:
    cached = np.load(cache_path, allow_pickle=False)
    eligible_mask = cached["eligible_mask"].astype(bool)
    train_mask, val_mask, test_mask = _build_split_masks(
        eligible_mask=eligible_mask,
        seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        split_strategy=config.split_strategy,
        tile_size=config.tile_size,
    )
    return DenseForecastTask(
        input_maps=cached["input_maps"].astype(np.float32),
        target_map=cached["target_map"].astype(np.float32),
        input_valid_mask=cached["input_valid_mask"].astype(bool),
        target_valid_mask=cached["target_valid_mask"].astype(bool),
        history_coverage=cached["history_coverage"].astype(np.float32),
        eligible_mask=eligible_mask,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        interpolation_method=str(cached["interpolation_method"][0]),
        csv_path=csv_path,
    )


def _save_cached_dense_forecast_task(task: DenseForecastTask, cache_path: Path) -> None:
    np.savez_compressed(
        cache_path,
        input_maps=task.input_maps.astype(np.float32),
        target_map=task.target_map.astype(np.float32),
        input_valid_mask=task.input_valid_mask.astype(np.uint8),
        target_valid_mask=task.target_valid_mask.astype(np.uint8),
        history_coverage=task.history_coverage.astype(np.float32),
        eligible_mask=task.eligible_mask.astype(np.uint8),
        train_mask=task.train_mask.astype(np.uint8),
        val_mask=task.val_mask.astype(np.uint8),
        test_mask=task.test_mask.astype(np.uint8),
        interpolation_method=np.array([task.interpolation_method]),
    )


def _grid_query_points(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def _idw_predict(
    points: np.ndarray,
    values: np.ndarray,
    query_points: np.ndarray,
    neighbors: int,
    power: float,
) -> np.ndarray:
    tree = cKDTree(points)
    k = max(1, min(int(neighbors), len(points)))
    distances, indices = tree.query(query_points, k=k)

    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    zero_mask = distances <= 1e-12
    safe_distances = np.where(zero_mask, 1.0, distances)
    weights = 1.0 / np.power(safe_distances, power)
    weights[zero_mask] = 0.0

    gathered = values[indices]
    weighted_sum = np.sum(weights * gathered, axis=1)
    weight_total = np.sum(weights, axis=1)
    pred = weighted_sum / np.clip(weight_total, 1e-12, None)

    if np.any(zero_mask):
        exact_rows = np.any(zero_mask, axis=1)
        exact_cols = np.argmax(zero_mask[exact_rows], axis=1)
        pred[exact_rows] = gathered[exact_rows, exact_cols]
    return pred.astype(np.float32)


def interpolate_query_points(
    points: np.ndarray,
    values: np.ndarray,
    query_points: np.ndarray,
    method: str,
    config: RevisionConfig,
    fill_missing: bool = False,
) -> np.ndarray:
    method_key = method.lower()
    if method_key in {"linear", "nearest", "cubic"}:
        pred = griddata(points, values, query_points, method=method_key)
        pred = np.asarray(pred, dtype=np.float64)
        if fill_missing and np.any(~np.isfinite(pred)):
            nearest = griddata(points, values, query_points, method="nearest")
            pred = np.where(np.isfinite(pred), pred, nearest)
        return pred.astype(np.float32)

    if method_key == "idw":
        return _idw_predict(
            points=points,
            values=np.asarray(values, dtype=np.float32),
            query_points=query_points,
            neighbors=config.idw_neighbors,
            power=config.idw_power,
        )

    if method_key == "rbf":
        from scipy.interpolate import RBFInterpolator

        neighbors = max(4, min(int(config.rbf_neighbors), len(points)))
        interpolator = RBFInterpolator(
            points.astype(np.float64),
            np.asarray(values, dtype=np.float64),
            kernel=config.rbf_kernel,
            smoothing=float(config.rbf_smoothing),
            neighbors=neighbors,
        )
        pred = interpolator(query_points.astype(np.float64))
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        if fill_missing:
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred.astype(np.float32)

    raise ValueError(f"Unsupported interpolation method: {method}")


def _interpolate_frame(
    points: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    config: RevisionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    query_points = _grid_query_points(grid_x, grid_y)
    raw_values = interpolate_query_points(
        points=points,
        values=np.asarray(values, dtype=np.float32),
        query_points=query_points,
        method=method,
        config=config,
        fill_missing=False,
    )
    raw_grid = raw_values.reshape(grid_x.shape)
    valid_mask = np.isfinite(raw_grid)
    filled_grid = np.nan_to_num(raw_grid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return filled_grid, valid_mask


def interpolate_column_to_grid(
    config: RevisionConfig,
    column_values: np.ndarray,
    interpolation_method: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    df = load_revision_dataframe(config)
    easting = df.iloc[:, 1].astype(float).values
    northing = df.iloc[:, 2].astype(float).values
    points = np.column_stack((easting, northing))
    grid_x, grid_y = resolve_grid_coordinates(easting, northing, config.grid_size)
    return _interpolate_frame(
        points=points,
        values=column_values,
        grid_x=grid_x,
        grid_y=grid_y,
        method=interpolation_method or config.interpolation_method,
        config=config,
    )


def _build_spatial_tile_split_masks(
    eligible_mask: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = eligible_mask.shape
    tiles = []
    for row in range(0, h, tile_size):
        for col in range(0, w, tile_size):
            tile_slice = eligible_mask[row : row + tile_size, col : col + tile_size]
            tile_count = int(tile_slice.sum())
            if tile_count > 0:
                tiles.append((row, col, tile_count))

    if len(tiles) < 3:
        raise ValueError("Spatial tile split requires at least three non-empty tiles.")

    rng = np.random.default_rng(seed)
    rng.shuffle(tiles)

    total_cells = int(eligible_mask.sum())
    target_counts = {
        "train": train_ratio * total_cells,
        "val": val_ratio * total_cells,
        "test": test_ratio * total_cells,
    }
    current_counts = {"train": 0, "val": 0, "test": 0}
    assignments: Dict[str, list[tuple[int, int]]] = {"train": [], "val": [], "test": []}

    for row, col, tile_count in tiles:
        split_name = max(
            current_counts.keys(),
            key=lambda key: target_counts[key] - current_counts[key],
        )
        assignments[split_name].append((row, col))
        current_counts[split_name] += tile_count

    train_mask = np.zeros_like(eligible_mask, dtype=bool)
    val_mask = np.zeros_like(eligible_mask, dtype=bool)
    test_mask = np.zeros_like(eligible_mask, dtype=bool)
    split_masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }

    for split_name, coords in assignments.items():
        split_mask = split_masks[split_name]
        for row, col in coords:
            tile_slice = eligible_mask[row : row + tile_size, col : col + tile_size]
            split_mask[row : row + tile_size, col : col + tile_size] = tile_slice

    if min(train_mask.sum(), val_mask.sum(), test_mask.sum()) <= 0:
        raise ValueError("Spatial tile split produced an empty partition; adjust tile size or ratios.")
    return train_mask, val_mask, test_mask


def _build_split_masks(
    eligible_mask: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_strategy: str,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")

    strategy = split_strategy.lower()
    if strategy == "spatial_tile":
        return _build_spatial_tile_split_masks(
            eligible_mask=eligible_mask,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            tile_size=tile_size,
        )
    if strategy != "random_pixel":
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    eligible_indices = np.flatnonzero(eligible_mask.ravel())
    if eligible_indices.size < 10:
        raise ValueError("Eligible mask is too small to create reproducible splits.")

    rng = np.random.default_rng(seed)
    shuffled = eligible_indices.copy()
    rng.shuffle(shuffled)

    n_total = shuffled.size
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("At least one split would be empty; adjust split ratios.")

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val :]

    train_mask = np.zeros_like(eligible_mask, dtype=bool)
    val_mask = np.zeros_like(eligible_mask, dtype=bool)
    test_mask = np.zeros_like(eligible_mask, dtype=bool)

    train_mask.ravel()[train_idx] = True
    val_mask.ravel()[val_idx] = True
    test_mask.ravel()[test_idx] = True
    return train_mask, val_mask, test_mask


def build_dense_forecast_task(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> DenseForecastTask:
    method = interpolation_method or config.interpolation_method
    csv_path = config.resolve_csv_path()
    cache_path = _task_cache_path(config, csv_path, method)

    if config.use_task_cache and cache_path.exists():
        return _load_cached_dense_forecast_task(cache_path, csv_path, config)

    df = pd.read_csv(csv_path)

    easting = df.iloc[:, 1].astype(float).values
    northing = df.iloc[:, 2].astype(float).values
    points = np.column_stack((easting, northing))

    grid_x, grid_y = resolve_grid_coordinates(easting, northing, config.grid_size)

    history_slice = slice(config.history_start_col, config.history_start_col + config.history_length)
    disp_history = df.iloc[:, history_slice].values
    disp_target = df.iloc[:, config.target_col].values

    input_frames = []
    input_valid_frames = []
    for time_idx in range(disp_history.shape[1]):
        frame, valid_mask = _interpolate_frame(
            points=points,
            values=disp_history[:, time_idx],
            grid_x=grid_x,
            grid_y=grid_y,
            method=method,
            config=config,
        )
        input_frames.append(frame)
        input_valid_frames.append(valid_mask)

    target_map, target_valid_mask = _interpolate_frame(
        points=points,
        values=disp_target,
        grid_x=grid_x,
        grid_y=grid_y,
        method=method,
        config=config,
    )

    input_maps = np.stack(input_frames, axis=0)
    input_valid_mask = np.stack(input_valid_frames, axis=0)
    history_coverage = input_valid_mask.mean(axis=0)

    eligible_mask = target_valid_mask & (history_coverage >= config.min_history_coverage)
    train_mask, val_mask, test_mask = _build_split_masks(
        eligible_mask=eligible_mask,
        seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        split_strategy=config.split_strategy,
        tile_size=config.tile_size,
    )

    task = DenseForecastTask(
        input_maps=input_maps,
        target_map=target_map,
        input_valid_mask=input_valid_mask,
        target_valid_mask=target_valid_mask,
        history_coverage=history_coverage.astype(np.float32),
        eligible_mask=eligible_mask,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        interpolation_method=method,
        csv_path=csv_path,
    )
    if config.use_task_cache:
        _save_cached_dense_forecast_task(task, cache_path)
    return task


def save_split_bundle(task: DenseForecastTask, output_dir: Path) -> None:
    ensure_dir(output_dir)
    np.savez_compressed(
        output_dir / "split_masks.npz",
        eligible_mask=task.eligible_mask.astype(np.uint8),
        train_mask=task.train_mask.astype(np.uint8),
        val_mask=task.val_mask.astype(np.uint8),
        test_mask=task.test_mask.astype(np.uint8),
        target_valid_mask=task.target_valid_mask.astype(np.uint8),
        history_coverage=task.history_coverage.astype(np.float32),
    )


def build_tabular_dataset(task: DenseForecastTask) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_table = task.input_maps.reshape(task.input_maps.shape[0], -1).T
    full_target = task.target_map.reshape(-1)
    eligible_indices = np.flatnonzero(task.eligible_mask.ravel())
    return (
        full_table[eligible_indices].astype(np.float32),
        full_target[eligible_indices].astype(np.float32),
        eligible_indices.astype(np.int64),
    )


def split_from_eligible_indices(
    task: DenseForecastTask,
    eligible_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    train_positions = np.flatnonzero(task.train_mask.ravel()[eligible_indices])
    val_positions = np.flatnonzero(task.val_mask.ravel()[eligible_indices])
    test_positions = np.flatnonzero(task.test_mask.ravel()[eligible_indices])
    return {
        "train": train_positions,
        "val": val_positions,
        "test": test_positions,
    }


def masked_regression_metrics(
    y_true_map: np.ndarray,
    y_pred_map: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true_map)[mask]
    y_pred = np.asarray(y_pred_map)[mask]
    if y_true.size == 0:
        raise ValueError("Metric mask is empty.")

    residuals = y_pred - y_true
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "n_valid_pixels": int(y_true.size),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def save_metrics(metrics: Dict[str, object], output_dir: Path, filename_stem: str = "metrics") -> None:
    ensure_dir(output_dir)
    with (output_dir / f"{filename_stem}.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    csv_path = output_dir / f"{filename_stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def save_prediction_map(pred_map: np.ndarray, output_dir: Path, filename: str = "prediction_map.npy") -> None:
    ensure_dir(output_dir)
    np.save(output_dir / filename, pred_map.astype(np.float32))


def save_error_diagnostics(
    y_true_map: np.ndarray,
    y_pred_map: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    n_bins: int = 20,
) -> None:
    ensure_dir(output_dir)
    y_true = np.asarray(y_true_map)[mask]
    y_pred = np.asarray(y_pred_map)[mask]
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=1, alpha=0.3)
    diag_min = float(min(y_true.min(), y_pred.min()))
    diag_max = float(max(y_true.max(), y_pred.max()))
    plt.plot([diag_min, diag_max], [diag_min, diag_max], "r--", linewidth=1)
    plt.xlabel("True Displacement (mm)")
    plt.ylabel("Predicted Displacement (mm)")
    plt.title("Scatter: True vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_true_vs_pred.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, residuals, s=1, alpha=0.3)
    plt.axhline(0.0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("True Displacement (mm)")
    plt.ylabel("Residual (Pred - True) (mm)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_plot.png", dpi=150)
    plt.close()

    bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_indices = np.clip(np.digitize(y_true, bins) - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_abs_err = [
        abs_residuals[bin_indices == idx].mean() if np.any(bin_indices == idx) else np.nan
        for idx in range(n_bins)
    ]

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, mean_abs_err, marker="o", linestyle="-")
    plt.xlabel("True Displacement Bin Center (mm)")
    plt.ylabel("Mean Absolute Error (mm)")
    plt.title("Binned Error Analysis")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "binned_error.png", dpi=150)
    plt.close()

    residuals_by_bin = [residuals[bin_indices == idx] for idx in range(n_bins)]
    plt.figure(figsize=(8, 4))
    plt.boxplot(residuals_by_bin, showfliers=True, widths=0.6)
    plt.xticks(np.arange(1, n_bins + 1), [f"{center:.1f}" for center in bin_centers], rotation=45)
    plt.xlabel("True Displacement Bin Center (mm)")
    plt.ylabel("Residual (Pred - True) (mm)")
    plt.title("Binned Residuals Boxplot")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "binned_residuals_boxplot.png", dpi=150)
    plt.close()


def save_map_comparison(
    y_true_map: np.ndarray,
    y_pred_map: np.ndarray,
    valid_mask: np.ndarray,
    output_dir: Path,
    filename: str = "reference_vs_prediction.png",
) -> None:
    ensure_dir(output_dir)
    truth = np.where(valid_mask, y_true_map, np.nan)
    pred = np.where(valid_mask, y_pred_map, np.nan)

    vmin = float(np.nanmin(truth))
    vmax = float(np.nanmax(truth))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Reference Displacement Map")
    plt.imshow(truth, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Displacement (mm)")
    plt.subplot(1, 2, 2)
    plt.title("Predicted Displacement Map")
    plt.imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Displacement (mm)")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
