"""Analytic deformation fields for testing interpolation self-consistency.

Targets are evaluated directly from analytic functions.  They are never made
by IDW or another gridding operator, so agreement cannot be explained by using
the same interpolator for inputs and supervision.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_utils import DenseForecastTask, _build_spatial_tile_split_masks  # noqa: E402

from raw_holdout_data import RawHoldoutTask, idw_interpolate


Scenario = Literal["seasonal_trend", "localized_acceleration", "moving_front", "composite"]
InputInterpolation = Literal["idw", "linear", "nearest"]


@dataclass(frozen=True)
class SyntheticTruthSpec:
    scenario: Scenario = "composite"
    grid_size: int = 128
    history_length: int = 300
    target_step: int = 301
    support_points: int = 1024
    observation_noise_std: float = 0.35
    seed: int = 42
    split_seed: int = 42
    tile_size: int = 16
    idw_neighbors: int = 8
    idw_power: float = 2.0
    input_interpolation: InputInterpolation = "idw"


def analytic_field(x: np.ndarray, y: np.ndarray, t: float, scenario: Scenario) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    base = 1.8 * np.sin(2.0 * np.pi * x) * np.cos(1.5 * np.pi * y)
    velocity = 0.018 * (x - 0.45) - 0.012 * (y - 0.55)
    trend = base + t * velocity
    seasonal_amp = 1.2 + 2.8 * (0.25 + x) * (0.35 + y)
    seasonal_phase = 1.6 * x - 1.1 * y
    seasonal = seasonal_amp * np.sin(2.0 * np.pi * t / 60.0 + seasonal_phase)

    g1 = np.exp(-((x - 0.33) ** 2 / 0.012 + (y - 0.62) ** 2 / 0.018))
    g2 = np.exp(-((x - 0.76) ** 2 / 0.025 + (y - 0.30) ** 2 / 0.015))
    acceleration_time = max(t - 220.0, 0.0)
    localized = -0.0018 * acceleration_time**2 * g1 + 0.0008 * acceleration_time**2 * g2

    front_center = 0.26 + 0.00155 * t
    moving_front = 4.5 * np.tanh((x - front_center) / 0.035) * (0.55 + 0.45 * np.cos(np.pi * y))

    if scenario == "seasonal_trend":
        field = trend + seasonal
    elif scenario == "localized_acceleration":
        field = trend + 0.35 * seasonal + localized
    elif scenario == "moving_front":
        field = trend + 0.25 * seasonal + moving_front
    elif scenario == "composite":
        field = trend + seasonal + localized + moving_front
    else:
        raise ValueError(scenario)
    return field.astype(np.float32)


def sample_irregular_support(spec: SyntheticTruthSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    n_uniform = int(round(spec.support_points * 0.60))
    n_cluster = spec.support_points - n_uniform
    uniform = rng.uniform(0.0, 1.0, size=(n_uniform * 2, 2))
    gap = ((uniform[:, 0] - 0.55) / 0.14) ** 2 + ((uniform[:, 1] - 0.56) / 0.18) ** 2
    uniform = uniform[gap > 1.0][:n_uniform]
    while len(uniform) < n_uniform:
        extra = rng.uniform(0.0, 1.0, size=(n_uniform, 2))
        extra_gap = ((extra[:, 0] - 0.55) / 0.14) ** 2 + ((extra[:, 1] - 0.56) / 0.18) ** 2
        uniform = np.vstack((uniform, extra[extra_gap > 1.0]))[:n_uniform]

    centers = np.asarray([[0.25, 0.70], [0.78, 0.28], [0.72, 0.78]], dtype=np.float64)
    chosen = centers[rng.integers(0, len(centers), size=n_cluster)]
    cluster = np.clip(chosen + rng.normal(0.0, 0.075, size=(n_cluster, 2)), 0.0, 1.0)
    support = np.vstack((uniform, cluster))
    rng.shuffle(support)
    return support.astype(np.float64)


def build_synthetic_truth_task(spec: SyntheticTruthSpec) -> RawHoldoutTask:
    rng = np.random.default_rng(spec.seed)
    axis = np.linspace(0.0, 1.0, spec.grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    queries = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    support = sample_irregular_support(spec)
    history_steps = np.arange(spec.history_length, dtype=np.float64)
    support_history = np.stack(
        [analytic_field(support[:, 0], support[:, 1], float(step), spec.scenario) for step in history_steps], axis=1
    )
    support_history += rng.normal(0.0, spec.observation_noise_std, size=support_history.shape).astype(np.float32)
    if spec.input_interpolation == "idw":
        input_flat = idw_interpolate(
            support,
            support_history,
            queries,
            neighbors=spec.idw_neighbors,
            power=spec.idw_power,
        )
    elif spec.input_interpolation == "nearest":
        nearest_indices = cKDTree(support).query(queries, k=1)[1]
        input_flat = support_history[nearest_indices]
    elif spec.input_interpolation == "linear":
        input_flat = np.asarray(LinearNDInterpolator(support, support_history, fill_value=np.nan)(queries))
        missing = ~np.isfinite(input_flat).all(axis=1)
        if missing.any():
            nearest_indices = cKDTree(support).query(queries[missing], k=1)[1]
            input_flat[missing] = support_history[nearest_indices]
    else:
        raise ValueError(spec.input_interpolation)
    input_maps = input_flat.T.reshape(spec.history_length, spec.grid_size, spec.grid_size).astype(np.float32)
    target_map = analytic_field(grid_x, grid_y, float(spec.target_step), spec.scenario)
    eligible = np.ones((spec.grid_size, spec.grid_size), dtype=bool)
    train_mask, val_mask, test_mask = _build_spatial_tile_split_masks(
        eligible_mask=eligible,
        seed=spec.split_seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        tile_size=spec.tile_size,
    )
    input_valid = np.isfinite(input_maps)
    dense_task = DenseForecastTask(
        input_maps=np.nan_to_num(input_maps, nan=0.0),
        target_map=target_map.astype(np.float32),
        input_valid_mask=input_valid,
        target_valid_mask=eligible,
        history_coverage=input_valid.mean(axis=0).astype(np.float32),
        eligible_mask=eligible,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        interpolation_method=f"{spec.input_interpolation}_inputs_analytic_target",
        csv_path=Path(f"synthetic://{spec.scenario}/seed-{spec.seed}"),
    )
    split_codes = np.full(spec.grid_size * spec.grid_size, -1, dtype=np.int8)
    split_codes[train_mask.ravel()] = 0
    split_codes[val_mask.ravel()] = 1
    split_codes[test_mask.ravel()] = 2
    train_indices = np.flatnonzero(split_codes == 0)
    val_indices = np.flatnonzero(split_codes == 1)
    test_indices = np.flatnonzero(split_codes == 2)
    distance_to_support = cKDTree(support).query(queries, k=1)[0].reshape(spec.grid_size, spec.grid_size)
    metadata: dict[str, object] = {
        "protocol": "analytic-known-truth-v1",
        "scenario": spec.scenario,
        "grid_size": spec.grid_size,
        "history_length": spec.history_length,
        "target_step": spec.target_step,
        "support_points": spec.support_points,
        "observation_noise_std": spec.observation_noise_std,
        "data_seed": spec.seed,
        "split_seed": spec.split_seed,
        "input_interpolation": spec.input_interpolation,
        "target_source": "analytic_function",
        "test_target_used_in_any_target_grid": False,
        "distance_to_support_mean": float(distance_to_support.mean()),
        "distance_to_support_p95": float(np.quantile(distance_to_support, 0.95)),
    }
    return RawHoldoutTask(
        dense_task=dense_task,
        raw_points=queries,
        raw_target=target_map.ravel().astype(np.float32),
        raw_split_codes=split_codes,
        raw_block_ids=np.full(len(queries), -1, dtype=np.int32),
        easting_axis=axis,
        northing_axis=axis,
        train_target_source_indices=train_indices,
        val_target_source_indices=val_indices,
        test_target_indices=test_indices,
        metadata=metadata,
    )


def build_synthetic_support_holdout(
    spec: SyntheticTruthSpec,
) -> tuple[RawHoldoutTask, np.ndarray, np.ndarray]:
    """Return analytic targets at the original irregular measurement support.

    The dense history grid is still constructed with the requested input
    interpolation operator, but all train/validation/test supervision is
    evaluated directly at the irregular support points.  This separates
    measurement forecasting from interpolator self-consistency.
    """
    dense_holder = build_synthetic_truth_task(spec)
    support = sample_irregular_support(spec)
    rng = np.random.default_rng(spec.seed)
    history_steps = np.arange(spec.history_length, dtype=np.float64)
    support_history = np.stack(
        [analytic_field(support[:, 0], support[:, 1], float(step), spec.scenario) for step in history_steps],
        axis=1,
    )
    support_history += rng.normal(
        0.0,
        spec.observation_noise_std,
        size=support_history.shape,
    ).astype(np.float32)
    support_target = analytic_field(
        support[:, 0],
        support[:, 1],
        float(spec.target_step),
        spec.scenario,
    )
    east_index = np.rint(support[:, 0] * (spec.grid_size - 1)).astype(np.int64)
    north_index = np.rint(support[:, 1] * (spec.grid_size - 1)).astype(np.int64)
    split_codes = np.full(len(support), -1, dtype=np.int8)
    split_codes[dense_holder.dense_task.train_mask[east_index, north_index]] = 0
    split_codes[dense_holder.dense_task.val_mask[east_index, north_index]] = 1
    split_codes[dense_holder.dense_task.test_mask[east_index, north_index]] = 2
    train_indices = np.flatnonzero(split_codes == 0)
    val_indices = np.flatnonzero(split_codes == 1)
    test_indices = np.flatnonzero(split_codes == 2)
    if min(len(train_indices), len(val_indices), len(test_indices)) == 0:
        raise ValueError("Synthetic irregular support produced an empty split.")
    metadata = {
        **dense_holder.metadata,
        "protocol": "analytic-irregular-support-holdout-v2",
        "raw_support_point_count": int(len(support)),
        "raw_train_points": int(len(train_indices)),
        "raw_val_points": int(len(val_indices)),
        "raw_test_points": int(len(test_indices)),
        "test_target_used_in_any_target_grid": False,
    }
    raw_task = RawHoldoutTask(
        dense_task=dense_holder.dense_task,
        raw_points=support.astype(np.float64),
        raw_target=support_target.astype(np.float32),
        raw_split_codes=split_codes,
        raw_block_ids=np.full(len(support), -1, dtype=np.int32),
        easting_axis=dense_holder.easting_axis,
        northing_axis=dense_holder.northing_axis,
        train_target_source_indices=train_indices,
        val_target_source_indices=val_indices,
        test_target_indices=test_indices,
        metadata=metadata,
    )
    return raw_task, support_history.astype(np.float32), support.astype(np.float64)


def analytic_truth_diagnostics(
    prediction: np.ndarray,
    raw_task: RawHoldoutTask,
    *,
    support_points: np.ndarray | None = None,
) -> dict[str, float]:
    truth = raw_task.dense_task.target_map
    test = raw_task.dense_task.test_mask
    pred_grad = np.gradient(prediction.astype(np.float64))
    truth_grad = np.gradient(truth.astype(np.float64))
    grad_error = (pred_grad[0] - truth_grad[0]) ** 2 + (pred_grad[1] - truth_grad[1]) ** 2
    change = np.abs(truth - raw_task.dense_task.input_maps[-1])
    threshold = float(np.quantile(change[test], 0.90))
    extreme = test & (change >= threshold)
    diagnostics = {
        "gradient_vector_rmse": float(np.sqrt(np.mean(grad_error[test]))),
        "extreme_change_rmse": float(np.sqrt(np.mean((prediction[extreme] - truth[extreme]) ** 2))),
        "extreme_change_threshold": threshold,
        "peak_absolute_truth": float(np.max(np.abs(truth[test]))),
        "peak_absolute_prediction": float(np.max(np.abs(prediction[test]))),
        "peak_amplitude_absolute_error": float(
            abs(np.max(np.abs(prediction[test])) - np.max(np.abs(truth[test])))
        ),
    }
    if support_points is not None:
        axis = raw_task.easting_axis
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
        queries = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        distances = cKDTree(np.asarray(support_points, dtype=np.float64)).query(queries, k=1)[0].reshape(truth.shape)
        test_distances = distances[test]
        q50, q90 = np.quantile(test_distances, [0.50, 0.90])
        error_sq = (prediction - truth) ** 2
        near = test & (distances <= q50)
        middle = test & (distances > q50) & (distances <= q90)
        far = test & (distances > q90)
        diagnostics.update(
            {
                "support_distance_test_q50": float(q50),
                "support_distance_test_q90": float(q90),
                "near_support_rmse": float(np.sqrt(np.mean(error_sq[near]))),
                "middle_support_rmse": float(np.sqrt(np.mean(error_sq[middle]))),
                "far_support_rmse": float(np.sqrt(np.mean(error_sq[far]))),
            }
        )
    return diagnostics
