from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_ext"))

from synthetic_truth_data import (  # noqa: E402
    SyntheticTruthSpec,
    analytic_field,
    analytic_truth_diagnostics,
    build_synthetic_truth_task,
    sample_irregular_support,
)


def test_analytic_target_is_not_idw_target() -> None:
    spec = SyntheticTruthSpec(grid_size=32, support_points=96, scenario="composite", seed=7, split_seed=11, tile_size=8)
    task = build_synthetic_truth_task(spec)
    truth = task.dense_task.target_map
    assert task.metadata["target_source"] == "analytic_function"
    assert task.metadata["test_target_used_in_any_target_grid"] is False
    assert truth.shape == (32, 32)
    assert np.isfinite(truth).all()
    assert np.std(truth - task.dense_task.input_maps[-1]) > 0.05


def test_scenarios_have_nonzero_target_gradients() -> None:
    axis = np.linspace(0.0, 1.0, 24)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    for scenario in ("seasonal_trend", "localized_acceleration", "moving_front", "composite"):
        field = analytic_field(x, y, 301.0, scenario)
        gx, gy = np.gradient(field)
        assert float(np.sqrt(np.mean(gx**2 + gy**2))) > 0.01


def test_analytic_diagnostics_include_support_distance_bins() -> None:
    spec = SyntheticTruthSpec(
        grid_size=32,
        history_length=8,
        target_step=9,
        support_points=64,
        seed=3,
        split_seed=5,
        tile_size=8,
    )
    task = build_synthetic_truth_task(spec)
    diagnostics = analytic_truth_diagnostics(
        task.dense_task.input_maps[-1],
        task,
        support_points=sample_irregular_support(spec),
    )
    assert diagnostics["far_support_rmse"] >= 0.0
    assert diagnostics["support_distance_test_q90"] >= diagnostics["support_distance_test_q50"]


def test_all_input_interpolators_produce_finite_history_grids() -> None:
    for method in ("idw", "linear", "nearest"):
        spec = SyntheticTruthSpec(
            grid_size=24,
            history_length=6,
            target_step=7,
            support_points=48,
            seed=9,
            split_seed=4,
            tile_size=8,
            input_interpolation=method,
        )
        task = build_synthetic_truth_task(spec)
        assert np.isfinite(task.dense_task.input_maps).all()
        assert task.metadata["input_interpolation"] == method
