from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(ROOT / "experiments_ext"))

from raw_holdout_data import (  # noqa: E402
    RawHoldoutSpec,
    SPLIT_TO_CODE,
    build_raw_holdout_task,
    cell_aggregated_metrics,
    idw_interpolate,
    raw_point_metrics,
)


def test_idw_exact_queries_recover_values() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    values = np.asarray([2.0, 4.0, 8.0], dtype=np.float32)
    result = idw_interpolate(points, values, points, neighbors=3, power=2.0)
    np.testing.assert_allclose(result, values, rtol=0, atol=1e-6)


def test_raw_target_test_points_are_pregrid_held_out(tmp_path: Path) -> None:
    csv_path = SOURCE_EXPERIMENTS / "examples" / "synthetic_egms_small.csv"
    spec = RawHoldoutSpec(csv_path=csv_path, tile="synthetic", grid_size=32, block_side=4, split_seed=42)
    task = build_raw_holdout_task(spec, cache_dir=tmp_path)
    assert np.intersect1d(task.test_target_indices, task.train_target_source_indices).size == 0
    assert np.intersect1d(task.test_target_indices, task.val_target_source_indices).size == 0
    assert task.metadata["test_target_used_in_any_target_grid"] is False
    assert np.all(task.raw_split_codes[task.test_target_indices] == SPLIT_TO_CODE["test"])
    metrics = raw_point_metrics(task.dense_task.input_maps[-1], task, split="test")
    assert metrics["n_points"] == len(task.test_target_indices)
    assert np.isfinite(metrics["rmse"])
    cell_metrics = cell_aggregated_metrics(task.dense_task.input_maps[-1], task, split="test")
    assert cell_metrics["cell_count"] <= metrics["n_points"]
    assert cell_metrics["mse_decomposition_error"] < 1e-5
