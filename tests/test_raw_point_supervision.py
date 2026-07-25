from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "experiments"
if not SOURCE.is_dir():
    SOURCE = ROOT / "experiments"
sys.path.insert(0, str(ROOT / "experiments_ext"))
sys.path.insert(0, str(SOURCE))

from revision_config import RevisionConfig  # noqa: E402
from raw_holdout_data import RawHoldoutSpec, build_raw_holdout_task  # noqa: E402
from raw_point_supervision import _raw_norm_stats, build_raw_point_patches  # noqa: E402


def test_each_raw_label_is_assigned_at_most_once(tmp_path: Path) -> None:
    csv_path = SOURCE / "examples" / "synthetic_egms_small.csv"
    spec = RawHoldoutSpec(csv_path=csv_path, tile="synthetic", grid_size=32, block_side=4, split_seed=42)
    raw_task = build_raw_holdout_task(spec, cache_dir=tmp_path)
    config = RevisionConfig(
        csv_path=str(csv_path),
        grid_size=32,
        split_seed=42,
        patch_size=16,
        patch_stride=8,
        patch_min_valid_pixels=1,
        patch_batch_size=2,
        use_task_cache=False,
    )
    norm_stats, residual, _ = _raw_norm_stats(raw_task)
    patches = build_raw_point_patches(
        raw_task,
        config,
        norm_stats,
        residual,
        split_code=0,
        max_points_per_patch=512,
        min_points_per_patch=1,
    )
    assert patches.raw_point_count <= len(raw_task.train_target_source_indices)
    assert patches.raw_point_count > 0
    assert np.all(np.abs(patches.sample_coordinates[patches.point_masks.astype(bool)]) <= 1.0001)
