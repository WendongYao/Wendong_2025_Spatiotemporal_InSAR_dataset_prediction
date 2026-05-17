"""
Preflight validation for the revision-aligned standalone experiment bundle.

Revision skeleton alignment:
- Section 3.2 / task definition, interpolation, and masked split protocol
- Section 3.6 / verifies evaluation domain sizes before long-running experiments
- Section 4 / ensures the first-round comparable experiments use the same setup
"""

from __future__ import annotations

import argparse
import json

from revision_config import RevisionConfig
from revision_utils import build_dense_forecast_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight preflight check for the revision-aligned task.")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional explicit path to the EGMS CSV.")
    parser.add_argument("--interpolation", type=str, default="linear", help="Interpolation method for dense map reconstruction.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for the pixel split.")
    args = parser.parse_args()

    config = RevisionConfig(
        csv_path=args.csv_path,
        interpolation_method=args.interpolation,
        split_seed=args.split_seed,
    )
    task = build_dense_forecast_task(config, interpolation_method=args.interpolation)

    payload = {
        "csv_path": str(task.csv_path),
        "interpolation_method": task.interpolation_method,
        "input_shape": list(task.input_maps.shape),
        "target_shape": list(task.target_map.shape),
        "history_length": int(task.input_maps.shape[0]),
        "target_valid_pixels": int(task.target_valid_mask.sum()),
        "eligible_pixels": int(task.eligible_mask.sum()),
        "train_pixels": int(task.train_mask.sum()),
        "val_pixels": int(task.val_mask.sum()),
        "test_pixels": int(task.test_mask.sum()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
