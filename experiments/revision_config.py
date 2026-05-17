"""
Revision-aligned project configuration.

Revision skeleton alignment:
- Section 4 / "Reproducibility-critical settings"
- Section 9 / minimum placeholders for data size, split, and hyperparameters

This file centralizes the current code-supported experiment scope so that all
new scripts use the same task definition, split policy, and output structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_FILENAME = "EGMS_L3_E32N34_100km_U_2018_2022_1.csv"


def _default_output_root() -> Path:
    return PROJECT_ROOT / "revision_outputs"


def _default_task_cache_root() -> Path:
    return _default_output_root() / "_task_cache"


@dataclass
class RevisionConfig:
    csv_filename: str = DEFAULT_CSV_FILENAME
    csv_path: str | None = None
    grid_size: int = 256
    history_start_col: int = 11
    history_length: int = 300
    target_col: int = 312
    interpolation_method: str = "linear"
    min_history_coverage: float = 0.0
    split_strategy: str = "spatial_tile"
    tile_size: int = 32

    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    cnn_hidden_dim: int = 128
    cnn_learning_rate: float = 3e-4
    cnn_weight_decay: float = 1e-5
    cnn_epochs: int = 60
    cnn_patience: int = 12
    patch_size: int = 16
    patch_stride: int = 8
    patch_min_valid_pixels: int = 24
    patch_batch_size: int = 16
    temporal_hybrid_recent_lags: int = 8
    temporal_hybrid_recent_scale_init: float = 1.0
    temporal_hybrid_correction_scale_init: float = 0.1
    nontransformer_hybrid_hidden_channels: int = 64

    lasso_alpha: float = 1e-3
    lasso_max_iter: int = 5000
    lasso_learning_rate: float = 2e-2
    lasso_epochs: int = 600
    lasso_patience: int = 60

    random_forest_n_estimators: int = 300
    random_forest_max_depth: int | None = None
    random_forest_min_samples_leaf: int = 1

    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    lightgbm_feature_fraction: float = 0.8
    lightgbm_bagging_fraction: float = 0.8
    lightgbm_bagging_freq: int = 5
    lightgbm_num_boost_round: int = 300
    lightgbm_early_stopping_rounds: int = 30
    lightgbm_device_type: str = "auto"

    tcn_hidden_channels: int = 64
    tcn_num_layers: int = 4
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1

    convlstm_hidden_dim: int = 64
    convlstm_num_layers: int = 1
    convlstm_kernel_size: int = 3

    idw_power: float = 2.0
    idw_neighbors: int = 8
    rbf_neighbors: int = 64
    rbf_smoothing: float = 0.0
    rbf_kernel: str = "linear"
    interpolation_holdout_fraction: float = 0.02
    interpolation_holdout_max_points: int = 4000

    quality_col: int = 4
    diag_bins: int = 20
    output_root: Path = field(default_factory=_default_output_root)
    task_cache_root: Path = field(default_factory=_default_task_cache_root)
    use_task_cache: bool = True

    def resolve_csv_path(self) -> Path:
        if self.csv_path is not None:
            path = Path(self.csv_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"CSV path does not exist: {path}")
            return path

        candidates = [
            PROJECT_ROOT / self.csv_filename,
            PROJECT_ROOT / "datasets" / self.csv_filename,
            PROJECT_ROOT.parent / self.csv_filename,
            PROJECT_ROOT.parent / "datasets" / self.csv_filename,
            PROJECT_ROOT.parent / "pytorch-tcn-main" / self.csv_filename,
            Path.home() / "Desktop" / "pytorch-tcn-main" / self.csv_filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            "Could not locate the EGMS CSV. Pass --csv-path explicitly or place "
            f"{self.csv_filename} next to the project."
        )

    def output_dir(self, model_name: str, interpolation_method: str | None = None) -> Path:
        method = interpolation_method or self.interpolation_method
        return self.output_root / model_name / method

    def split_ratios(self) -> Dict[str, float]:
        return {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }

    def as_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["task_cache_root"] = str(self.task_cache_root)
        data["resolved_csv_path"] = str(self.resolve_csv_path())
        return data
