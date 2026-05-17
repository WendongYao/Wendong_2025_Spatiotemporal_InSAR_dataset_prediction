"""
Additional Computers & Geosciences experiment suite aligned to the DOCX plan.

Revision skeleton alignment:
- E0 / metric sanity, hash traceability, and reproducibility audit
- E1 / naive geodetic baselines
- E2 / model-backend expansion under identical protocol
- E3 / validity-mask ablation
- E4 / interpolation sensitivity
- E5 / leakage-free versus random split comparison
- E6 / multi-seed aggregation and paired testing
- E7 / resolution, runtime, and memory scaling
- E10 / persistence and interpretability diagnostics
- E11 / reproducibility packaging and manifest generation

The code below intentionally uses only the standalone assets available in the
current project bundle.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
from scipy import stats

from deep_patch_models import run_patch_deep_model
from revision_config import PROJECT_ROOT, RevisionConfig
from revision_experiments import run_lasso_experiment, run_lightgbm_experiment
from revision_utils import (
    DenseForecastTask,
    build_dense_forecast_task,
    build_tabular_dataset,
    ensure_dir,
    interpolate_query_points,
    load_revision_dataframe,
    masked_regression_metrics,
    resolve_grid_coordinates,
    save_config_snapshot,
    save_error_diagnostics,
    save_map_comparison,
    save_metrics,
    save_prediction_map,
    save_split_bundle,
    set_random_seed,
    sha256_array,
    sha256_file,
    split_from_eligible_indices,
)


CG_SUITE_ROOT = PROJECT_ROOT / "revision_outputs" / "cg_suite"
CG_CONFIG_ROOT = PROJECT_ROOT / "configs"
CG_SPLIT_ROOT = PROJECT_ROOT / "splits"
CG_SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
CG_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

PRIMARY_MODELS = [
    "persistence",
    "linear_trend",
    "lasso",
    "random_forest",
    "lightgbm",
    "cnn_tcn",
    "cnn_lstm_maskaware",
]
SPLIT_COMPARISON_MODELS = [
    "persistence",
    "lightgbm",
    "cnn_tcn",
    "cnn_lstm_maskaware",
]
INTERPOLATION_MODELS = [
    "persistence",
    "lasso",
    "lightgbm",
    "cnn_lstm_maskaware",
]
INTERPOLATION_METHODS = ["linear", "nearest", "idw", "rbf"]
MANDATORY_SEEDS = [42, 43, 44, 45, 46]
SCALING_GRID_SIZES = [128, 256, 512]


def _with_updates(config: RevisionConfig, **updates) -> RevisionConfig:
    updates.setdefault("task_cache_root", config.task_cache_root)
    return replace(config, **updates)


def _model_output_dir(config: RevisionConfig, model_name: str, interpolation_method: str) -> Path:
    return ensure_dir(config.output_dir(model_name, interpolation_method) / f"split_seed_{config.split_seed}")


def _save_rows_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(payload: Dict[str, object], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_existing_metrics_if_available(
    config: RevisionConfig,
    model_name: str,
    interpolation_method: str | None = None,
) -> Dict[str, object] | None:
    output_dir = _model_output_dir(config, model_name, interpolation_method or config.interpolation_method)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return None


def _all_grid_metrics(y_true_map: np.ndarray, y_pred_map: np.ndarray) -> Dict[str, float]:
    all_mask = np.ones_like(y_true_map, dtype=bool)
    return masked_regression_metrics(y_true_map, y_pred_map, all_mask)


def _package_metrics(
    model_name: str,
    config: RevisionConfig,
    task: DenseForecastTask,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": model_name,
        "interpolation_method": task.interpolation_method,
        "csv_path": str(task.csv_path),
        "grid_size": int(task.target_map.shape[0]),
        "history_length": int(task.input_maps.shape[0]),
        "history_target_gap": int(config.target_col - (config.history_start_col + config.history_length - 1)),
        "split_strategy": config.split_strategy,
        "tile_size": int(config.tile_size),
        "eligible_pixels": int(task.eligible_mask.sum()),
        "train_pixels": int(task.train_mask.sum()),
        "val_pixels": int(task.val_mask.sum()),
        "test_pixels": int(task.test_mask.sum()),
    }
    payload.update(metrics)
    return payload


def _finalize_run_outputs(
    config: RevisionConfig,
    model_name: str,
    task: DenseForecastTask,
    pred_map: np.ndarray,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    output_dir = _model_output_dir(config, model_name, task.interpolation_method)
    save_config_snapshot(config, output_dir)
    save_split_bundle(task, output_dir)
    save_prediction_map(pred_map, output_dir)
    np.save(output_dir / "target_map.npy", task.target_map.astype(np.float32))
    np.save(output_dir / "last_input_map.npy", task.input_maps[-1].astype(np.float32))
    save_error_diagnostics(task.target_map, pred_map, task.test_mask, output_dir, n_bins=config.diag_bins)
    save_map_comparison(task.target_map, pred_map, task.target_valid_mask, output_dir)
    payload = _package_metrics(model_name, config, task, metrics)
    save_metrics(payload, output_dir)
    return payload


def _timed_existing_runner(
    config: RevisionConfig,
    model_name: str,
    runner: Callable[..., Dict[str, object]],
    *,
    interpolation_method: str | None = None,
    enable_shap: bool = False,
) -> Dict[str, object]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    if runner is run_lightgbm_experiment:
        payload = runner(config, interpolation_method=interpolation_method, enable_shap=enable_shap)
    else:
        payload = runner(config, interpolation_method=interpolation_method)
    runtime_seconds = time.perf_counter() - started

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    output_dir = _model_output_dir(
        config,
        model_name,
        interpolation_method or config.interpolation_method,
    )
    payload["runtime_seconds"] = float(runtime_seconds)
    payload["peak_gpu_memory_mb"] = float(peak_memory_mb)
    save_metrics(payload, output_dir)
    return payload


def _build_quality_map(config: RevisionConfig, task: DenseForecastTask) -> np.ndarray:
    df = load_revision_dataframe(config)
    quality_values = df.iloc[:, config.quality_col].astype(np.float32).values
    easting = df.iloc[:, 1].astype(float).values
    northing = df.iloc[:, 2].astype(float).values
    points = np.column_stack((easting, northing))
    grid_x, grid_y = resolve_grid_coordinates(easting, northing, config.grid_size)
    quality_grid = interpolate_query_points(
        points=points,
        values=quality_values,
        query_points=np.column_stack((grid_x.ravel(), grid_y.ravel())),
        method=task.interpolation_method,
        config=config,
        fill_missing=True,
    ).reshape(grid_x.shape)
    return quality_grid.astype(np.float32)


def _build_distance_to_measurements_map(config: RevisionConfig) -> np.ndarray:
    df = load_revision_dataframe(config)
    easting = df.iloc[:, 1].astype(float).values
    northing = df.iloc[:, 2].astype(float).values
    points = np.column_stack((easting, northing))
    grid_x, grid_y = resolve_grid_coordinates(easting, northing, config.grid_size)
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, _ = tree.query(np.column_stack((grid_x.ravel(), grid_y.ravel())), k=1)
    return np.asarray(distances, dtype=np.float32).reshape(grid_x.shape)


def run_persistence_baseline(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    task = build_dense_forecast_task(config, interpolation_method)
    started = time.perf_counter()
    pred_map = task.input_maps[-1].astype(np.float32)
    runtime_seconds = time.perf_counter() - started
    metrics = masked_regression_metrics(task.target_map, pred_map, task.test_mask)
    metrics.update(
        {
            "device": "analytical",
            "runtime_seconds": float(runtime_seconds),
            "peak_gpu_memory_mb": 0.0,
        }
    )
    metrics.update({f"full_grid_{k}": v for k, v in _all_grid_metrics(task.target_map, pred_map).items()})
    return _finalize_run_outputs(config, "persistence", task, pred_map, metrics)


def run_linear_trend_baseline(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    task = build_dense_forecast_task(config, interpolation_method)
    started = time.perf_counter()

    n_time = task.input_maps.shape[0]
    x = np.arange(n_time, dtype=np.float32)
    x_mean = float(x.mean())
    x_var = float(np.sum((x - x_mean) ** 2))
    history_flat = task.input_maps.reshape(n_time, -1).astype(np.float32)
    y_mean = history_flat.mean(axis=0)
    cov = np.sum((x[:, None] - x_mean) * (history_flat - y_mean[None, :]), axis=0)
    slope = cov / max(x_var, 1e-12)
    intercept = y_mean - slope * x_mean
    target_position = float(config.target_col - config.history_start_col)
    pred_map = (intercept + slope * target_position).reshape(task.target_map.shape).astype(np.float32)
    runtime_seconds = time.perf_counter() - started

    metrics = masked_regression_metrics(task.target_map, pred_map, task.test_mask)
    metrics.update(
        {
            "device": "analytical",
            "runtime_seconds": float(runtime_seconds),
            "peak_gpu_memory_mb": 0.0,
        }
    )
    metrics.update({f"full_grid_{k}": v for k, v in _all_grid_metrics(task.target_map, pred_map).items()})
    return _finalize_run_outputs(config, "linear_trend", task, pred_map, metrics)


def run_random_forest_baseline(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    from sklearn.ensemble import RandomForestRegressor

    task = build_dense_forecast_task(config, interpolation_method)
    output_dir = _model_output_dir(config, "random_forest", task.interpolation_method)
    save_config_snapshot(config, output_dir)
    save_split_bundle(task, output_dir)

    X_all, y_all, eligible_indices = build_tabular_dataset(task)
    split_positions = split_from_eligible_indices(task, eligible_indices)
    X_train = X_all[split_positions["train"]]
    y_train = y_all[split_positions["train"]]

    started = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators=config.random_forest_n_estimators,
        max_depth=config.random_forest_max_depth,
        min_samples_leaf=config.random_forest_min_samples_leaf,
        random_state=config.split_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    full_predictions = model.predict(X_all).astype(np.float32)
    runtime_seconds = time.perf_counter() - started

    pred_map = np.full(task.target_map.shape, np.nan, dtype=np.float32)
    pred_map.reshape(-1)[eligible_indices] = full_predictions
    pred_map = np.nan_to_num(pred_map, nan=0.0)

    metrics = masked_regression_metrics(task.target_map, pred_map, task.test_mask)
    metrics.update(
        {
            "device": "cpu",
            "runtime_seconds": float(runtime_seconds),
            "peak_gpu_memory_mb": 0.0,
        }
    )
    metrics.update({f"full_grid_{k}": v for k, v in _all_grid_metrics(task.target_map, pred_map).items()})
    np.save(output_dir / "target_map.npy", task.target_map.astype(np.float32))
    np.save(output_dir / "last_input_map.npy", task.input_maps[-1].astype(np.float32))
    save_prediction_map(pred_map, output_dir)
    save_error_diagnostics(task.target_map, pred_map, task.test_mask, output_dir, n_bins=config.diag_bins)
    save_map_comparison(task.target_map, pred_map, task.target_valid_mask, output_dir)
    payload = _package_metrics("random_forest", config, task, metrics)
    save_metrics(payload, output_dir)
    return payload


class _MaskAwareCNNLSTMBuilder:
    def __init__(self, hidden_dim: int, output_size: tuple[int, int], in_channels: int) -> None:
        import torch.nn as nn

        h, w = output_size
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError("The CNN-LSTM encoder expects spatial dimensions divisible by 8.")

        class _Model(nn.Module):
            def __init__(self, hidden_dim_: int, output_size_: tuple[int, int], in_channels_: int) -> None:
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(in_channels_, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                feature_dim = 128 * (output_size_[0] // 8) * (output_size_[1] // 8)
                self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim_, batch_first=True)
                self.fc = nn.Linear(hidden_dim_, output_size_[0] * output_size_[1])
                self.output_size = output_size_

            def forward(self, x):
                batch_size, time_steps, channels, height, width = x.shape
                x = x.view(batch_size * time_steps, channels, height, width)
                cnn_features = self.cnn(x)
                cnn_features = cnn_features.view(batch_size, time_steps, -1)
                lstm_out, _ = self.lstm(cnn_features)
                final_feature = lstm_out[:, -1, :]
                out = self.fc(final_feature)
                return out.view(batch_size, 1, self.output_size[0], self.output_size[1])

        self.model_class = _Model
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.in_channels = in_channels

    def build(self):
        return self.model_class(self.hidden_dim, self.output_size, self.in_channels)


class _TemporalResidualBlock:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        import torch.nn as nn

        padding = dilation * (kernel_size - 1)

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

            def forward(self, x):
                out = self.net(x)
                out = out[..., : x.shape[-1]]
                residual = x if self.downsample is None else self.downsample(x)
                return out + residual

        self.block_class = _Block

    def build(self):
        return self.block_class()


class _CNNTCNBuilder:
    def __init__(self, hidden_channels: int, output_size: tuple[int, int], in_channels: int, config: RevisionConfig) -> None:
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self, hidden_channels_: int, output_size_: tuple[int, int], in_channels_: int, config_: RevisionConfig) -> None:
                super().__init__()
                self.spatial_encoder = nn.Sequential(
                    nn.Conv2d(in_channels_, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                layers = []
                in_ch = 128
                for layer_idx in range(config_.tcn_num_layers):
                    dilation = 2**layer_idx
                    layers.append(
                        _TemporalResidualBlock(
                            in_channels=in_ch,
                            out_channels=hidden_channels_,
                            kernel_size=config_.tcn_kernel_size,
                            dilation=dilation,
                            dropout=config_.tcn_dropout,
                        ).build()
                    )
                    in_ch = hidden_channels_
                self.tcn = nn.Sequential(*layers)
                self.head = nn.Sequential(
                    nn.Linear(hidden_channels_, hidden_channels_ * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_channels_ * 4, output_size_[0] * output_size_[1]),
                )
                self.output_size = output_size_

            def forward(self, x):
                batch_size, time_steps, channels, height, width = x.shape
                x = x.view(batch_size * time_steps, channels, height, width)
                features = self.spatial_encoder(x).view(batch_size, time_steps, 128)
                features = features.transpose(1, 2)
                temporal = self.tcn(features)
                final_feature = temporal[:, :, -1]
                out = self.head(final_feature)
                return out.view(batch_size, 1, self.output_size[0], self.output_size[1])

        self.model_class = _Model
        self.hidden_channels = hidden_channels
        self.output_size = output_size
        self.in_channels = in_channels
        self.config = config

    def build(self):
        return self.model_class(self.hidden_channels, self.output_size, self.in_channels, self.config)


def _build_model_input(task: DenseForecastTask, use_input_mask: bool) -> np.ndarray:
    if use_input_mask:
        return np.stack([task.input_maps, task.input_valid_mask.astype(np.float32)], axis=1)
    return task.input_maps[:, np.newaxis, :, :].astype(np.float32)


def _masked_mse_torch(pred, target, mask):
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    squared_error = (pred - target) ** 2
    return (squared_error * mask_4d).sum() / mask_4d.sum().clamp_min(1.0)


def _train_dense_torch_model(
    config: RevisionConfig,
    model_name: str,
    model_builder,
    *,
    use_input_mask: bool,
    loss_mask_mode: str,
    metric_mask_mode: str,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    import torch
    import torch.optim as optim

    set_random_seed(config.split_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    task = build_dense_forecast_task(config, interpolation_method)
    output_dir = _model_output_dir(config, model_name, task.interpolation_method)
    save_config_snapshot(config, output_dir)
    save_split_bundle(task, output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    x_tensor = torch.tensor(_build_model_input(task, use_input_mask)[np.newaxis, ...], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(task.target_map[np.newaxis, np.newaxis, :, :], dtype=torch.float32, device=device)
    train_mask = torch.tensor(task.train_mask, dtype=torch.float32, device=device)
    val_mask = torch.tensor(task.val_mask, dtype=torch.float32, device=device)
    eval_mask = task.test_mask if metric_mask_mode == "masked" else np.ones_like(task.test_mask, dtype=bool)

    model = model_builder.build().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.cnn_learning_rate, weight_decay=config.cnn_weight_decay)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    history_rows: List[Dict[str, float]] = []
    patience_counter = 0
    started = time.perf_counter()

    for epoch in range(1, config.cnn_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_train = model(x_tensor)
        if loss_mask_mode == "masked":
            train_loss = _masked_mse_torch(pred_train, y_tensor, train_mask)
            val_mask_tensor = val_mask
        elif loss_mask_mode == "unmasked":
            train_loss = torch.mean((pred_train - y_tensor) ** 2)
            val_mask_tensor = torch.ones_like(val_mask)
        else:
            raise ValueError(f"Unsupported loss mask mode: {loss_mask_mode}")
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_tensor)
            val_loss = _masked_mse_torch(pred_val, y_tensor, val_mask_tensor)

        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item()),
            }
        )

        if val_loss.item() < best_val - 1e-9:
            best_val = float(val_loss.item())
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.cnn_patience:
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / f"{model_name}_best_model.pth")
    _save_rows_csv(history_rows, output_dir / "training_history.csv")

    model.eval()
    with torch.no_grad():
        pred_tensor = model(x_tensor)
    pred_map = pred_tensor.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    runtime_seconds = time.perf_counter() - started
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

    masked_metrics = masked_regression_metrics(task.target_map, pred_map, eval_mask)
    full_metrics = _all_grid_metrics(task.target_map, pred_map)
    metrics: Dict[str, object] = {
        **masked_metrics,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "device": str(device),
        "runtime_seconds": float(runtime_seconds),
        "peak_gpu_memory_mb": float(peak_memory_mb),
        "input_mask_channel": bool(use_input_mask),
        "loss_mask_mode": loss_mask_mode,
        "metric_mask_mode": metric_mask_mode,
    }
    metrics.update({f"full_grid_{k}": v for k, v in full_metrics.items()})
    return _finalize_run_outputs(config, model_name, task, pred_map, metrics)


def run_cnnlstm_maskaware_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="cnn_lstm_maskaware",
        model_kind="cnn_lstm",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_cnnlstm_noinputmask_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="cnn_lstm_noinputmask",
        model_kind="cnn_lstm",
        interpolation_method=interpolation_method,
        use_input_mask=False,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_cnnlstm_nolossmask_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="cnn_lstm_nolossmask",
        model_kind="cnn_lstm",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="all",
        metric_mask_mode="all",
    )


def run_cnnlstm_metriconlymask_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="cnn_lstm_metriconlymask",
        model_kind="cnn_lstm",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="all",
        metric_mask_mode="masked",
    )


def run_cnn_tcn_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="cnn_tcn",
        model_kind="cnn_tcn",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_cnn_lstm_hybrid_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="cnn_lstm_hybrid",
        model_kind="cnn_lstm_hybrid",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_cnn_tcn_hybrid_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="cnn_tcn_hybrid",
        model_kind="cnn_tcn_hybrid",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_conv_lstm_residual_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="conv_lstm_residual",
        model_kind="conv_lstm_residual",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_temporal_channel_cnn_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="temporal_channel_cnn",
        model_kind="temporal_channel_cnn",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_patch_unet_residual_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="patch_unet_residual",
        model_kind="patch_unet_residual",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_conv3d_residual_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="conv3d_residual",
        model_kind="conv3d_residual",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_temporal_linear_hybrid_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return run_patch_deep_model(
        config=config,
        model_name="temporal_linear_hybrid",
        model_kind="temporal_linear_hybrid",
        interpolation_method=interpolation_method,
        use_input_mask=True,
        loss_mask_mode="masked",
        metric_mask_mode="masked",
    )


def run_model_by_name(
    model_name: str,
    config: RevisionConfig,
    interpolation_method: str | None = None,
    *,
    enable_shap: bool = False,
    reuse_existing: bool = True,
) -> Dict[str, object]:
    if reuse_existing:
        cached = _load_existing_metrics_if_available(config, model_name, interpolation_method)
        if cached is not None:
            return cached

    if model_name == "persistence":
        return run_persistence_baseline(config, interpolation_method=interpolation_method)
    if model_name == "linear_trend":
        return run_linear_trend_baseline(config, interpolation_method=interpolation_method)
    if model_name == "lasso":
        return _timed_existing_runner(config, "lasso", run_lasso_experiment, interpolation_method=interpolation_method)
    if model_name == "random_forest":
        return run_random_forest_baseline(config, interpolation_method=interpolation_method)
    if model_name == "lightgbm":
        return _timed_existing_runner(
            config,
            "lightgbm",
            run_lightgbm_experiment,
            interpolation_method=interpolation_method,
            enable_shap=enable_shap,
        )
    if model_name == "cnn_lstm_maskaware":
        return run_cnnlstm_maskaware_experiment(config, interpolation_method=interpolation_method)
    if model_name == "cnn_lstm_noinputmask":
        return run_cnnlstm_noinputmask_experiment(config, interpolation_method=interpolation_method)
    if model_name == "cnn_lstm_nolossmask":
        return run_cnnlstm_nolossmask_experiment(config, interpolation_method=interpolation_method)
    if model_name == "cnn_lstm_metriconlymask":
        return run_cnnlstm_metriconlymask_experiment(config, interpolation_method=interpolation_method)
    if model_name == "cnn_tcn":
        return run_cnn_tcn_experiment(config, interpolation_method=interpolation_method)
    if model_name == "cnn_lstm_hybrid":
        return run_cnn_lstm_hybrid_experiment(config, interpolation_method=interpolation_method)
    if model_name == "cnn_tcn_hybrid":
        return run_cnn_tcn_hybrid_experiment(config, interpolation_method=interpolation_method)
    if model_name == "conv_lstm_residual":
        return run_conv_lstm_residual_experiment(config, interpolation_method=interpolation_method)
    if model_name == "temporal_channel_cnn":
        return run_temporal_channel_cnn_experiment(config, interpolation_method=interpolation_method)
    if model_name == "patch_unet_residual":
        return run_patch_unet_residual_experiment(config, interpolation_method=interpolation_method)
    if model_name == "conv3d_residual":
        return run_conv3d_residual_experiment(config, interpolation_method=interpolation_method)
    if model_name == "temporal_linear_hybrid":
        return run_temporal_linear_hybrid_experiment(config, interpolation_method=interpolation_method)
    raise ValueError(f"Unsupported model: {model_name}")


def _seed_run_root(experiment_id: str, *parts: str) -> Path:
    return ensure_dir(CG_SUITE_ROOT / experiment_id / Path(*parts))


def run_primary_model_suite(
    base_config: RevisionConfig,
    seeds: Iterable[int] = MANDATORY_SEEDS,
    model_names: Iterable[str] = PRIMARY_MODELS,
) -> Path:
    output_root = _seed_run_root("E2_primary_multiseed", "spatial_tile", f"grid_{base_config.grid_size}")
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for model_name in model_names:
            config = _with_updates(
                base_config,
                split_seed=int(seed),
                split_strategy="spatial_tile",
                output_root=output_root,
            )
            payload = run_model_by_name(model_name, config, interpolation_method=base_config.interpolation_method, enable_shap=(model_name == "lightgbm"))
            rows.append(
                {
                    "seed": int(seed),
                    "model": model_name,
                    "split_strategy": config.split_strategy,
                    "rmse": payload["rmse"],
                    "mae": payload["mae"],
                    "mse": payload["mse"],
                    "r2": payload["r2"],
                    "runtime_seconds": payload.get("runtime_seconds", float("nan")),
                    "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb", float("nan")),
                    "device": payload.get("device", payload.get("device_type", "n/a")),
                    "metrics_path": str(_model_output_dir(config, model_name, base_config.interpolation_method) / "metrics.json"),
                }
            )
    _save_rows_csv(rows, output_root / "seed_level_results.csv")
    return output_root


def run_mask_ablation_suite(
    base_config: RevisionConfig,
    seeds: Iterable[int] = MANDATORY_SEEDS,
) -> Path:
    output_root = _seed_run_root("E3_mask_ablation", "spatial_tile", f"grid_{base_config.grid_size}")
    variants = [
        "cnn_lstm_maskaware",
        "cnn_lstm_noinputmask",
        "cnn_lstm_nolossmask",
        "cnn_lstm_metriconlymask",
    ]
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for model_name in variants:
            config = _with_updates(
                base_config,
                split_seed=int(seed),
                split_strategy="spatial_tile",
                output_root=output_root,
            )
            payload = run_model_by_name(model_name, config, interpolation_method=base_config.interpolation_method)
            rows.append(
                {
                    "seed": int(seed),
                    "variant": model_name,
                    "rmse": payload["rmse"],
                    "mae": payload["mae"],
                    "mse": payload["mse"],
                    "r2": payload["r2"],
                    "full_grid_rmse": payload.get("full_grid_rmse", float("nan")),
                    "runtime_seconds": payload.get("runtime_seconds", float("nan")),
                    "device": payload.get("device", "n/a"),
                }
            )
    _save_rows_csv(rows, output_root / "seed_level_mask_ablation.csv")
    return output_root


def _interpolation_holdout_metrics(
    config: RevisionConfig,
    method: str,
    seed: int,
) -> Dict[str, object]:
    df = load_revision_dataframe(config)
    points = np.column_stack(
        (
            df.iloc[:, 1].astype(float).values,
            df.iloc[:, 2].astype(float).values,
        )
    )
    target_values = df.iloc[:, config.target_col].astype(np.float32).values
    rng = np.random.default_rng(seed)
    n_points = len(points)
    max_holdout = min(int(n_points * config.interpolation_holdout_fraction), config.interpolation_holdout_max_points)
    holdout_size = max(1000, max_holdout)
    holdout_size = min(holdout_size, n_points // 5)
    holdout_idx = rng.choice(n_points, size=holdout_size, replace=False)
    train_mask = np.ones(n_points, dtype=bool)
    train_mask[holdout_idx] = False

    pred = interpolate_query_points(
        points=points[train_mask],
        values=target_values[train_mask],
        query_points=points[holdout_idx],
        method=method,
        config=config,
        fill_missing=True,
    )
    truth = target_values[holdout_idx]
    residual = pred - truth
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    return {
        "seed": int(seed),
        "method": method,
        "holdout_points": int(holdout_size),
        "point_holdout_mse": mse,
        "point_holdout_rmse": rmse,
        "point_holdout_mae": mae,
    }


def run_interpolation_sensitivity_suite(
    base_config: RevisionConfig,
    seeds: Iterable[int] = MANDATORY_SEEDS,
    methods: Iterable[str] = INTERPOLATION_METHODS,
    model_names: Iterable[str] = INTERPOLATION_MODELS,
) -> Path:
    output_root = _seed_run_root("E4_interpolation_sensitivity", "spatial_tile", f"grid_{base_config.grid_size}")
    forecast_rows: List[Dict[str, object]] = []
    holdout_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []

    reference_target = None
    for method in methods:
        method_config = _with_updates(
            base_config,
            split_strategy="spatial_tile",
            output_root=output_root / method,
        )
        task = build_dense_forecast_task(method_config, interpolation_method=method)
        if method == "linear":
            reference_target = task.target_map.copy()
        elif reference_target is not None:
            delta = task.target_map - reference_target
            delta_rows.append(
                {
                    "method": method,
                    "target_delta_mae_vs_linear": float(np.mean(np.abs(delta))),
                    "target_delta_rmse_vs_linear": float(np.sqrt(np.mean(delta**2))),
                    "target_corr_vs_linear": float(np.corrcoef(task.target_map.ravel(), reference_target.ravel())[0, 1]),
                }
            )

        for seed in seeds:
            holdout_rows.append(_interpolation_holdout_metrics(method_config, method, int(seed)))
            for model_name in model_names:
                config = _with_updates(
                    method_config,
                    split_seed=int(seed),
                    interpolation_method=method,
                    output_root=output_root / method,
                )
                payload = run_model_by_name(model_name, config, interpolation_method=method, enable_shap=False)
                forecast_rows.append(
                    {
                        "seed": int(seed),
                        "method": method,
                        "model": model_name,
                        "rmse": payload["rmse"],
                        "mae": payload["mae"],
                        "mse": payload["mse"],
                        "r2": payload["r2"],
                        "runtime_seconds": payload.get("runtime_seconds", float("nan")),
                    }
                )

    _save_rows_csv(forecast_rows, output_root / "forecast_metric_deltas.csv")
    _save_rows_csv(holdout_rows, output_root / "point_holdout_interpolation_error.csv")
    _save_rows_csv(delta_rows, output_root / "target_field_deltas.csv")
    return output_root


def run_split_comparison_suite(
    base_config: RevisionConfig,
    seeds: Iterable[int] = MANDATORY_SEEDS,
    model_names: Iterable[str] = SPLIT_COMPARISON_MODELS,
) -> Path:
    output_root = _seed_run_root("E5_split_comparison", f"grid_{base_config.grid_size}")
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        per_seed: Dict[str, Dict[str, float]] = {}
        for split_strategy in ["random_pixel", "spatial_tile"]:
            for model_name in model_names:
                config = _with_updates(
                    base_config,
                    split_seed=int(seed),
                    split_strategy=split_strategy,
                    output_root=output_root / split_strategy,
                )
                payload = run_model_by_name(model_name, config, interpolation_method=base_config.interpolation_method, enable_shap=False)
                rows.append(
                    {
                        "seed": int(seed),
                        "split_strategy": split_strategy,
                        "model": model_name,
                        "rmse": payload["rmse"],
                        "mae": payload["mae"],
                        "mse": payload["mse"],
                        "r2": payload["r2"],
                    }
                )
                per_seed.setdefault(model_name, {})[split_strategy] = float(payload["rmse"])

        for model_name, metrics in per_seed.items():
            if {"random_pixel", "spatial_tile"} <= set(metrics.keys()):
                leak_free = metrics["spatial_tile"]
                random_split = metrics["random_pixel"]
                rows.append(
                    {
                        "seed": int(seed),
                        "split_strategy": "comparison",
                        "model": model_name,
                        "rmse": random_split,
                        "mae": float("nan"),
                        "mse": float("nan"),
                        "r2": float("nan"),
                        "leakage_free_rmse": leak_free,
                        "inflation_optimism_pct": 100.0 * (leak_free - random_split) / max(leak_free, 1e-9),
                    }
                )
    _save_rows_csv(rows, output_root / "split_comparison_seed_level.csv")
    return output_root


def run_resolution_scaling_suite(
    base_config: RevisionConfig,
    seeds: Iterable[int] = MANDATORY_SEEDS,
    grid_sizes: Iterable[int] = SCALING_GRID_SIZES,
    model_names: Iterable[str] = ("persistence", "lasso", "lightgbm", "cnn_lstm_maskaware"),
) -> Path:
    output_root = _seed_run_root("E7_resolution_scaling")
    rows: List[Dict[str, object]] = []
    oom_failures: set[tuple[int, str]] = set()
    for grid_size in grid_sizes:
        for seed in seeds:
            for model_name in model_names:
                if (int(grid_size), model_name) in oom_failures:
                    rows.append(
                        {
                            "grid_size": int(grid_size),
                            "seed": int(seed),
                            "model": model_name,
                            "status": "skipped_after_oom",
                            "error_type": "OutOfMemoryError",
                            "error_message": "Skipped because this model-resolution pair already OOMed on an earlier seed.",
                        }
                    )
                    continue
                config = _with_updates(
                    base_config,
                    grid_size=int(grid_size),
                    split_seed=int(seed),
                    split_strategy="spatial_tile",
                    output_root=output_root / f"grid_{grid_size}",
                )
                try:
                    payload = run_model_by_name(model_name, config, interpolation_method=base_config.interpolation_method, enable_shap=False)
                    rows.append(
                        {
                            "grid_size": int(grid_size),
                            "seed": int(seed),
                            "model": model_name,
                            "status": "ok",
                            "rmse": payload["rmse"],
                            "mae": payload["mae"],
                            "mse": payload["mse"],
                            "r2": payload["r2"],
                            "runtime_seconds": payload.get("runtime_seconds", float("nan")),
                            "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb", float("nan")),
                            "device": payload.get("device", payload.get("device_type", "n/a")),
                        }
                    )
                except Exception as exc:
                    error_type = type(exc).__name__
                    error_message = str(exc)
                    if "out of memory" in error_message.lower() or "OutOfMemory" in error_type:
                        oom_failures.add((int(grid_size), model_name))
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    rows.append(
                        {
                            "grid_size": int(grid_size),
                            "seed": int(seed),
                            "model": model_name,
                            "status": "failed",
                            "error_type": error_type,
                            "error_message": error_message,
                        }
                    )
    _save_rows_csv(rows, output_root / "resolution_scaling_seed_level.csv")
    return output_root


def summarize_seed_table(
    seed_level_csv: Path,
    group_fields: List[str],
    metric_fields: List[str],
    output_csv: Path,
) -> List[Dict[str, object]]:
    import pandas as pd

    df = pd.read_csv(seed_level_csv)
    if df.empty:
        raise ValueError(f"No rows found in {seed_level_csv}")

    summary_rows: List[Dict[str, object]] = []
    grouped = df.groupby(group_fields, dropna=False)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {field: value for field, value in zip(group_fields, group_key)}
        for metric in metric_fields:
            if metric not in group_df.columns:
                continue
            series = group_df[metric].dropna()
            if len(series) == 0:
                continue
            row[f"{metric}_mean"] = float(series.mean())
            row[f"{metric}_std"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        summary_rows.append(row)

    _save_rows_csv(summary_rows, output_csv)
    return summary_rows


def run_paired_model_stats(
    seed_level_csv: Path,
    candidate_models: Iterable[str],
    proposed_model: str,
    output_json: Path,
) -> Dict[str, object]:
    import pandas as pd

    df = pd.read_csv(seed_level_csv)
    subset = df[df["model"].isin(list(candidate_models))].copy()
    mean_rmse = subset.groupby("model")["rmse"].mean().to_dict()
    strongest_baseline = min((m for m in candidate_models if m != proposed_model), key=lambda key: mean_rmse[key])

    proposed = subset[subset["model"] == proposed_model].sort_values("seed")
    baseline = subset[subset["model"] == strongest_baseline].sort_values("seed")
    merged = proposed[["seed", "rmse"]].merge(
        baseline[["seed", "rmse"]],
        on="seed",
        suffixes=("_proposed", "_baseline"),
    )
    diff = merged["rmse_baseline"].to_numpy(dtype=float) - merged["rmse_proposed"].to_numpy(dtype=float)

    shapiro_p = float(stats.shapiro(diff).pvalue) if len(diff) >= 3 else float("nan")
    if len(diff) >= 3 and shapiro_p >= 0.05:
        stat = stats.ttest_rel(merged["rmse_proposed"], merged["rmse_baseline"])
        test_name = "paired_t_test"
        p_value = float(stat.pvalue)
    else:
        stat = stats.wilcoxon(merged["rmse_proposed"], merged["rmse_baseline"], zero_method="wilcox")
        test_name = "wilcoxon_signed_rank"
        p_value = float(stat.pvalue)

    effect_size = float(diff.mean() / max(diff.std(ddof=1), 1e-12)) if len(diff) > 1 else float("nan")
    ci_half_width = float(stats.t.ppf(0.975, max(len(diff) - 1, 1)) * diff.std(ddof=1) / math.sqrt(max(len(diff), 1))) if len(diff) > 1 else 0.0
    payload = {
        "proposed_model": proposed_model,
        "strongest_baseline": strongest_baseline,
        "mean_rmse_by_model": {k: float(v) for k, v in mean_rmse.items()},
        "test_name": test_name,
        "p_value": p_value,
        "effect_size_cohens_dz": effect_size,
        "mean_rmse_delta_baseline_minus_proposed": float(diff.mean()),
        "ci95_low": float(diff.mean() - ci_half_width),
        "ci95_high": float(diff.mean() + ci_half_width),
        "n_seeds": int(len(diff)),
    }
    _save_json(payload, output_json)
    return payload


def run_metric_sanity_audit(output_roots: Iterable[Path], output_dir: Path) -> Path:
    rows: List[Dict[str, object]] = []
    for output_root in output_roots:
        for metrics_path in output_root.rglob("metrics.json"):
            run_dir = metrics_path.parent
            prediction_path = run_dir / "prediction_map.npy"
            target_path = run_dir / "target_map.npy"
            split_path = run_dir / "split_masks.npz"
            config_path = run_dir / "config_snapshot.json"
            if not prediction_path.exists() or not target_path.exists() or not split_path.exists():
                continue

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            pred_map = np.load(prediction_path)
            target_map = np.load(target_path)
            split_bundle = np.load(split_path)
            test_mask = split_bundle["test_mask"].astype(bool)
            recomputed = masked_regression_metrics(target_map, pred_map, test_mask)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "model": metrics.get("model", "unknown"),
                    "reported_rmse": float(metrics["rmse"]),
                    "reported_mse": float(metrics["mse"]),
                    "rmse_squared": float(metrics["rmse"]) ** 2,
                    "recomputed_rmse": float(recomputed["rmse"]),
                    "recomputed_mse": float(recomputed["mse"]),
                    "rmse_mse_abs_diff": abs(float(metrics["rmse"]) ** 2 - float(metrics["mse"])),
                    "mask_valid_count": int(test_mask.sum()),
                    "prediction_hash": sha256_file(prediction_path),
                    "target_hash": sha256_file(target_path),
                    "split_hash": sha256_file(split_path),
                    "config_hash": sha256_file(config_path) if config_path.exists() else "",
                    "metrics_hash": sha256_file(metrics_path),
                    "prediction_array_hash": sha256_array(pred_map),
                    "status": "pass" if abs(float(metrics["rmse"]) ** 2 - float(metrics["mse"])) < 1e-6 else "check",
                }
            )
    output_csv = output_dir / "metric_sanity_audit.csv"
    _save_rows_csv(rows, output_csv)
    return output_csv


def run_interpretability_suite(base_config: RevisionConfig, seed: int = 42) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd

    output_root = _seed_run_root("E10_interpretability", "spatial_tile", f"seed_{seed}")
    config = _with_updates(
        base_config,
        split_seed=int(seed),
        split_strategy="spatial_tile",
        output_root=output_root,
    )

    lightgbm_payload = run_model_by_name("lightgbm", config, enable_shap=True)
    cnn_payload = run_model_by_name("cnn_lstm_maskaware", config)
    tcn_payload = run_model_by_name("cnn_tcn", config)
    _ = (cnn_payload, tcn_payload)

    task = build_dense_forecast_task(config, interpolation_method=base_config.interpolation_method)
    last_input = task.input_maps[-1]
    rows: List[Dict[str, object]] = []
    for model_name in ["persistence", "lightgbm", "cnn_tcn", "cnn_lstm_maskaware"]:
        metrics_path = _model_output_dir(config, model_name, task.interpolation_method) / "metrics.json"
        pred_path = _model_output_dir(config, model_name, task.interpolation_method) / "prediction_map.npy"
        if not metrics_path.exists() or not pred_path.exists():
            payload = run_model_by_name(model_name, config, enable_shap=(model_name == "lightgbm"))
            del payload
        pred_map = np.load(pred_path)
        corr = np.corrcoef(pred_map[task.test_mask], last_input[task.test_mask])[0, 1]
        rows.append({"model": model_name, "persistence_similarity_corr": float(corr)})
    _save_rows_csv(rows, output_root / "persistence_similarity.csv")

    model_path = _model_output_dir(config, "lightgbm", task.interpolation_method) / "lightgbm_model.txt"
    if model_path.exists():
        import lightgbm as lgb
        import shap

        booster = lgb.Booster(model_file=str(model_path))
        X_all, _, eligible_indices = build_tabular_dataset(task)
        split_positions = split_from_eligible_indices(task, eligible_indices)
        X_val = X_all[split_positions["val"]]
        X_shap = X_val[: min(2000, len(X_val))]
        if len(X_shap) > 0:
            explainer = shap.TreeExplainer(booster)
            shap_values = np.asarray(explainer.shap_values(X_shap), dtype=np.float64)
            abs_shap = np.abs(shap_values)
            lag_counts = [1, 3, 6, 12, 24]
            lag_rows = []
            total = float(abs_shap.sum())
            for lag_count in lag_counts:
                share = float(abs_shap[:, -lag_count:].sum() / max(total, 1e-12))
                lag_rows.append({"last_k_lags": lag_count, "abs_shap_share": share})
            _save_rows_csv(lag_rows, output_root / "lightgbm_lag_concentration.csv")

    quality_map = _build_quality_map(config, task)
    distance_map = _build_distance_to_measurements_map(config)
    cnn_pred = np.load(_model_output_dir(config, "cnn_lstm_maskaware", task.interpolation_method) / "prediction_map.npy")
    residual_map = cnn_pred - task.target_map
    residual_abs = np.abs(residual_map)
    low_q, high_q = np.quantile(quality_map[task.test_mask], [0.33, 0.66])
    low_d, high_d = np.quantile(distance_map[task.test_mask], [0.33, 0.66])
    strata = {
        "quality_low_rmse": task.test_mask & (quality_map <= low_q),
        "quality_mid_rmse": task.test_mask & (quality_map > low_q) & (quality_map <= high_q),
        "quality_high_rmse": task.test_mask & (quality_map > high_q),
        "dist_low": task.test_mask & (distance_map <= low_d),
        "dist_mid": task.test_mask & (distance_map > low_d) & (distance_map <= high_d),
        "dist_high": task.test_mask & (distance_map > high_d),
    }
    strata_rows = []
    for name, mask in strata.items():
        if mask.sum() == 0:
            continue
        metric = masked_regression_metrics(task.target_map, cnn_pred, mask)
        strata_rows.append(
            {
                "stratum": name,
                "n_valid_cells": int(mask.sum()),
                "rmse": float(metric["rmse"]),
                "mae": float(metric["mae"]),
                "mean_abs_residual": float(residual_abs[mask].mean()),
            }
        )
    _save_rows_csv(strata_rows, output_root / "residual_strata.csv")

    tile_size = max(16, config.tile_size)
    tile_rows = []
    for row in range(0, residual_abs.shape[0], tile_size):
        for col in range(0, residual_abs.shape[1], tile_size):
            tile_mask = task.test_mask[row : row + tile_size, col : col + tile_size]
            if tile_mask.sum() == 0:
                continue
            tile_res = residual_abs[row : row + tile_size, col : col + tile_size][tile_mask]
            tile_rows.append((float(tile_res.mean()), row, col))
    tile_rows.sort(reverse=True)
    if tile_rows:
        _, row, col = tile_rows[0]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        slices = (slice(row, row + tile_size), slice(col, col + tile_size))
        panels = [
            ("Truth", task.target_map[slices]),
            ("Prediction", cnn_pred[slices]),
            ("Residual", residual_map[slices]),
            ("Mask", task.test_mask[slices].astype(float)),
        ]
        for ax, (title, panel) in zip(axes, panels):
            im = ax.imshow(panel, cmap="viridis")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(output_root / "failure_case_top_tile.png", dpi=180)
        plt.close()

    return output_root


def run_reproducibility_pack(base_config: RevisionConfig, experiment_roots: Iterable[Path]) -> Path:
    ensure_dir(CG_CONFIG_ROOT)
    ensure_dir(CG_SPLIT_ROOT)
    ensure_dir(CG_SCRIPTS_ROOT)
    ensure_dir(CG_OUTPUT_ROOT)

    environment_yml = PROJECT_ROOT / "environment.yml"
    if not environment_yml.exists():
        environment_yml.write_text(
            "\n".join(
                [
                    "name: found_training_project",
                    "channels:",
                    "  - conda-forge",
                    "dependencies:",
                    "  - python=3.11",
                    "  - pip",
                    "  - pip:",
                    "      - -r requirements-revision.txt",
                    "      - -r requirements-revision-optional.txt",
                ]
            ),
            encoding="utf-8",
        )

    readme_path = PROJECT_ROOT / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "\n".join(
                [
                    "# found_training_project",
                    "",
                    "Reproduction entry points:",
                    "- `python run_cg_additional_suite.py --phase all`",
                    "- `python run_cg_additional_suite.py --phase primary`",
                    "",
                    "Key outputs are written to `revision_outputs/cg_suite/`.",
                ]
            ),
            encoding="utf-8",
        )

    reproduce_ps1 = CG_SCRIPTS_ROOT / "reproduce_all.ps1"
    reproduce_ps1.write_text(
        "\n".join(
            [
                "& .\\.venv\\Scripts\\python.exe .\\run_cg_additional_suite.py --phase primary",
                "& .\\.venv\\Scripts\\python.exe .\\run_cg_additional_suite.py --phase mask",
                "& .\\.venv\\Scripts\\python.exe .\\run_cg_additional_suite.py --phase interpolation",
                "& .\\.venv\\Scripts\\python.exe .\\run_cg_additional_suite.py --phase split",
                "& .\\.venv\\Scripts\\python.exe .\\run_cg_additional_suite.py --phase scaling",
                "& .\\.venv\\Scripts\\python.exe .\\run_cg_additional_suite.py --phase interpretability",
                "& .\\.venv\\Scripts\\python.exe .\\run_cg_additional_suite.py --phase audit",
            ]
        ),
        encoding="utf-8",
    )

    reproduce_sh = CG_SCRIPTS_ROOT / "reproduce_all.sh"
    reproduce_sh.write_text(
        "\n".join(
            [
                "python run_cg_additional_suite.py --phase primary",
                "python run_cg_additional_suite.py --phase mask",
                "python run_cg_additional_suite.py --phase interpolation",
                "python run_cg_additional_suite.py --phase split",
                "python run_cg_additional_suite.py --phase scaling",
                "python run_cg_additional_suite.py --phase interpretability",
                "python run_cg_additional_suite.py --phase audit",
            ]
        ),
        encoding="utf-8",
    )

    _save_json(base_config.as_dict(), CG_CONFIG_ROOT / "base_revision_config.json")

    manifest_rows: List[Dict[str, object]] = []
    exported_split_count = 0
    exported_config_count = 0
    seen_split_hashes: set[str] = set()
    seen_config_hashes: set[str] = set()
    for root in experiment_roots:
        for artifact in root.rglob("*"):
            if artifact.is_file():
                manifest_rows.append(
                    {
                        "artifact_path": str(artifact),
                        "sha256": sha256_file(artifact),
                        "size_bytes": int(artifact.stat().st_size),
                    }
                )
                if artifact.name == "split_masks.npz":
                    split_hash = sha256_file(artifact)
                    if split_hash not in seen_split_hashes:
                        seen_split_hashes.add(split_hash)
                        bundle = np.load(artifact)
                        split_payload = {
                            "source_path": str(artifact),
                            "sha256": split_hash,
                            "eligible_indices": np.flatnonzero(bundle["eligible_mask"].astype(bool)).tolist(),
                            "train_indices": np.flatnonzero(bundle["train_mask"].astype(bool)).tolist(),
                            "val_indices": np.flatnonzero(bundle["val_mask"].astype(bool)).tolist(),
                            "test_indices": np.flatnonzero(bundle["test_mask"].astype(bool)).tolist(),
                        }
                        _save_json(split_payload, CG_SPLIT_ROOT / f"split_{split_hash}.json")
                        exported_split_count += 1
                if artifact.name == "config_snapshot.json":
                    config_hash = sha256_file(artifact)
                    if config_hash not in seen_config_hashes:
                        seen_config_hashes.add(config_hash)
                        payload = json.loads(artifact.read_text(encoding="utf-8"))
                        payload["source_path"] = str(artifact)
                        payload["sha256"] = config_hash
                        _save_json(payload, CG_CONFIG_ROOT / f"config_{config_hash}.json")
                        exported_config_count += 1
    _save_rows_csv(manifest_rows, CG_OUTPUT_ROOT / "manifest.csv")

    license_path = PROJECT_ROOT.parent / "LICENSE"
    checklist = [
        {"item": "README", "status": "present" if readme_path.exists() else "missing"},
        {"item": "environment.yml", "status": "present" if environment_yml.exists() else "missing"},
        {"item": "configs", "status": "present" if exported_config_count > 0 else "missing"},
        {"item": "splits", "status": "present" if exported_split_count > 0 else "missing"},
        {"item": "scripts/reproduce_all", "status": "present" if reproduce_ps1.exists() and reproduce_sh.exists() else "missing"},
        {"item": "outputs/manifest.csv", "status": "present" if (CG_OUTPUT_ROOT / "manifest.csv").exists() else "missing"},
        {"item": "LICENSE", "status": "present" if license_path.exists() else "missing"},
    ]
    _save_rows_csv(checklist, CG_OUTPUT_ROOT / "reproducibility_checklist.csv")
    return CG_OUTPUT_ROOT
