"""
Patch-based deep learning training for the revision-aligned displacement task.

Revision skeleton alignment:
- Section 3.2 / training samples must be defined in a leakage-aware way
- Section 3.3 / deep backends should be trained on comparable supervision volume
- Section 3.4 / normalized residual training protocol
- Section 3.6 / masked evaluation on the held-out domain

Why this file exists:
- The earlier deep models treated the whole spatio-temporal cube as a single
  training sample, while pointwise baselines received tens of thousands of
  pixel-level samples.
- That mismatch made the deep models collapse toward near-constant outputs.
- This module fixes the sample-construction problem by building many
  patch-level training samples while preserving the same split masks.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal

import numpy as np

from revision_experiments import _fit_torch_l1_regressor
from revision_config import RevisionConfig
from revision_utils import (
    DenseForecastTask,
    build_dense_forecast_task,
    build_tabular_dataset,
    ensure_dir,
    masked_regression_metrics,
    save_config_snapshot,
    save_error_diagnostics,
    save_map_comparison,
    save_metrics,
    save_prediction_map,
    save_split_bundle,
    set_random_seed,
    split_from_eligible_indices,
)


ModelKind = Literal[
    "cnn_lstm",
    "cnn_tcn",
    "cnn_lstm_hybrid",
    "cnn_tcn_hybrid",
    "conv_lstm_residual",
    "temporal_channel_cnn",
    "patch_unet_residual",
    "conv3d_residual",
    "temporal_linear_hybrid",
]
MaskMode = Literal["masked", "all"]


@dataclass
class PatchNormStats:
    input_mean: float
    input_std: float
    residual_mean: float
    residual_std: float


@dataclass
class PatchBatchArrays:
    inputs: np.ndarray
    residual_targets: np.ndarray
    absolute_targets: np.ndarray
    last_frames: np.ndarray
    masks: np.ndarray
    positions: np.ndarray


@dataclass
class LinearWarmStart:
    value_weights: np.ndarray
    bias: float


class Residual2DBlock:
    def __init__(self, channels: int, dilation: int = 1) -> None:
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                padding = dilation
                self.net = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                )
                self.act = nn.GELU()

            def forward(self, x):
                return self.act(self.net(x) + x)

        self.block_class = _Block

    def build(self):
        return self.block_class()


def _model_output_dir(config: RevisionConfig, model_name: str, interpolation_method: str) -> Path:
    return ensure_dir(config.output_dir(model_name, interpolation_method) / f"split_seed_{config.split_seed}")


def _write_history_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _package_metrics(
    model_name: str,
    task: DenseForecastTask,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": model_name,
        "interpolation_method": task.interpolation_method,
        "csv_path": str(task.csv_path),
        "grid_size": int(task.target_map.shape[0]),
        "history_length": int(task.input_maps.shape[0]),
        "eligible_pixels": int(task.eligible_mask.sum()),
        "train_pixels": int(task.train_mask.sum()),
        "val_pixels": int(task.val_mask.sum()),
        "test_pixels": int(task.test_mask.sum()),
    }
    payload.update(metrics)
    return payload


def _compute_patch_norm_stats(task: DenseForecastTask) -> PatchNormStats:
    train_values = task.input_maps[:, task.train_mask].astype(np.float32)
    residual_map = task.target_map - task.input_maps[-1]
    train_residuals = residual_map[task.train_mask].astype(np.float32)
    return PatchNormStats(
        input_mean=float(train_values.mean()),
        input_std=float(max(train_values.std(), 1e-6)),
        residual_mean=float(train_residuals.mean()),
        residual_std=float(max(train_residuals.std(), 1e-6)),
    )


def _fit_lasso_warm_start(
    task: DenseForecastTask,
    config: RevisionConfig,
    norm_stats: PatchNormStats,
    device_str: str,
) -> LinearWarmStart:
    X_all, y_all, eligible_indices = build_tabular_dataset(task)
    split_positions = split_from_eligible_indices(task, eligible_indices)

    X_train = X_all[split_positions["train"]]
    y_train = y_all[split_positions["train"]]
    X_val = X_all[split_positions["val"]]
    y_val = y_all[split_positions["val"]]

    best_state, _, _ = _fit_torch_l1_regressor(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        alpha=config.lasso_alpha,
        config=config,
        device_str=device_str,
    )

    weights_raw = (best_state["weights"].numpy().astype(np.float32) / best_state["X_std"].numpy().astype(np.float32).reshape(-1)) * float(best_state["y_std"])
    bias_raw = float(best_state["bias"]) * float(best_state["y_std"]) + float(best_state["y_mean"])
    bias_raw = bias_raw - float(np.sum(weights_raw * best_state["X_mean"].numpy().astype(np.float32).reshape(-1)))

    residual_weights_raw = weights_raw.copy()
    residual_weights_raw[-1] -= 1.0

    input_mean = norm_stats.input_mean
    input_std = norm_stats.input_std
    residual_mean = norm_stats.residual_mean
    residual_std = norm_stats.residual_std
    normalized_value_weights = residual_weights_raw * (input_std / residual_std)
    normalized_bias = (float(np.sum(residual_weights_raw * input_mean)) + bias_raw - residual_mean) / residual_std

    return LinearWarmStart(
        value_weights=normalized_value_weights.astype(np.float32),
        bias=float(normalized_bias),
    )


def _iter_patch_positions(height: int, width: int, patch_size: int, stride: int) -> Iterable[tuple[int, int]]:
    row_positions = list(range(0, max(height - patch_size, 0) + 1, stride))
    col_positions = list(range(0, max(width - patch_size, 0) + 1, stride))
    if row_positions[-1] != height - patch_size:
        row_positions.append(height - patch_size)
    if col_positions[-1] != width - patch_size:
        col_positions.append(width - patch_size)
    for row in row_positions:
        for col in col_positions:
            yield int(row), int(col)


def _build_patch_arrays(
    task: DenseForecastTask,
    config: RevisionConfig,
    norm_stats: PatchNormStats,
    *,
    supervision_mask: np.ndarray | None,
    use_input_mask: bool,
) -> PatchBatchArrays:
    patch_size = int(config.patch_size)
    stride = int(config.patch_stride)
    height, width = task.target_map.shape

    residual_map = task.target_map - task.input_maps[-1]
    input_maps_norm = (task.input_maps - norm_stats.input_mean) / norm_stats.input_std
    residual_map_norm = (residual_map - norm_stats.residual_mean) / norm_stats.residual_std

    inputs: List[np.ndarray] = []
    residual_targets: List[np.ndarray] = []
    absolute_targets: List[np.ndarray] = []
    last_frames: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    positions: List[np.ndarray] = []

    for row, col in _iter_patch_positions(height, width, patch_size, stride):
        row_slice = slice(row, row + patch_size)
        col_slice = slice(col, col + patch_size)
        if supervision_mask is None:
            patch_mask = np.ones((patch_size, patch_size), dtype=np.float32)
        else:
            patch_mask = supervision_mask[row_slice, col_slice].astype(np.float32)
            if int(patch_mask.sum()) < int(config.patch_min_valid_pixels):
                continue

        value_patch = input_maps_norm[:, row_slice, col_slice].astype(np.float32)
        if use_input_mask:
            mask_patch = task.input_valid_mask[:, row_slice, col_slice].astype(np.float32)
            input_patch = np.stack([value_patch, mask_patch], axis=1)
        else:
            input_patch = value_patch[:, np.newaxis, :, :]

        inputs.append(input_patch.astype(np.float32))
        residual_targets.append(residual_map_norm[row_slice, col_slice][np.newaxis, :, :].astype(np.float32))
        absolute_targets.append(task.target_map[row_slice, col_slice][np.newaxis, :, :].astype(np.float32))
        last_frames.append(task.input_maps[-1, row_slice, col_slice][np.newaxis, :, :].astype(np.float32))
        masks.append(patch_mask[np.newaxis, :, :].astype(np.float32))
        positions.append(np.asarray([row, col], dtype=np.int32))

    if not inputs:
        raise ValueError("No valid patches were created. Try reducing patch_min_valid_pixels or patch_size.")

    return PatchBatchArrays(
        inputs=np.stack(inputs, axis=0),
        residual_targets=np.stack(residual_targets, axis=0),
        absolute_targets=np.stack(absolute_targets, axis=0),
        last_frames=np.stack(last_frames, axis=0),
        masks=np.stack(masks, axis=0),
        positions=np.stack(positions, axis=0),
    )


def _masked_smooth_l1(pred, target, mask):
    import torch
    import torch.nn.functional as F

    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class PatchCNNLSTMModel:
    def __init__(self, input_channels: int, patch_size: int, hidden_dim: int) -> None:
        import torch.nn as nn

        if patch_size % 8 != 0:
            raise ValueError("patch_size must be divisible by 8 for the CNN-LSTM encoder.")

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                )
                feature_dim = 128 * (patch_size // 8) * (patch_size // 8)
                self.feature_proj = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
                self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, patch_size * patch_size),
                )
                self.patch_size = patch_size

            def forward(self, x):
                batch_size, time_steps, channels, height, width = x.shape
                x = x.view(batch_size * time_steps, channels, height, width)
                features = self.encoder(x).reshape(batch_size * time_steps, -1)
                features = self.feature_proj(features)
                features = features.view(batch_size, time_steps, -1)
                lstm_out, _ = self.lstm(features)
                final_feature = lstm_out[:, -1, :]
                residual = self.decoder(final_feature)
                return residual.view(batch_size, 1, self.patch_size, self.patch_size)

        self.model_class = _Model

    def build(self):
        return self.model_class()


class TemporalResidualBlock:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                padding = dilation * (kernel_size - 1)
                self.net = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                    nn.GELU(),
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


class PatchCNNTCNModel:
    def __init__(self, input_channels: int, patch_size: int, hidden_dim: int, config: RevisionConfig) -> None:
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                layers = []
                in_ch = 128
                for layer_idx in range(config.tcn_num_layers):
                    layers.append(
                        TemporalResidualBlock(
                            in_channels=in_ch,
                            out_channels=hidden_dim,
                            kernel_size=config.tcn_kernel_size,
                            dilation=2**layer_idx,
                            dropout=config.tcn_dropout,
                        ).build()
                    )
                    in_ch = hidden_dim
                self.tcn = nn.Sequential(*layers)
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, patch_size * patch_size),
                )
                self.patch_size = patch_size

            def forward(self, x):
                batch_size, time_steps, channels, height, width = x.shape
                x = x.view(batch_size * time_steps, channels, height, width)
                features = self.encoder(x).view(batch_size, time_steps, 128)
                features = features.transpose(1, 2)
                temporal = self.tcn(features)
                final_feature = temporal[:, :, -1]
                residual = self.decoder(final_feature)
                return residual.view(batch_size, 1, self.patch_size, self.patch_size)

        self.model_class = _Model

    def build(self):
        return self.model_class()


class PatchCNNLSTMHybridModel:
    def __init__(
        self,
        input_channels: int,
        total_input_channels: int,
        time_steps: int,
        config: RevisionConfig,
        warm_start: LinearWarmStart | None,
    ) -> None:
        import torch
        import torch.nn as nn

        hidden_channels = int(config.nontransformer_hybrid_hidden_channels)
        decoder_channels = max(hidden_channels // 2, 32)

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.time_steps = time_steps
                self.input_channels = input_channels
                self.hidden_channels = hidden_channels
                self.recent_lags = max(2, min(config.temporal_hybrid_recent_lags, max(time_steps - 1, 2)))
                self.linear_head = nn.Conv2d(total_input_channels, 1, kernel_size=1, bias=True)

                self.frame_encoder = nn.Sequential(
                    nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                )
                self.cells = nn.ModuleList(
                    [ConvLSTMCell(hidden_channels, hidden_channels, 3).build() for _ in range(config.convlstm_num_layers)]
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, decoder_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=2, stride=2),
                    nn.GELU(),
                )
                self.correction_head = nn.Sequential(
                    nn.Conv2d(decoder_channels + 2, decoder_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    Residual2DBlock(decoder_channels, dilation=1).build(),
                    nn.Conv2d(decoder_channels, 1, kernel_size=1),
                )
                self.recent_gate = nn.Sequential(
                    nn.Conv2d(self.recent_lags * 2, 64, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(64, self.recent_lags, kernel_size=3, padding=1),
                )
                self.recent_scale = nn.Parameter(
                    torch.tensor(float(config.temporal_hybrid_recent_scale_init), dtype=torch.float32)
                )
                self.correction_scale = nn.Parameter(
                    torch.tensor(float(config.temporal_hybrid_correction_scale_init), dtype=torch.float32)
                )
                self._initialize_linear_head()
                self._initialize_recent_gate()

            def _initialize_linear_head(self) -> None:
                nn.init.zeros_(self.linear_head.weight)
                nn.init.zeros_(self.linear_head.bias)
                if warm_start is None:
                    return
                with torch.no_grad():
                    value_weights = torch.tensor(warm_start.value_weights, dtype=torch.float32).view(1, self.time_steps, 1, 1)
                    self.linear_head.weight.zero_()
                    self.linear_head.weight[:, : self.time_steps] = value_weights
                    self.linear_head.bias[:] = float(warm_start.bias)

            def _initialize_recent_gate(self) -> None:
                first_conv = self.recent_gate[0]
                second_conv = self.recent_gate[2]
                nn.init.zeros_(first_conv.weight)
                nn.init.zeros_(first_conv.bias)
                nn.init.zeros_(second_conv.weight)
                with torch.no_grad():
                    recency_bias = torch.linspace(-0.6, 0.6, steps=self.recent_lags, dtype=torch.float32)
                    second_conv.bias.copy_(recency_bias)

            def forward(self, x):
                import torch

                batch_size, time_steps_, channels, height, width = x.shape
                x_flat = x.transpose(1, 2).reshape(batch_size, time_steps_ * channels, height, width)
                base = self.linear_head(x_flat)
                value_channels = x_flat[:, : self.time_steps]
                if self.input_channels > 1:
                    mask_channels = x_flat[:, self.time_steps : self.time_steps * 2]
                else:
                    mask_channels = torch.ones_like(value_channels)

                recent_values = value_channels[:, -(self.recent_lags + 1) : -1]
                last_value = value_channels[:, -1:].expand_as(recent_values)
                recent_deltas = recent_values - last_value
                recent_masks = mask_channels[:, -(self.recent_lags + 1) : -1]
                gate_logits = self.recent_gate(torch.cat([recent_deltas, recent_masks], dim=1))
                gate = torch.softmax(gate_logits, dim=1)
                recent_mix = (gate * recent_deltas).sum(dim=1, keepdim=True)

                encoded = self.frame_encoder(x.reshape(batch_size * time_steps_, channels, height, width))
                enc_h, enc_w = encoded.shape[-2:]
                encoded = encoded.view(batch_size, time_steps_, self.hidden_channels, enc_h, enc_w)
                states = [
                    (
                        torch.zeros(batch_size, self.hidden_channels, enc_h, enc_w, device=x.device),
                        torch.zeros(batch_size, self.hidden_channels, enc_h, enc_w, device=x.device),
                    )
                    for _ in self.cells
                ]

                for time_idx in range(time_steps_):
                    current = encoded[:, time_idx]
                    for layer_idx, cell in enumerate(self.cells):
                        h_next, c_next = cell(current, states[layer_idx])
                        states[layer_idx] = (h_next, c_next)
                        current = h_next

                decoded = self.decoder(states[-1][0])
                correction = self.correction_head(torch.cat([decoded, base, recent_mix], dim=1))
                return base + self.recent_scale * recent_mix + self.correction_scale * correction

        self.model_class = _Model

    def build(self):
        return self.model_class()


class PatchCNNTCNHybridModel:
    def __init__(
        self,
        input_channels: int,
        total_input_channels: int,
        time_steps: int,
        config: RevisionConfig,
        warm_start: LinearWarmStart | None,
    ) -> None:
        import torch
        import torch.nn as nn

        hidden_channels = int(config.nontransformer_hybrid_hidden_channels)
        decoder_channels = max(hidden_channels // 2, 32)

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.time_steps = time_steps
                self.input_channels = input_channels
                self.hidden_channels = hidden_channels
                self.recent_lags = max(2, min(config.temporal_hybrid_recent_lags, max(time_steps - 1, 2)))
                self.linear_head = nn.Conv2d(total_input_channels, 1, kernel_size=1, bias=True)

                self.frame_encoder = nn.Sequential(
                    nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                )
                layers = []
                in_ch = hidden_channels
                for layer_idx in range(config.tcn_num_layers):
                    layers.append(
                        TemporalResidualBlock(
                            in_channels=in_ch,
                            out_channels=hidden_channels,
                            kernel_size=config.tcn_kernel_size,
                            dilation=2**layer_idx,
                            dropout=config.tcn_dropout,
                        ).build()
                    )
                    in_ch = hidden_channels
                self.temporal_tcn = nn.Sequential(*layers)
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, decoder_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=2, stride=2),
                    nn.GELU(),
                )
                self.correction_head = nn.Sequential(
                    nn.Conv2d(decoder_channels + 2, decoder_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    Residual2DBlock(decoder_channels, dilation=1).build(),
                    nn.Conv2d(decoder_channels, 1, kernel_size=1),
                )
                self.recent_gate = nn.Sequential(
                    nn.Conv2d(self.recent_lags * 2, 64, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(64, self.recent_lags, kernel_size=3, padding=1),
                )
                self.recent_scale = nn.Parameter(
                    torch.tensor(float(config.temporal_hybrid_recent_scale_init), dtype=torch.float32)
                )
                self.correction_scale = nn.Parameter(
                    torch.tensor(float(config.temporal_hybrid_correction_scale_init), dtype=torch.float32)
                )
                self._initialize_linear_head()
                self._initialize_recent_gate()

            def _initialize_linear_head(self) -> None:
                nn.init.zeros_(self.linear_head.weight)
                nn.init.zeros_(self.linear_head.bias)
                if warm_start is None:
                    return
                with torch.no_grad():
                    value_weights = torch.tensor(warm_start.value_weights, dtype=torch.float32).view(1, self.time_steps, 1, 1)
                    self.linear_head.weight.zero_()
                    self.linear_head.weight[:, : self.time_steps] = value_weights
                    self.linear_head.bias[:] = float(warm_start.bias)

            def _initialize_recent_gate(self) -> None:
                first_conv = self.recent_gate[0]
                second_conv = self.recent_gate[2]
                nn.init.zeros_(first_conv.weight)
                nn.init.zeros_(first_conv.bias)
                nn.init.zeros_(second_conv.weight)
                with torch.no_grad():
                    recency_bias = torch.linspace(-0.6, 0.6, steps=self.recent_lags, dtype=torch.float32)
                    second_conv.bias.copy_(recency_bias)

            def forward(self, x):
                import torch

                batch_size, time_steps_, channels, height, width = x.shape
                x_flat = x.transpose(1, 2).reshape(batch_size, time_steps_ * channels, height, width)
                base = self.linear_head(x_flat)
                value_channels = x_flat[:, : self.time_steps]
                if self.input_channels > 1:
                    mask_channels = x_flat[:, self.time_steps : self.time_steps * 2]
                else:
                    mask_channels = torch.ones_like(value_channels)

                recent_values = value_channels[:, -(self.recent_lags + 1) : -1]
                last_value = value_channels[:, -1:].expand_as(recent_values)
                recent_deltas = recent_values - last_value
                recent_masks = mask_channels[:, -(self.recent_lags + 1) : -1]
                gate_logits = self.recent_gate(torch.cat([recent_deltas, recent_masks], dim=1))
                gate = torch.softmax(gate_logits, dim=1)
                recent_mix = (gate * recent_deltas).sum(dim=1, keepdim=True)

                encoded = self.frame_encoder(x.reshape(batch_size * time_steps_, channels, height, width))
                enc_h, enc_w = encoded.shape[-2:]
                encoded = encoded.view(batch_size, time_steps_, self.hidden_channels, enc_h, enc_w)
                temporal_input = encoded.permute(0, 3, 4, 2, 1).reshape(batch_size * enc_h * enc_w, self.hidden_channels, time_steps_)
                temporal_output = self.temporal_tcn(temporal_input)
                final_feature = temporal_output[:, :, -1].view(batch_size, enc_h, enc_w, self.hidden_channels).permute(0, 3, 1, 2)

                decoded = self.decoder(final_feature)
                correction = self.correction_head(torch.cat([decoded, base, recent_mix], dim=1))
                return base + self.recent_scale * recent_mix + self.correction_scale * correction

        self.model_class = _Model

    def build(self):
        return self.model_class()


class ConvLSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int) -> None:
        import torch.nn as nn

        class _Cell(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                padding = kernel_size // 2
                self.hidden_dim = hidden_dim
                self.conv = nn.Conv2d(
                    in_channels=input_dim + hidden_dim,
                    out_channels=4 * hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=True,
                )

            def forward(self, x, state):
                import torch

                h_cur, c_cur = state
                combined = torch.cat([x, h_cur], dim=1)
                gates = self.conv(combined)
                i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                o = torch.sigmoid(o)
                g = torch.tanh(g)
                c_next = f * c_cur + i * g
                h_next = o * torch.tanh(c_next)
                return h_next, c_next

        self.cell_class = _Cell
        self.hidden_dim = hidden_dim

    def build(self):
        return self.cell_class()

    def init_hidden(self, batch_size: int, image_size: tuple[int, int], device):
        import torch

        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class PatchConvLSTMResidualModel:
    def __init__(self, input_channels: int, hidden_dim: int, num_layers: int, kernel_size: int) -> None:
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.input_proj = nn.Sequential(
                    nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.cells = nn.ModuleList(
                    [ConvLSTMCell(hidden_dim, hidden_dim, kernel_size).build() for _ in range(num_layers)]
                )
                self.refine = nn.Sequential(
                    Residual2DBlock(hidden_dim, dilation=1).build(),
                    Residual2DBlock(hidden_dim, dilation=2).build(),
                    Residual2DBlock(hidden_dim, dilation=1).build(),
                )
                self.head = nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, 1, kernel_size=1),
                )

            def forward(self, x):
                batch_size, time_steps, _, height, width = x.shape
                device = x.device
                states = [
                    ConvLSTMCell(hidden_dim, hidden_dim, kernel_size).init_hidden(batch_size, (height, width), device)
                    for _ in range(self.num_layers)
                ]

                for t in range(time_steps):
                    current = self.input_proj(x[:, t])
                    for layer_idx, cell in enumerate(self.cells):
                        h_next, c_next = cell(current, states[layer_idx])
                        states[layer_idx] = (h_next, c_next)
                        current = h_next

                features = self.refine(states[-1][0])
                return self.head(features)

        self.model_class = _Model

    def build(self):
        return self.model_class()


class PatchTemporalChannelCNNModel:
    def __init__(self, total_input_channels: int) -> None:
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(total_input_channels, 128, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(128, 96, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.body = nn.Sequential(
                    Residual2DBlock(96, dilation=1).build(),
                    Residual2DBlock(96, dilation=2).build(),
                    Residual2DBlock(96, dilation=4).build(),
                    Residual2DBlock(96, dilation=1).build(),
                )
                self.head = nn.Sequential(
                    nn.Conv2d(96, 64, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(64, 1, kernel_size=1),
                )

            def forward(self, x):
                features = self.stem(x)
                features = self.body(features)
                return self.head(features)

        self.model_class = _Model

    def build(self):
        return self.model_class()


class PatchUNetResidualModel:
    def __init__(self, total_input_channels: int) -> None:
        import torch
        import torch.nn as nn

        def conv_block(in_ch: int, out_ch: int):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GELU(),
            )

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Conv2d(total_input_channels, 96, kernel_size=1),
                    nn.GELU(),
                )
                self.enc1 = conv_block(96, 96)
                self.pool1 = nn.MaxPool2d(2)
                self.enc2 = conv_block(96, 160)
                self.pool2 = nn.MaxPool2d(2)
                self.bottleneck = conv_block(160, 256)
                self.up1 = nn.ConvTranspose2d(256, 160, kernel_size=2, stride=2)
                self.dec1 = conv_block(320, 160)
                self.up2 = nn.ConvTranspose2d(160, 96, kernel_size=2, stride=2)
                self.dec2 = conv_block(192, 96)
                self.head = nn.Conv2d(96, 1, kernel_size=1)

            def forward(self, x):
                x0 = self.input_proj(x)
                e1 = self.enc1(x0)
                e2 = self.enc2(self.pool1(e1))
                b = self.bottleneck(self.pool2(e2))
                d1 = self.up1(b)
                d1 = self.dec1(torch.cat([d1, e2], dim=1))
                d2 = self.up2(d1)
                d2 = self.dec2(torch.cat([d2, e1], dim=1))
                return self.head(d2)

        self.model_class = _Model

    def build(self):
        return self.model_class()


class PatchConv3DResidualModel:
    def __init__(self, input_channels: int) -> None:
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temporal_encoder = nn.Sequential(
                    nn.Conv3d(input_channels, 16, kernel_size=(7, 3, 3), stride=(3, 1, 1), padding=(3, 1, 1)),
                    nn.GELU(),
                    nn.Conv3d(16, 32, kernel_size=(7, 3, 3), stride=(3, 1, 1), padding=(3, 1, 1)),
                    nn.GELU(),
                    nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(3, 1, 1), padding=(2, 1, 1)),
                    nn.GELU(),
                    nn.Conv3d(64, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
                    nn.GELU(),
                )
                self.spatial_refine = nn.Sequential(
                    Residual2DBlock(64, dilation=1).build(),
                    Residual2DBlock(64, dilation=2).build(),
                    Residual2DBlock(64, dilation=1).build(),
                )
                self.head = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(64, 1, kernel_size=1),
                )

            def forward(self, x):
                features3d = self.temporal_encoder(x)
                features2d = features3d.mean(dim=2)
                features2d = self.spatial_refine(features2d)
                return self.head(features2d)

        self.model_class = _Model

    def build(self):
        return self.model_class()


class PatchTemporalLinearHybridModel:
    def __init__(
        self,
        total_input_channels: int,
        time_steps: int,
        input_channels: int,
        config: RevisionConfig,
        warm_start: LinearWarmStart | None,
    ) -> None:
        import torch
        import torch.nn as nn

        def conv_block(in_ch: int, out_ch: int):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GELU(),
            )

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear_head = nn.Conv2d(total_input_channels, 1, kernel_size=1, bias=True)
                self.time_steps = time_steps
                self.input_channels = input_channels
                self.recent_lags = max(2, min(config.temporal_hybrid_recent_lags, max(time_steps - 1, 2)))

                self.feature_proj = nn.Sequential(
                    nn.Conv2d(total_input_channels + 2, 96, kernel_size=1),
                    nn.GELU(),
                )
                self.enc1 = conv_block(96, 96)
                self.pool1 = nn.MaxPool2d(2)
                self.enc2 = conv_block(96, 160)
                self.pool2 = nn.MaxPool2d(2)
                self.bottleneck = conv_block(160, 224)
                self.up1 = nn.ConvTranspose2d(224, 160, kernel_size=2, stride=2)
                self.dec1 = conv_block(320, 160)
                self.up2 = nn.ConvTranspose2d(160, 96, kernel_size=2, stride=2)
                self.dec2 = conv_block(192, 96)
                self.correction_head = nn.Conv2d(96, 1, kernel_size=1)
                self.recent_gate = nn.Sequential(
                    nn.Conv2d(self.recent_lags * 2, 64, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(64, self.recent_lags, kernel_size=3, padding=1),
                )
                self.recent_scale = nn.Parameter(
                    torch.tensor(float(config.temporal_hybrid_recent_scale_init), dtype=torch.float32)
                )
                self.correction_scale = nn.Parameter(
                    torch.tensor(float(config.temporal_hybrid_correction_scale_init), dtype=torch.float32)
                )
                self._initialize_linear_head()
                self._initialize_recent_gate()

            def _initialize_linear_head(self) -> None:
                nn.init.zeros_(self.linear_head.weight)
                nn.init.zeros_(self.linear_head.bias)
                if warm_start is None:
                    return
                with torch.no_grad():
                    self.linear_head.weight.zero_()
                    value_weights = torch.tensor(warm_start.value_weights, dtype=torch.float32).view(1, self.time_steps, 1, 1)
                    self.linear_head.weight[:, : self.time_steps] = value_weights
                    self.linear_head.bias[:] = float(warm_start.bias)

            def _initialize_recent_gate(self) -> None:
                first_conv = self.recent_gate[0]
                second_conv = self.recent_gate[2]
                nn.init.zeros_(first_conv.weight)
                nn.init.zeros_(first_conv.bias)
                nn.init.zeros_(second_conv.weight)
                with torch.no_grad():
                    recency_bias = torch.linspace(-0.6, 0.6, steps=self.recent_lags, dtype=torch.float32)
                    second_conv.bias.copy_(recency_bias)

            def forward(self, x):
                base = self.linear_head(x)
                value_channels = x[:, : self.time_steps]
                if self.input_channels > 1:
                    mask_channels = x[:, self.time_steps : self.time_steps * 2]
                else:
                    mask_channels = torch.ones_like(value_channels)

                recent_values = value_channels[:, -(self.recent_lags + 1) : -1]
                last_value = value_channels[:, -1:].expand_as(recent_values)
                recent_deltas = recent_values - last_value
                recent_masks = mask_channels[:, -(self.recent_lags + 1) : -1]
                gate_logits = self.recent_gate(torch.cat([recent_deltas, recent_masks], dim=1))
                gate = torch.softmax(gate_logits, dim=1)
                recent_mix = (gate * recent_deltas).sum(dim=1, keepdim=True)

                hybrid_input = torch.cat([x, base, recent_mix], dim=1)
                x0 = self.feature_proj(hybrid_input)
                e1 = self.enc1(x0)
                e2 = self.enc2(self.pool1(e1))
                b = self.bottleneck(self.pool2(e2))
                d1 = self.up1(b)
                d1 = self.dec1(torch.cat([d1, e2], dim=1))
                d2 = self.up2(d1)
                d2 = self.dec2(torch.cat([d2, e1], dim=1))
                correction = self.correction_head(d2)
                return base + self.recent_scale * recent_mix + self.correction_scale * correction

        self.model_class = _Model

    def build(self):
        return self.model_class()


def _select_model_builder(
    model_kind: ModelKind,
    time_steps: int,
    input_channels: int,
    patch_size: int,
    config: RevisionConfig,
    warm_start: LinearWarmStart | None = None,
):
    if model_kind == "cnn_lstm":
        return PatchCNNLSTMModel(
            input_channels=input_channels,
            patch_size=patch_size,
            hidden_dim=config.cnn_hidden_dim,
        )
    if model_kind == "cnn_tcn":
        return PatchCNNTCNModel(
            input_channels=input_channels,
            patch_size=patch_size,
            hidden_dim=config.tcn_hidden_channels,
            config=config,
        )
    if model_kind == "cnn_lstm_hybrid":
        return PatchCNNLSTMHybridModel(
            input_channels=input_channels,
            total_input_channels=time_steps * input_channels,
            time_steps=time_steps,
            config=config,
            warm_start=warm_start,
        )
    if model_kind == "cnn_tcn_hybrid":
        return PatchCNNTCNHybridModel(
            input_channels=input_channels,
            total_input_channels=time_steps * input_channels,
            time_steps=time_steps,
            config=config,
            warm_start=warm_start,
        )
    if model_kind == "conv_lstm_residual":
        return PatchConvLSTMResidualModel(
            input_channels=input_channels,
            hidden_dim=config.convlstm_hidden_dim,
            num_layers=config.convlstm_num_layers,
            kernel_size=config.convlstm_kernel_size,
        )
    if model_kind == "temporal_channel_cnn":
        return PatchTemporalChannelCNNModel(total_input_channels=time_steps * input_channels)
    if model_kind == "patch_unet_residual":
        return PatchUNetResidualModel(total_input_channels=time_steps * input_channels)
    if model_kind == "conv3d_residual":
        return PatchConv3DResidualModel(input_channels=input_channels)
    if model_kind == "temporal_linear_hybrid":
        return PatchTemporalLinearHybridModel(
            total_input_channels=time_steps * input_channels,
            time_steps=time_steps,
            input_channels=input_channels,
            config=config,
            warm_start=warm_start,
        )
    raise ValueError(f"Unsupported model kind: {model_kind}")


def _adapt_patch_input_layout(patch_inputs: np.ndarray, model_kind: ModelKind) -> np.ndarray:
    if model_kind in {"cnn_lstm", "cnn_tcn", "cnn_lstm_hybrid", "cnn_tcn_hybrid", "conv_lstm_residual"}:
        return patch_inputs.astype(np.float32)
    if model_kind in {"temporal_channel_cnn", "patch_unet_residual", "temporal_linear_hybrid"}:
        n_items, time_steps, channels, height, width = patch_inputs.shape
        return (
            patch_inputs.transpose(0, 2, 1, 3, 4)
            .reshape(n_items, time_steps * channels, height, width)
            .astype(np.float32)
        )
    if model_kind == "conv3d_residual":
        return patch_inputs.transpose(0, 2, 1, 3, 4).astype(np.float32)
    raise ValueError(f"Unsupported model kind: {model_kind}")


def _run_loader_epoch(model, loader, optimizer, device, *, loss_mask_mode: MaskMode) -> float:
    import torch

    is_train = optimizer is not None
    total_loss = 0.0
    total_weight = 0.0
    if is_train:
        model.train()
    else:
        model.eval()

    for batch in loader:
        inputs, residual_targets, masks = [tensor.to(device) for tensor in batch]
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred_residual = model(inputs)
            if loss_mask_mode == "masked":
                loss = _masked_smooth_l1(pred_residual, residual_targets, masks)
            else:
                ones_mask = torch.ones_like(masks)
                loss = _masked_smooth_l1(pred_residual, residual_targets, ones_mask)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_weight = float(masks.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight

    return total_loss / max(total_weight, 1.0)


def _predict_full_map(
    model,
    task: DenseForecastTask,
    config: RevisionConfig,
    norm_stats: PatchNormStats,
    *,
    use_input_mask: bool,
    model_kind: ModelKind,
    device,
) -> np.ndarray:
    import torch

    inference_patches = _build_patch_arrays(
        task=task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=None,
        use_input_mask=use_input_mask,
    )
    inputs = torch.tensor(_adapt_patch_input_layout(inference_patches.inputs, model_kind), dtype=torch.float32, device=device)
    last_frames = inference_patches.last_frames[:, 0]
    positions = inference_patches.positions
    patch_size = int(config.patch_size)
    batch_size = int(config.patch_batch_size)

    pred_sum = np.zeros_like(task.target_map, dtype=np.float32)
    weight_sum = np.zeros_like(task.target_map, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            end = min(start + batch_size, len(inputs))
            pred_residual_norm = model(inputs[start:end]).detach().cpu().numpy().astype(np.float32)
            pred_residual = pred_residual_norm[:, 0] * norm_stats.residual_std + norm_stats.residual_mean
            pred_absolute = pred_residual + last_frames[start:end]
            for local_idx, patch in enumerate(pred_absolute):
                row, col = positions[start + local_idx]
                row_slice = slice(int(row), int(row) + patch_size)
                col_slice = slice(int(col), int(col) + patch_size)
                pred_sum[row_slice, col_slice] += patch
                weight_sum[row_slice, col_slice] += 1.0

    pred_map = pred_sum / np.clip(weight_sum, 1.0, None)
    return pred_map.astype(np.float32)


def run_patch_deep_model(
    config: RevisionConfig,
    model_name: str,
    model_kind: ModelKind,
    *,
    interpolation_method: str | None = None,
    use_input_mask: bool = True,
    loss_mask_mode: MaskMode = "masked",
    metric_mask_mode: MaskMode = "masked",
) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    set_random_seed(config.split_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    task = build_dense_forecast_task(config, interpolation_method)
    output_dir = _model_output_dir(config, model_name, task.interpolation_method)
    save_config_snapshot(config, output_dir)
    save_split_bundle(task, output_dir)

    norm_stats = _compute_patch_norm_stats(task)
    train_patches = _build_patch_arrays(
        task=task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=task.train_mask if loss_mask_mode == "masked" else task.target_valid_mask,
        use_input_mask=use_input_mask,
    )
    val_patches = _build_patch_arrays(
        task=task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=task.val_mask,
        use_input_mask=use_input_mask,
    )
    train_inputs = _adapt_patch_input_layout(train_patches.inputs, model_kind)
    val_inputs = _adapt_patch_input_layout(val_patches.inputs, model_kind)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    warm_start: LinearWarmStart | None = None
    if model_kind in {"temporal_linear_hybrid", "cnn_lstm_hybrid", "cnn_tcn_hybrid"}:
        warm_start = _fit_lasso_warm_start(
            task=task,
            config=config,
            norm_stats=norm_stats,
            device_str=str(device),
        )

    builder = _select_model_builder(
        model_kind=model_kind,
        time_steps=train_patches.inputs.shape[1],
        input_channels=train_patches.inputs.shape[2],
        patch_size=config.patch_size,
        config=config,
        warm_start=warm_start,
    )
    model = builder.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.cnn_learning_rate, weight_decay=config.cnn_weight_decay)

    train_dataset = TensorDataset(
        torch.tensor(train_inputs, dtype=torch.float32),
        torch.tensor(train_patches.residual_targets, dtype=torch.float32),
        torch.tensor(train_patches.masks, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_inputs, dtype=torch.float32),
        torch.tensor(val_patches.residual_targets, dtype=torch.float32),
        torch.tensor(val_patches.masks, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.patch_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.patch_batch_size, shuffle=False, num_workers=0)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    history_rows: List[Dict[str, float]] = []
    patience_counter = 0
    started = time.perf_counter()

    for epoch in range(1, config.cnn_epochs + 1):
        train_loss = _run_loader_epoch(model, train_loader, optimizer, device, loss_mask_mode=loss_mask_mode)
        val_loss = _run_loader_epoch(model, val_loader, None, device, loss_mask_mode="masked")

        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

        if val_loss < best_val - 1e-9:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.cnn_patience:
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} patch training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / f"{model_name}_best_model.pth")
    _write_history_csv(history_rows, output_dir / "training_history.csv")

    pred_map = _predict_full_map(
        model=model,
        task=task,
        config=config,
        norm_stats=norm_stats,
        use_input_mask=use_input_mask,
        model_kind=model_kind,
        device=device,
    )

    runtime_seconds = time.perf_counter() - started
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    eval_mask = task.test_mask if metric_mask_mode == "masked" else np.ones_like(task.test_mask, dtype=bool)
    metrics = masked_regression_metrics(task.target_map, pred_map, eval_mask)
    metrics.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "device": str(device),
            "runtime_seconds": float(runtime_seconds),
            "peak_gpu_memory_mb": float(peak_memory_mb),
            "input_mask_channel": bool(use_input_mask),
            "loss_mask_mode": loss_mask_mode,
            "metric_mask_mode": metric_mask_mode,
            "patch_size": int(config.patch_size),
            "patch_stride": int(config.patch_stride),
            "patch_min_valid_pixels": int(config.patch_min_valid_pixels),
            "train_patch_count": int(train_patches.inputs.shape[0]),
            "val_patch_count": int(val_patches.inputs.shape[0]),
            "normalization_input_mean": float(norm_stats.input_mean),
            "normalization_input_std": float(norm_stats.input_std),
            "normalization_residual_mean": float(norm_stats.residual_mean),
            "normalization_residual_std": float(norm_stats.residual_std),
            "warm_start_enabled": bool(warm_start is not None),
            "temporal_hybrid_recent_lags": int(config.temporal_hybrid_recent_lags)
            if model_kind in {"temporal_linear_hybrid", "cnn_lstm_hybrid", "cnn_tcn_hybrid"}
            else None,
        }
    )

    payload = _package_metrics(model_name, task, metrics)
    save_metrics(payload, output_dir)
    save_prediction_map(pred_map, output_dir)
    np.save(output_dir / "target_map.npy", task.target_map.astype(np.float32))
    np.save(output_dir / "last_input_map.npy", task.input_maps[-1].astype(np.float32))
    save_error_diagnostics(task.target_map, pred_map, task.test_mask, output_dir, n_bins=config.diag_bins)
    save_map_comparison(task.target_map, pred_map, task.target_valid_mask, output_dir)

    with (output_dir / "patch_norm_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "input_mean": norm_stats.input_mean,
                "input_std": norm_stats.input_std,
                "residual_mean": norm_stats.residual_mean,
                "residual_std": norm_stats.residual_std,
            },
            fh,
            indent=2,
        )
    return payload
