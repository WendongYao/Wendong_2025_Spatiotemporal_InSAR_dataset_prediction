"""Native-support pointwise models for the locked CAGEO v2.3 experiments.

This module deliberately contains no raster, Transformer, STGCN, or RSASE
dependency.  SPAR and the causal TCN receive one EGMS Level-3 product-cell
history at a time and are evaluated once at every held-out native cell.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from raw_holdout_data import RawHoldoutTask
from raw_point_supervision import (
    PatchNormStats,
    _raw_lasso_warm_start,
    build_raw_point_patches,
)


SAMPLERS = (
    "legacy_capped_selection",
    "all_cells_uniform",
    "all_cells_density_balanced",
)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DirectSPARRegressor(nn.Module):
    """Exact pointwise 300/240→96→24→64→1 anchored SPAR parameterization."""

    def __init__(
        self,
        sequence_length: int,
        anchor_weights: np.ndarray,
        anchor_bias: float,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(sequence_length, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, 24),
            nn.GELU(),
        )
        self.query_decoder = nn.Sequential(
            nn.Linear(24, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.query_decoder[-1].weight)
        nn.init.zeros_(self.query_decoder[-1].bias)
        self.correction_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.register_buffer(
            "anchor_weights",
            torch.as_tensor(anchor_weights, dtype=torch.float32).reshape(sequence_length),
        )
        self.register_buffer(
            "anchor_bias",
            torch.tensor(float(anchor_bias), dtype=torch.float32),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        anchor = torch.einsum("bt,t->b", history, self.anchor_weights) + self.anchor_bias
        correction = self.query_decoder(self.temporal_encoder(history)).squeeze(-1)
        return anchor + self.correction_scale * correction


class CausalResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = int(dilation * (kernel_size - 1))
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def _causal_conv(self, values: torch.Tensor, convolution: nn.Conv1d) -> torch.Tensor:
        return convolution(F.pad(values, (self.left_padding, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.projection(values)
        hidden = self.dropout(F.gelu(self._causal_conv(values, self.conv1)))
        hidden = self.dropout(F.gelu(self._causal_conv(hidden, self.conv2)))
        return F.gelu(hidden + residual)


class CausalTCNRegressor(nn.Module):
    """Compact direct sequence baseline with no spatial or raster features."""

    def __init__(
        self,
        *,
        channels: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_channels = 1
        for dilation in dilations:
            blocks.append(
                CausalResidualBlock(
                    input_channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            input_channels = channels
        self.network = nn.Sequential(*blocks)
        self.readout = nn.Linear(channels, 1)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)
        self.dropout = float(dropout)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded = self.network(history.unsqueeze(1))
        return self.readout(encoded[:, :, -1]).squeeze(-1)


def _flatten_patch_points(arrays) -> tuple[np.ndarray, np.ndarray]:
    if arrays.point_indices is None:
        raise ValueError("Sampler arrays are missing point indices.")
    indices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for row_indices, row_mask in zip(
        arrays.point_indices,
        arrays.point_masks,
        strict=True,
    ):
        valid = row_mask > 0
        selected = row_indices[valid].astype(np.int64)
        if len(selected) == 0:
            continue
        indices.append(selected)
        weights.append(np.full(len(selected), 1.0 / len(selected), dtype=np.float32))
    if not indices:
        raise ValueError("Sampler did not retain any native product cells.")
    return np.concatenate(indices), np.concatenate(weights)


def sampler_indices_and_weights(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    config,
    norm_stats: PatchNormStats,
    raw_increment: np.ndarray,
    *,
    split_code: int,
    sampler: str,
    history_mean: np.ndarray,
    history_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique cell indices and normalized objective weights."""
    if sampler not in SAMPLERS:
        raise ValueError(f"Unsupported sampler: {sampler}")
    expected = np.flatnonzero(raw_task.raw_split_codes == split_code).astype(np.int64)
    if sampler == "all_cells_uniform":
        return expected, np.ones(len(expected), dtype=np.float32)

    arrays = build_raw_point_patches(
        raw_task,
        config,
        norm_stats,
        raw_increment,
        split_code=split_code,
        min_points_per_patch=1 if sampler == "all_cells_density_balanced" else 8,
        max_points_per_patch=None if sampler == "all_cells_density_balanced" else 128,
        raw_history=raw_history,
        input_mean=history_mean,
        input_std=history_std,
    )
    indices, inverse_density_weights = _flatten_patch_points(arrays)
    order = np.argsort(indices)
    indices = indices[order]
    inverse_density_weights = inverse_density_weights[order]
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("A native cell was assigned to more than one pointwise patch.")
    if sampler == "all_cells_density_balanced":
        if not np.array_equal(indices, expected):
            raise AssertionError("Density-balanced sampler must retain every split cell.")
        weights = inverse_density_weights
    else:
        if not np.all(np.isin(indices, expected)):
            raise AssertionError("Legacy capped sampler selected a cell outside the split.")
        weights = np.ones(len(indices), dtype=np.float32)
    weights = (weights / max(float(weights.mean()), 1e-12)).astype(np.float32)
    return indices, weights


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def _save_history(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _native_metrics(
    *,
    model_name: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
    parameter_count: int,
    extra: dict[str, object],
) -> dict[str, object]:
    residual = prediction.astype(np.float64) - truth.astype(np.float64)
    return {
        "model": model_name,
        "native_cell_count": int(len(truth)),
        "native_cell_rmse": float(np.sqrt(np.mean(residual**2))),
        "native_cell_mae": float(np.mean(np.abs(residual))),
        "native_cell_bias": float(np.mean(residual)),
        "direct_raw_rmse": float(np.sqrt(np.mean(residual**2))),
        "direct_raw_mae": float(np.mean(np.abs(residual))),
        "direct_raw_bias": float(np.mean(residual)),
        "native_product_cell_endpoint": True,
        "primary_endpoint": "native_egms_l3_valid_product_cell",
        "target_supervision": "native_egms_l3_product_values_only",
        "interpolated_future_target_used_for_loss": False,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "core_seconds": float(training_seconds + inference_seconds),
        "parameter_count": int(parameter_count),
        **extra,
    }


def run_direct_spar(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    config,
    output_dir: Path,
    *,
    seed: int,
    sampler: str,
    epochs: int = 60,
    patience: int = 12,
    batch_size: int = 1024,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_all = raw_task.train_target_source_indices
    test = raw_task.test_target_indices
    history_mean = raw_history[train_all].mean(axis=0, dtype=np.float64).astype(np.float32)
    history_std = np.maximum(
        raw_history[train_all].std(axis=0, dtype=np.float64),
        1e-6,
    ).astype(np.float32)
    raw_increment = (
        raw_task.raw_target.astype(np.float32) - raw_history[:, -1].astype(np.float32)
    )
    train_increment = raw_increment[train_all]
    increment_mean = float(train_increment.mean(dtype=np.float64))
    increment_std = float(max(train_increment.std(dtype=np.float64), 1e-6))
    norm_stats = PatchNormStats(
        input_mean=float(raw_history[train_all].mean(dtype=np.float64)),
        input_std=float(max(raw_history[train_all].std(dtype=np.float64), 1e-6)),
        residual_mean=increment_mean,
        residual_std=increment_std,
    )
    warm_start = _raw_lasso_warm_start(
        raw_history,
        raw_task,
        config,
        norm_stats,
        str(device),
        history_mean=history_mean,
        history_std=history_std,
    )
    train_indices, train_weights = sampler_indices_and_weights(
        raw_task,
        raw_history,
        config,
        norm_stats,
        raw_increment,
        split_code=0,
        sampler=sampler,
        history_mean=history_mean,
        history_std=history_std,
    )
    val_indices, val_weights = sampler_indices_and_weights(
        raw_task,
        raw_history,
        config,
        norm_stats,
        raw_increment,
        split_code=1,
        sampler=sampler,
        history_mean=history_mean,
        history_std=history_std,
    )

    def x(indices: np.ndarray) -> np.ndarray:
        return (
            (raw_history[indices] - history_mean[None, :])
            / history_std[None, :]
        ).astype(np.float32)

    def y(indices: np.ndarray) -> np.ndarray:
        return (
            (raw_increment[indices] - increment_mean) / increment_std
        ).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x(train_indices)),
        torch.from_numpy(y(train_indices)),
        torch.from_numpy(train_weights),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_x = torch.from_numpy(x(val_indices)).to(device)
    val_y = torch.from_numpy(y(val_indices)).to(device)
    val_w = torch.from_numpy(val_weights).to(device)
    model = DirectSPARRegressor(
        raw_history.shape[1],
        warm_start.value_weights,
        warm_start.bias,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    rows: list[dict[str, object]] = []
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        batches = 0
        for history_batch, target_batch, weight_batch in train_loader:
            history_batch = history_batch.to(device)
            target_batch = target_batch.to(device)
            weight_batch = weight_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            element = F.smooth_l1_loss(
                model(history_batch),
                target_batch,
                reduction="none",
            )
            loss = _weighted_mean(element, weight_batch)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            val_element = F.smooth_l1_loss(model(val_x), val_y, reduction="none")
            val_loss = float(_weighted_mean(val_element, val_w).item())
        rows.append(
            {
                "epoch": epoch,
                "train_weighted_smooth_l1": loss_total / max(batches, 1),
                "val_weighted_smooth_l1": val_loss,
            }
        )
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Direct SPAR did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    test_x = torch.from_numpy(x(test)).to(device)
    inference_started = time.perf_counter()
    with torch.no_grad():
        normalized = model(test_x).detach().cpu().numpy().astype(np.float32)
    prediction = (
        normalized * increment_std
        + increment_mean
        + raw_history[test, -1]
    ).astype(np.float32)
    inference_seconds = time.perf_counter() - inference_started
    truth = raw_task.raw_target[test].astype(np.float32)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    metrics = _native_metrics(
        model_name=f"direct_spar_{sampler}",
        prediction=prediction,
        truth=truth,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=parameter_count,
        extra={
            "sampler": sampler,
            "training_loss": "weighted Smooth-L1 on standardized future increment",
            "anchor": "fixed native-support LASSO",
            "anchor_preserving_initialization": True,
            "best_epoch": int(best_epoch),
            "best_val_weighted_smooth_l1": float(best_val),
            "train_available_cell_count": int(len(train_all)),
            "train_used_cell_count": int(len(train_indices)),
            "train_cell_coverage": float(len(train_indices) / len(train_all)),
            "val_available_cell_count": int(len(raw_task.val_target_source_indices)),
            "val_used_cell_count": int(len(val_indices)),
            "val_cell_coverage": float(
                len(val_indices) / len(raw_task.val_target_source_indices)
            ),
            "history_length": int(raw_history.shape[1]),
            "batch_size": int(batch_size),
            "maximum_epochs": int(epochs),
            "patience": int(patience),
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "device": str(device),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else 0.0
            ),
        },
    )
    np.savez_compressed(
        output_dir / "direct_native_test_predictions.npz",
        indices=test.astype(np.int64),
        points=raw_task.raw_points[test].astype(np.float64),
        truth=truth,
        prediction=prediction,
        residual=prediction - truth,
    )
    np.savez_compressed(
        output_dir / "normalization.npz",
        history_mean=history_mean,
        history_std=history_std,
        increment_mean=np.asarray(increment_mean, dtype=np.float32),
        increment_std=np.asarray(increment_std, dtype=np.float32),
    )
    torch.save(
        {
            "model_state": best_state,
            "sampler": sampler,
            "sequence_length": int(raw_history.shape[1]),
        },
        output_dir / "best_model.pth",
    )
    _save_history(output_dir / "training_history.csv", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def run_native_tcn(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
    epochs: int = 60,
    patience: int = 12,
    batch_size: int = 1024,
    channels: int = 32,
    kernel_size: int = 3,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    dropout: float = 0.1,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    train = raw_task.train_target_source_indices
    val = raw_task.val_target_source_indices
    test = raw_task.test_target_indices
    history_mean = raw_history[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    history_std = np.maximum(
        raw_history[train].std(axis=0, dtype=np.float64),
        1e-6,
    ).astype(np.float32)
    target_mean = float(raw_task.raw_target[train].mean(dtype=np.float64))
    target_std = float(max(raw_task.raw_target[train].std(dtype=np.float64), 1e-6))

    def x(indices: np.ndarray) -> np.ndarray:
        return (
            (raw_history[indices] - history_mean[None, :])
            / history_std[None, :]
        ).astype(np.float32)

    def y(indices: np.ndarray) -> np.ndarray:
        return (
            (raw_task.raw_target[indices] - target_mean) / target_std
        ).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(x(train)), torch.from_numpy(y(train)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_x = torch.from_numpy(x(val)).to(device)
    val_y = torch.from_numpy(y(val)).to(device)
    model = CausalTCNRegressor(
        channels=channels,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    stale = 0
    rows: list[dict[str, object]] = []
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for history_batch, target_batch in loader:
            history_batch = history_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history_batch)
            loss = F.mse_loss(prediction, target_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_rmse = float(
                torch.sqrt(F.mse_loss(val_prediction, val_y)).item() * target_std
            )
        rows.append(
            {
                "epoch": epoch,
                "train_standardized_mse": total / max(batches, 1),
                "val_native_cell_rmse": val_rmse,
            }
        )
        if val_rmse < best_val_rmse - 1e-7:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Causal TCN did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    test_x = torch.from_numpy(x(test)).to(device)
    inference_started = time.perf_counter()
    with torch.no_grad():
        prediction = (
            model(test_x).detach().cpu().numpy().astype(np.float32) * target_std
            + target_mean
        )
    inference_seconds = time.perf_counter() - inference_started
    truth = raw_task.raw_target[test].astype(np.float32)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    metrics = _native_metrics(
        model_name="native_causal_tcn",
        prediction=prediction,
        truth=truth,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=parameter_count,
        extra={
            "architecture": "causal_residual_tcn",
            "channels": int(channels),
            "kernel_size": int(kernel_size),
            "dilations": list(dilations),
            "dropout": float(dropout),
            "best_epoch": int(best_epoch),
            "best_val_native_cell_rmse": float(best_val_rmse),
            "history_length": int(raw_history.shape[1]),
            "batch_size": int(batch_size),
            "maximum_epochs": int(epochs),
            "patience": int(patience),
            "loss": "MSE on standardized EGMS L3 product values",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "device": str(device),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else 0.0
            ),
        },
    )
    np.savez_compressed(
        output_dir / "direct_native_test_predictions.npz",
        indices=test.astype(np.int64),
        points=raw_task.raw_points[test].astype(np.float64),
        truth=truth,
        prediction=prediction,
        residual=prediction - truth,
    )
    torch.save(
        {
            "model_state": best_state,
            "history_mean": torch.from_numpy(history_mean),
            "history_std": torch.from_numpy(history_std),
            "target_mean": torch.tensor(target_mean, dtype=torch.float32),
            "target_std": torch.tensor(target_std, dtype=torch.float32),
        },
        output_dir / "best_model.pth",
    )
    _save_history(output_dir / "training_history.csv", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics
