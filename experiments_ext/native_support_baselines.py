"""Native EGMS Level-3 product-cell baselines for the 300-to-one task."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import random
import time
from pathlib import Path

import numpy as np

from raw_holdout_data import RawHoldoutSpec, RawHoldoutTask
from raw_point_supervision import _direct_point_metrics


def load_forecast_dates(spec: RawHoldoutSpec) -> tuple[np.ndarray, float, dict[str, object]]:
    """Return history and target times in days relative to the first history date."""
    with spec.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    history_labels = header[
        spec.history_start_col : spec.history_start_col + spec.history_length
    ]
    target_label = header[spec.target_col]
    skipped_labels = header[
        spec.history_start_col + spec.history_length : spec.target_col
    ]
    def parse_date(label: str) -> np.datetime64:
        return np.datetime64(datetime.strptime(label, "%Y%m%d").date(), "D")

    history_dates = np.asarray(
        [parse_date(label) for label in history_labels],
        dtype="datetime64[D]",
    )
    target_date = parse_date(target_label)
    origin = history_dates[0]
    history_days = (history_dates - origin).astype(np.float64)
    target_day = float((target_date - origin).astype(np.float64))
    return history_days, target_day, {
        "history_date_start": history_labels[0],
        "history_date_end": history_labels[-1],
        "history_date_count": len(history_labels),
        "skipped_dates": skipped_labels,
        "target_date": target_label,
        "forecast_horizon_days_from_last_history": int(
            (target_date - history_dates[-1]).astype(np.int64)
        ),
    }


def _metrics(
    *,
    model: str,
    prediction: np.ndarray,
    raw_task: RawHoldoutTask,
    training_seconds: float,
    inference_seconds: float,
    parameter_count: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    test = raw_task.test_target_indices
    truth = raw_task.raw_target[test].astype(np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    direct = _direct_point_metrics(
        prediction,
        truth,
        raw_task.raw_points[test],
        raw_task,
    )
    residual = prediction - truth
    return {
        "model": model,
        **direct,
        "native_product_cell_endpoint": True,
        "primary_endpoint": "native_egms_l3_valid_product_cell",
        "target_supervision": "native_egms_l3_product_values_only",
        "interpolated_future_target_used_for_loss": False,
        "native_cell_count": int(len(test)),
        "native_cell_rmse": float(np.sqrt(np.mean(residual**2))),
        "native_cell_mae": float(np.mean(np.abs(residual))),
        "native_cell_bias": float(np.mean(residual)),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "core_seconds": float(training_seconds + inference_seconds),
        "parameter_count": int(parameter_count),
        **({} if extra is None else extra),
        **raw_task.metadata,
    }


def _save(
    output_dir: Path,
    raw_task: RawHoldoutTask,
    prediction: np.ndarray,
    metrics: dict[str, object],
    *,
    training_history: list[dict[str, object]] | None = None,
    state: object | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test = raw_task.test_target_indices
    truth = raw_task.raw_target[test].astype(np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    np.savez_compressed(
        output_dir / "direct_native_test_predictions.npz",
        indices=test.astype(np.int64),
        points=raw_task.raw_points[test].astype(np.float64),
        truth=truth,
        prediction=prediction,
        residual=prediction - truth,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    if training_history:
        with (output_dir / "training_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(training_history[0]))
            writer.writeheader()
            writer.writerows(training_history)
    if state is not None:
        import torch

        torch.save(state, output_dir / "best_model.pth")


def run_native_persistence(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    prediction = raw_history[raw_task.test_target_indices, -1].astype(np.float32)
    inference_seconds = time.perf_counter() - started
    metrics = _metrics(
        model="native_persistence",
        prediction=prediction,
        raw_task=raw_task,
        training_seconds=0.0,
        inference_seconds=inference_seconds,
        parameter_count=0,
        extra={
            "forecast_rule": "last observed native L3 product value",
        },
    )
    _save(output_dir, raw_task, prediction, metrics)
    return metrics


def run_native_linear_trend(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    history_days: np.ndarray,
    target_day: float,
    output_dir: Path,
) -> dict[str, object]:
    """Per-cell ordinary least-squares trend over all 300 dated observations."""
    test = raw_task.test_target_indices
    centered_time = history_days - float(history_days.mean())
    denominator = float(np.sum(centered_time**2))
    if denominator <= 0:
        raise ValueError("History dates do not span a positive interval.")
    started = time.perf_counter()
    histories = raw_history[test].astype(np.float64)
    means = histories.mean(axis=1)
    slopes = (histories - means[:, None]) @ centered_time / denominator
    prediction = (means + slopes * (target_day - float(history_days.mean()))).astype(
        np.float32
    )
    inference_seconds = time.perf_counter() - started
    metrics = _metrics(
        model="native_linear_trend",
        prediction=prediction,
        raw_task=raw_task,
        training_seconds=0.0,
        inference_seconds=inference_seconds,
        parameter_count=0,
        extra={
            "forecast_rule": "per-cell OLS over all dated history values",
            "history_time_unit": "days",
            "history_span_days": float(history_days[-1] - history_days[0]),
            "target_day_from_origin": float(target_day),
        },
    )
    _save(output_dir, raw_task, prediction, metrics)
    return metrics


class DLinear:
    """Factory namespace for a compact univariate DLinear baseline."""

    @staticmethod
    def build(sequence_length: int, moving_average: int = 25):
        import torch
        import torch.nn as nn

        if moving_average < 1 or moving_average % 2 == 0:
            raise ValueError("moving_average must be a positive odd integer.")

        class _DLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sequence_length = int(sequence_length)
                self.moving_average = int(moving_average)
                self.seasonal = nn.Linear(sequence_length, 1)
                self.trend = nn.Linear(sequence_length, 1)
                nn.init.constant_(self.seasonal.weight, 1.0 / sequence_length)
                nn.init.constant_(self.trend.weight, 1.0 / sequence_length)
                nn.init.zeros_(self.seasonal.bias)
                nn.init.zeros_(self.trend.bias)

            def moving_mean(self, history):
                pad = (self.moving_average - 1) // 2
                left = history[:, :1].repeat(1, pad)
                right = history[:, -1:].repeat(1, pad)
                padded = torch.cat([left, history, right], dim=1).unsqueeze(1)
                return torch.nn.functional.avg_pool1d(
                    padded,
                    kernel_size=self.moving_average,
                    stride=1,
                ).squeeze(1)

            def forward(self, history):
                trend = self.moving_mean(history)
                seasonal = history - trend
                return (self.seasonal(seasonal) + self.trend(trend)).squeeze(-1)

        return _DLinear()


def run_native_dlinear(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
    epochs: int = 60,
    patience: int = 12,
    batch_size: int = 4096,
    moving_average: int = 25,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
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
    target_std = float(
        max(raw_task.raw_target[train].std(dtype=np.float64), 1e-6)
    )

    def standardized_history(indices: np.ndarray) -> np.ndarray:
        return (
            (raw_history[indices] - history_mean[None, :])
            / history_std[None, :]
        ).astype(np.float32)

    def standardized_target(indices: np.ndarray) -> np.ndarray:
        return (
            (raw_task.raw_target[indices] - target_mean) / target_std
        ).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(standardized_history(train)),
        torch.from_numpy(standardized_target(train)),
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    val_x = torch.from_numpy(standardized_history(val)).to(device)
    val_y = torch.from_numpy(standardized_target(val)).to(device)
    model = DLinear.build(raw_history.shape[1], moving_average=moving_average).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    stale = 0
    rows: list[dict[str, object]] = []
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        batches = 0
        for history_batch, target_batch in train_loader:
            history_batch = history_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history_batch)
            loss = F.mse_loss(prediction, target_batch)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
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
                "train_standardized_mse": loss_total / max(batches, 1),
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
        raise RuntimeError("DLinear failed to produce a validation checkpoint.")
    model.load_state_dict(best_state)
    test_x = torch.from_numpy(standardized_history(test)).to(device)
    inference_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        prediction = (
            model(test_x).detach().cpu().numpy().astype(np.float32) * target_std
            + target_mean
        )
    inference_seconds = time.perf_counter() - inference_started
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    state = {
        "model_state": best_state,
        "history_mean": torch.from_numpy(history_mean),
        "history_std": torch.from_numpy(history_std),
        "target_mean": torch.tensor(target_mean, dtype=torch.float32),
        "target_std": torch.tensor(target_std, dtype=torch.float32),
        "moving_average": int(moving_average),
        "sequence_length": int(raw_history.shape[1]),
    }
    metrics = _metrics(
        model="native_dlinear",
        prediction=prediction,
        raw_task=raw_task,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=parameter_count,
        extra={
            "best_epoch": int(best_epoch),
            "best_val_native_cell_rmse": float(best_val_rmse),
            "moving_average": int(moving_average),
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "maximum_epochs": int(epochs),
            "patience": int(patience),
            "batch_size": int(batch_size),
            "loss": "MSE on standardized EGMS L3 product values",
            "device": str(device),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else 0.0
            ),
        },
    )
    _save(
        output_dir,
        raw_task,
        prediction,
        metrics,
        training_history=rows,
        state=state,
    )
    return metrics
