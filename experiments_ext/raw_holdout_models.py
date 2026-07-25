"""Fair baselines and Hybrid CNN-LSTM on a pre-gridding raw holdout task."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from deep_patch_models import (  # noqa: E402
    _adapt_patch_input_layout,
    _build_patch_arrays,
    _compute_patch_norm_stats,
    _fit_lasso_warm_start,
    _predict_full_map,
    _run_loader_epoch,
    _select_model_builder,
)
from revision_config import RevisionConfig  # noqa: E402
from revision_experiments import (  # noqa: E402
    _fit_torch_l1_regressor,
    _predict_torch_l1_regressor,
)
from revision_utils import (  # noqa: E402
    build_tabular_dataset,
    set_random_seed,
    split_from_eligible_indices,
)

from raw_holdout_data import RawHoldoutTask, cell_aggregated_metrics, idw_interpolate, raw_point_metrics, sample_grid_at_raw_points
from hybrid_ablation import apply_hybrid_ablation, trainable_parameter_count
from modern_baselines import PatchSimVPStyleResidualModel


def _ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_prediction_evidence(output_dir: Path, prediction_grid: np.ndarray, raw_task: RawHoldoutTask) -> None:
    test_indices = raw_task.test_target_indices
    raw_prediction = sample_grid_at_raw_points(prediction_grid, raw_task, test_indices)
    np.save(output_dir / "prediction_grid.npy", np.asarray(prediction_grid, dtype=np.float32))
    np.savez_compressed(
        output_dir / "raw_test_predictions.npz",
        indices=test_indices.astype(np.int64),
        points=raw_task.raw_points[test_indices].astype(np.float64),
        truth=raw_task.raw_target[test_indices].astype(np.float32),
        prediction=raw_prediction.astype(np.float32),
        residual=(raw_prediction - raw_task.raw_target[test_indices]).astype(np.float32),
    )


def run_persistence(raw_task: RawHoldoutTask, output_dir: Path) -> dict[str, object]:
    output_dir = _ensure_output(output_dir)
    started = time.perf_counter()
    prediction = raw_task.dense_task.input_maps[-1].copy()
    metrics: dict[str, object] = {
        "model": "persistence",
        **raw_point_metrics(prediction, raw_task, split="test"),
        **cell_aggregated_metrics(prediction, raw_task, split="test"),
        "training_seconds": 0.0,
        "inference_seconds": float(time.perf_counter() - started),
        "parameter_count": 0,
        **raw_task.metadata,
    }
    _save_prediction_evidence(output_dir, prediction, raw_task)
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_target_idw_diagnostic(raw_task: RawHoldoutTask, output_dir: Path, *, neighbors: int = 8, power: float = 2.0) -> dict[str, object]:
    """Non-deployable diagnostic: interpolate future train targets to test points."""
    output_dir = _ensure_output(output_dir)
    grid_east, grid_north = np.meshgrid(raw_task.easting_axis, raw_task.northing_axis, indexing="ij")
    queries = np.column_stack((grid_east.ravel(), grid_north.ravel()))
    started = time.perf_counter()
    prediction = idw_interpolate(
        raw_task.raw_points[raw_task.train_target_source_indices],
        raw_task.raw_target[raw_task.train_target_source_indices],
        queries,
        neighbors=neighbors,
        power=power,
    ).reshape(len(raw_task.easting_axis), len(raw_task.northing_axis))
    metrics: dict[str, object] = {
        "model": "target_idw_train_only_diagnostic",
        **raw_point_metrics(prediction, raw_task, split="test"),
        **cell_aggregated_metrics(prediction, raw_task, split="test"),
        "training_seconds": 0.0,
        "inference_seconds": float(time.perf_counter() - started),
        "parameter_count": 0,
        "deployable_forecast": False,
        "uses_future_target_values_from_train_blocks": True,
        **raw_task.metadata,
    }
    _save_prediction_evidence(output_dir, prediction, raw_task)
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_lasso(raw_task: RawHoldoutTask, config: RevisionConfig, output_dir: Path) -> dict[str, object]:
    import torch

    output_dir = _ensure_output(output_dir)
    set_random_seed(config.split_seed)
    task = raw_task.dense_task
    X_all, y_all, eligible_indices = build_tabular_dataset(task)
    positions = split_from_eligible_indices(task, eligible_indices)
    X_train, y_train = X_all[positions["train"]], y_all[positions["train"]]
    X_val, y_val = X_all[positions["val"]], y_all[positions["val"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_alphas = sorted({1e-4, 5e-4, config.lasso_alpha, 5e-3, 1e-2})
    rows: list[dict[str, object]] = []
    best_state = None
    best_alpha = None
    best_val = float("inf")
    best_epoch = 0
    training_started = time.perf_counter()
    for alpha in candidate_alphas:
        state, val_rmse, epoch = _fit_torch_l1_regressor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            alpha=alpha,
            config=config,
            device_str=device,
        )
        rows.append({"alpha": float(alpha), "val_dense_rmse": float(val_rmse), "best_epoch": int(epoch)})
        if val_rmse < best_val:
            best_state, best_alpha, best_val, best_epoch = state, alpha, val_rmse, epoch
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("LASSO failed to produce a state.")

    inference_started = time.perf_counter()
    # Inference must cover the full grid, including unsupervised buffer cells.
    # Restricting reconstruction to ``eligible_indices`` leaves NaNs beside
    # held-out blocks and contaminates bilinear sampling at raw test points.
    full_inputs = task.input_maps.reshape(task.input_maps.shape[0], -1).T
    prediction_values = _predict_torch_l1_regressor(full_inputs, best_state, device_str=device)
    prediction_grid = prediction_values.reshape(task.target_map.shape).astype(np.float32)
    inference_seconds = time.perf_counter() - inference_started
    metrics: dict[str, object] = {
        "model": "lasso",
        **raw_point_metrics(prediction_grid, raw_task, split="test"),
        **cell_aggregated_metrics(prediction_grid, raw_task, split="test"),
        "best_alpha": float(best_alpha),
        "best_val_dense_rmse": float(best_val),
        "best_epoch": int(best_epoch),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "parameter_count": int(X_all.shape[1] + 1),
        "device": device,
        **raw_task.metadata,
    }
    torch.save(best_state, output_dir / "lasso_state.pth")
    _write_rows(output_dir / "alpha_sweep.csv", rows)
    _save_prediction_evidence(output_dir, prediction_grid, raw_task)
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_patch_model(
    raw_task: RawHoldoutTask,
    config: RevisionConfig,
    output_dir: Path,
    *,
    model_name: str,
    model_kind: str,
    use_warm_start: bool,
    disable_recent_gate: bool = False,
    disable_spatial_correction: bool = False,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    output_dir = _ensure_output(output_dir)
    set_random_seed(config.split_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    task = raw_task.dense_task
    norm_stats = _compute_patch_norm_stats(task)
    train_patches = _build_patch_arrays(
        task=task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=task.train_mask,
        use_input_mask=True,
    )
    val_patches = _build_patch_arrays(
        task=task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=task.val_mask,
        use_input_mask=True,
    )
    if model_kind == "simvp_style_residual":
        train_inputs = train_patches.inputs.astype(np.float32)
        val_inputs = val_patches.inputs.astype(np.float32)
    else:
        train_inputs = _adapt_patch_input_layout(train_patches.inputs, model_kind)
        val_inputs = _adapt_patch_input_layout(val_patches.inputs, model_kind)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    warm_start_started = time.perf_counter()
    warm_start = None
    if use_warm_start:
        warm_start = _fit_lasso_warm_start(task, config, norm_stats, str(device))
    warm_start_seconds = time.perf_counter() - warm_start_started
    if model_kind == "simvp_style_residual":
        builder = PatchSimVPStyleResidualModel(
            input_channels=train_patches.inputs.shape[2],
            hidden_channels=max(16, config.nontransformer_hybrid_hidden_channels // 2),
            temporal_bins=16,
        )
    else:
        builder = _select_model_builder(
            model_kind=model_kind,
            time_steps=train_patches.inputs.shape[1],
            input_channels=train_patches.inputs.shape[2],
            patch_size=config.patch_size,
            config=config,
            warm_start=warm_start,
        )
    model = builder.build().to(device)
    disabled_components = apply_hybrid_ablation(
        model,
        disable_recent_gate=disable_recent_gate,
        disable_spatial_correction=disable_spatial_correction,
    ) if (disable_recent_gate or disable_spatial_correction) else ()
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
    patience_counter = 0
    history: list[dict[str, object]] = []
    training_started = time.perf_counter()
    for epoch in range(1, config.cnn_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _run_loader_epoch(model, train_loader, optimizer, device, loss_mask_mode="masked")
        val_loss = _run_loader_epoch(model, val_loader, None, device, loss_mask_mode="masked")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "epoch_seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if val_loss < best_val - 1e-9:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.cnn_patience:
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError(f"{model_name} did not produce a checkpoint.")
    model.load_state_dict(best_state)

    inference_started = time.perf_counter()
    if model_kind == "simvp_style_residual":
        prediction_grid = _predict_full_map_custom_layout(model, task, config, norm_stats, device)
    else:
        prediction_grid = _predict_full_map(
            model=model,
            task=task,
            config=config,
            norm_stats=norm_stats,
            use_input_mask=True,
            model_kind=model_kind,
            device=device,
        )
    inference_seconds = time.perf_counter() - inference_started
    metrics: dict[str, object] = {
        "model": model_name,
        "model_kind": model_kind,
        **raw_point_metrics(prediction_grid, raw_task, split="test"),
        **cell_aggregated_metrics(prediction_grid, raw_task, split="test"),
        "best_epoch": int(best_epoch),
        "best_val_normalized_loss": float(best_val),
        "warm_start_enabled": bool(use_warm_start),
        "convlstm_num_layers": int(config.convlstm_num_layers) if model_kind == "cnn_lstm_hybrid" else None,
        "disabled_components": list(disabled_components),
        "warm_start_seconds": float(warm_start_seconds),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": trainable_parameter_count(model),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "device": str(device),
        "train_patch_count": int(len(train_dataset)),
        "val_patch_count": int(len(val_dataset)),
        "normalization_input_mean": float(norm_stats.input_mean),
        "normalization_input_std": float(norm_stats.input_std),
        "normalization_residual_mean": float(norm_stats.residual_mean),
        "normalization_residual_std": float(norm_stats.residual_std),
        **raw_task.metadata,
    }
    torch.save(best_state, output_dir / "best_model.pth")
    _write_rows(output_dir / "training_history.csv", history)
    _save_prediction_evidence(output_dir, prediction_grid, raw_task)
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def _predict_full_map_custom_layout(model, task, config: RevisionConfig, norm_stats, device) -> np.ndarray:
    """Full-grid overlap-add inference for models that keep B,T,C,H,W."""
    import torch

    patches = _build_patch_arrays(
        task=task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=None,
        use_input_mask=True,
    )
    inputs = torch.tensor(patches.inputs, dtype=torch.float32, device=device)
    prediction_sum = np.zeros(task.target_map.shape, dtype=np.float64)
    prediction_weight = np.zeros(task.target_map.shape, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), config.patch_batch_size):
            end = min(start + config.patch_batch_size, len(inputs))
            residual_norm = model(inputs[start:end]).detach().cpu().numpy()[:, 0]
            residual = residual_norm * norm_stats.residual_std + norm_stats.residual_mean
            absolute = residual + patches.last_frames[start:end, 0]
            for local, patch in enumerate(absolute):
                row, col = patches.positions[start + local]
                row_slice = slice(int(row), int(row) + config.patch_size)
                col_slice = slice(int(col), int(col) + config.patch_size)
                prediction_sum[row_slice, col_slice] += patch
                prediction_weight[row_slice, col_slice] += 1.0
    return (prediction_sum / np.clip(prediction_weight, 1.0, None)).astype(np.float32)
