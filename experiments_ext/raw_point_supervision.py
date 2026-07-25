"""Direct raw-observation supervision for dense patch forecasting.

Unlike the legacy pipeline, no interpolated future target map is used in the
loss.  Each raw target observation is assigned to one spatial patch and the
predicted residual patch is bilinearly sampled at the observation coordinate.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from deep_patch_models import (  # noqa: E402
    LinearWarmStart,
    PatchNormStats,
    _adapt_patch_input_layout,
    _build_patch_arrays,
    _predict_full_map,
    _select_model_builder,
)
from revision_config import RevisionConfig  # noqa: E402
from revision_experiments import _fit_torch_l1_regressor  # noqa: E402
from revision_experiments import _predict_torch_l1_regressor  # noqa: E402
from revision_utils import set_random_seed  # noqa: E402

from raw_holdout_data import RawHoldoutTask, cell_aggregated_metrics, raw_point_metrics, sample_grid_at_raw_points
from hybrid_ablation import apply_hybrid_ablation, trainable_parameter_count
from modern_baselines import PatchSimVPStyleResidualModel
from support_aware_model import SupportAwarePointQueryModel


@dataclass
class RawPointPatchArrays:
    inputs: np.ndarray
    sample_coordinates: np.ndarray
    residual_targets: np.ndarray
    point_masks: np.ndarray
    positions: np.ndarray
    raw_point_count: int
    point_histories: np.ndarray | None = None
    point_indices: np.ndarray | None = None
    global_coordinates: np.ndarray | None = None


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _raw_norm_stats(
    raw_task: RawHoldoutTask,
    formulation: str = "normalized_residual",
) -> tuple[PatchNormStats, np.ndarray, bool]:
    task = raw_task.dense_task
    train_values = task.input_maps[:, task.train_mask].astype(np.float32)
    last_at_points = sample_grid_at_raw_points(task.input_maps[-1], raw_task, np.arange(len(raw_task.raw_points)))
    raw_residual = raw_task.raw_target - last_at_points
    if formulation == "normalized_residual":
        raw_output = raw_residual
        train_output = raw_output[raw_task.train_target_source_indices]
        output_mean = float(train_output.mean())
        output_std = float(max(train_output.std(), 1e-6))
        add_last_frame = True
    elif formulation == "raw_residual":
        raw_output = raw_residual
        output_mean = 0.0
        output_std = 1.0
        add_last_frame = True
    elif formulation == "normalized_absolute":
        raw_output = raw_task.raw_target.astype(np.float32)
        train_output = raw_output[raw_task.train_target_source_indices]
        output_mean = float(train_output.mean())
        output_std = float(max(train_output.std(), 1e-6))
        add_last_frame = False
    else:
        raise ValueError(f"Unsupported formulation: {formulation}")
    stats = PatchNormStats(
        input_mean=float(train_values.mean()),
        input_std=float(max(train_values.std(), 1e-6)),
        residual_mean=output_mean,
        residual_std=output_std,
    )
    return stats, raw_output.astype(np.float32), add_last_frame


def _patch_starts(grid_size: int, patch_size: int, stride: int) -> list[int]:
    starts = list(range(0, max(grid_size - patch_size, 0) + 1, stride))
    if starts[-1] != grid_size - patch_size:
        starts.append(grid_size - patch_size)
    return starts


def _nearest_containing_start(coordinate: float, starts: list[int], patch_size: int) -> int:
    candidates = [start for start in starts if start - 1e-6 <= coordinate <= start + patch_size - 1 + 1e-6]
    if not candidates:
        return min(starts, key=lambda start: abs(coordinate - (start + (patch_size - 1) / 2.0)))
    return min(candidates, key=lambda start: abs(coordinate - (start + (patch_size - 1) / 2.0)))


def build_raw_point_patches(
    raw_task: RawHoldoutTask,
    config: RevisionConfig,
    norm_stats: PatchNormStats,
    raw_output: np.ndarray,
    *,
    split_code: int,
    max_points_per_patch: int | None = 256,
    min_points_per_patch: int = 8,
    raw_point_weights: np.ndarray | None = None,
    raw_history: np.ndarray | None = None,
    input_mean: float | np.ndarray | None = None,
    input_std: float | np.ndarray | None = None,
) -> RawPointPatchArrays:
    task = raw_task.dense_task
    base = _build_patch_arrays(
        task=task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=None,
        use_input_mask=True,
    )
    position_to_index = {tuple(map(int, position)): idx for idx, position in enumerate(base.positions)}
    row_starts = _patch_starts(task.target_map.shape[0], config.patch_size, config.patch_stride)
    col_starts = _patch_starts(task.target_map.shape[1], config.patch_size, config.patch_stride)
    east_index = (raw_task.raw_points[:, 0] - raw_task.easting_axis[0]) / (raw_task.easting_axis[-1] - raw_task.easting_axis[0]) * (len(raw_task.easting_axis) - 1)
    north_index = (raw_task.raw_points[:, 1] - raw_task.northing_axis[0]) / (raw_task.northing_axis[-1] - raw_task.northing_axis[0]) * (len(raw_task.northing_axis) - 1)
    selected_indices = np.flatnonzero(raw_task.raw_split_codes == split_code)
    assigned: dict[tuple[int, int], list[int]] = {}
    for raw_index in selected_indices:
        row = _nearest_containing_start(float(east_index[raw_index]), row_starts, config.patch_size)
        col = _nearest_containing_start(float(north_index[raw_index]), col_starts, config.patch_size)
        assigned.setdefault((row, col), []).append(int(raw_index))

    effective_max_points = (
        max((len(indices) for indices in assigned.values()), default=1)
        if max_points_per_patch is None
        else int(max_points_per_patch)
    )

    rng = np.random.default_rng(config.split_seed + 1009 * (split_code + 2))
    inputs: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    point_histories: list[np.ndarray] = []
    point_indices: list[np.ndarray] = []
    global_coordinates: list[np.ndarray] = []
    total_points = 0
    if raw_history is not None and (input_mean is None or input_std is None):
        raise ValueError("input_mean and input_std are required with raw_history.")
    for position in sorted(assigned):
        raw_indices = np.asarray(assigned[position], dtype=np.int64)
        if len(raw_indices) < min_points_per_patch:
            continue
        if len(raw_indices) > effective_max_points:
            raw_indices = np.sort(rng.choice(raw_indices, size=effective_max_points, replace=False))
        row, col = position
        local_row = east_index[raw_indices] - row
        local_col = north_index[raw_indices] - col
        grid_x = 2.0 * local_col / max(config.patch_size - 1, 1) - 1.0
        grid_y = 2.0 * local_row / max(config.patch_size - 1, 1) - 1.0
        coord = np.column_stack((grid_x, grid_y)).astype(np.float32)
        global_x = 2.0 * north_index[raw_indices] / max(len(raw_task.northing_axis) - 1, 1) - 1.0
        global_y = 2.0 * east_index[raw_indices] / max(len(raw_task.easting_axis) - 1, 1) - 1.0
        global_coord = np.column_stack((global_x, global_y)).astype(np.float32)
        residual_norm = ((raw_output[raw_indices] - norm_stats.residual_mean) / norm_stats.residual_std).astype(np.float32)
        pad = effective_max_points - len(raw_indices)
        if pad > 0:
            coord = np.pad(coord, ((0, pad), (0, 0)), mode="constant")
            residual_norm = np.pad(residual_norm, (0, pad), mode="constant")
            global_coord = np.pad(global_coord, ((0, pad), (0, 0)), mode="constant")
        if raw_history is not None:
            history = (
                (raw_history[raw_indices] - np.asarray(input_mean, dtype=np.float32))
                / np.asarray(input_std, dtype=np.float32)
            ).astype(np.float32)
            point_index = raw_indices.astype(np.int64)
            if pad > 0:
                history = np.pad(history, ((0, pad), (0, 0)), mode="constant")
                point_index = np.pad(point_index, (0, pad), mode="constant", constant_values=-1)
            point_histories.append(history)
            point_indices.append(point_index)
            global_coordinates.append(global_coord)
        point_mask = np.zeros(effective_max_points, dtype=np.float32)
        point_mask[: len(raw_indices)] = (
            1.0 if raw_point_weights is None else raw_point_weights[raw_indices].astype(np.float32)
        )
        inputs.append(base.inputs[position_to_index[position]])
        coordinates.append(coord)
        targets.append(residual_norm)
        masks.append(point_mask)
        positions.append(np.asarray(position, dtype=np.int32))
        total_points += len(raw_indices)
    if not inputs:
        raise ValueError("No raw-supervised patches were created.")
    return RawPointPatchArrays(
        inputs=np.stack(inputs).astype(np.float32),
        sample_coordinates=np.stack(coordinates).astype(np.float32),
        residual_targets=np.stack(targets).astype(np.float32),
        point_masks=np.stack(masks).astype(np.float32),
        positions=np.stack(positions).astype(np.int32),
        raw_point_count=int(total_points),
        point_histories=(np.stack(point_histories).astype(np.float32) if point_histories else None),
        point_indices=(np.stack(point_indices).astype(np.int64) if point_indices else None),
        global_coordinates=(np.stack(global_coordinates).astype(np.float32) if global_coordinates else None),
    )


def _raw_lasso_warm_start(
    raw_history: np.ndarray,
    raw_task: RawHoldoutTask,
    config: RevisionConfig,
    norm_stats: PatchNormStats,
    device: str,
    *,
    history_mean: float | np.ndarray | None = None,
    history_std: float | np.ndarray | None = None,
) -> LinearWarmStart:
    train = raw_task.train_target_source_indices
    val = raw_task.val_target_source_indices
    state, _, _ = _fit_torch_l1_regressor(
        X_train=raw_history[train],
        y_train=raw_task.raw_target[train],
        X_val=raw_history[val],
        y_val=raw_task.raw_target[val],
        alpha=config.lasso_alpha,
        config=config,
        device_str=device,
    )
    weights_raw = (
        state["weights"].numpy().astype(np.float32)
        / state["X_std"].numpy().astype(np.float32).reshape(-1)
        * float(state["y_std"])
    )
    bias_raw = float(state["bias"]) * float(state["y_std"]) + float(state["y_mean"])
    bias_raw -= float(np.sum(weights_raw * state["X_mean"].numpy().astype(np.float32).reshape(-1)))
    residual_weights = weights_raw.copy()
    residual_weights[-1] -= 1.0
    anchor_history_mean = np.asarray(
        norm_stats.input_mean if history_mean is None else history_mean,
        dtype=np.float32,
    )
    anchor_history_std = np.asarray(
        norm_stats.input_std if history_std is None else history_std,
        dtype=np.float32,
    )
    normalized_weights = residual_weights * (anchor_history_std / norm_stats.residual_std)
    normalized_bias = (
        float(np.sum(residual_weights * anchor_history_mean)) + bias_raw - norm_stats.residual_mean
    ) / norm_stats.residual_std
    return LinearWarmStart(value_weights=normalized_weights.astype(np.float32), bias=float(normalized_bias))


def _run_raw_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    import torch
    import torch.nn.functional as F

    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    grad_sum = 0.0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, coordinates, targets, masks in loader:
            inputs = inputs.to(device)
            coordinates = coordinates.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            sampled = F.grid_sample(
                prediction,
                coordinates[:, None, :, :],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[:, 0, 0, :]
            element_loss = F.smooth_l1_loss(sampled, targets, reduction="none")
            loss = (element_loss * masks).sum() / masks.sum().clamp_min(1.0)
            if training:
                loss.backward()
                grad_sq = torch.zeros((), device=device)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        grad_sq = grad_sq + parameter.grad.detach().pow(2).sum()
                grad_sum += float(torch.sqrt(grad_sq).item())
                optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
    return loss_sum / max(batches, 1), grad_sum / max(batches, 1) if training else 0.0


def _run_support_query_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    import torch
    import torch.nn.functional as F

    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    grad_sum = 0.0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, coordinates, histories, global_coordinates, targets, masks in loader:
            inputs = inputs.to(device)
            coordinates = coordinates.to(device)
            histories = histories.to(device)
            global_coordinates = global_coordinates.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, coordinates, histories, global_coordinates)
            element_loss = F.smooth_l1_loss(prediction, targets, reduction="none")
            loss = (element_loss * masks).sum() / masks.sum().clamp_min(1.0)
            if training:
                loss.backward()
                grad_sq = torch.zeros((), device=device)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        grad_sq = grad_sq + parameter.grad.detach().pow(2).sum()
                grad_sum += float(torch.sqrt(grad_sq).item())
                optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
    return loss_sum / max(batches, 1), grad_sum / max(batches, 1) if training else 0.0


def run_raw_point_supervised_model(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    config: RevisionConfig,
    output_dir: Path,
    *,
    model_name: str = "cnn_lstm_raw_supervised",
    model_kind: str = "cnn_lstm_hybrid",
    use_warm_start: bool = True,
    formulation: str = "normalized_residual",
    raw_quality_rmse: np.ndarray | None = None,
    disable_recent_gate: bool = False,
    disable_spatial_correction: bool = False,
    support_use_spatial_context: bool = True,
    support_use_global_coordinates: bool = True,
    support_use_local_coordinates: bool = True,
    support_history_source: str = "direct_raw_point",
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(config.split_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    norm_stats, raw_output, add_last_frame = _raw_norm_stats(raw_task, formulation=formulation)
    support_query = model_kind == "support_aware_point_query"
    if support_query and formulation != "normalized_residual":
        raise ValueError("Support-aware query model requires normalized_residual formulation.")
    if support_query:
        direct_residual = raw_task.raw_target.astype(np.float32) - raw_history[:, -1].astype(np.float32)
        train_residual = direct_residual[raw_task.train_target_source_indices]
        norm_stats = PatchNormStats(
            input_mean=norm_stats.input_mean,
            input_std=norm_stats.input_std,
            residual_mean=float(train_residual.mean()),
            residual_std=float(max(train_residual.std(), 1e-6)),
        )
        raw_output = direct_residual
        add_last_frame = True
        query_history_mean = raw_history[raw_task.train_target_source_indices].mean(axis=0).astype(np.float32)
        query_history_std = np.maximum(
            raw_history[raw_task.train_target_source_indices].std(axis=0), 1e-6
        ).astype(np.float32)
    else:
        query_history_mean = None
        query_history_std = None
    train_weights = None
    quality_weighted = raw_quality_rmse is not None
    if raw_quality_rmse is not None:
        if len(raw_quality_rmse) != len(raw_task.raw_points):
            raise ValueError("Quality array length differs from raw point count.")
        floor = max(float(np.quantile(raw_quality_rmse[raw_task.train_target_source_indices], 0.10)), 0.3)
        train_weights = 1.0 / (np.maximum(raw_quality_rmse, floor) ** 2)
        train_mean = float(train_weights[raw_task.train_target_source_indices].mean())
        train_weights = np.clip(train_weights / max(train_mean, 1e-12), 0.1, 10.0).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    warm_started = time.perf_counter()
    if use_warm_start and formulation != "normalized_residual":
        raise ValueError("LASSO warm start is defined only for normalized_residual formulation.")
    warm_start = (
        _raw_lasso_warm_start(
            raw_history,
            raw_task,
            config,
            norm_stats,
            str(device),
            history_mean=query_history_mean,
            history_std=query_history_std,
        )
        if use_warm_start
        else None
    )
    warm_start_seconds = time.perf_counter() - warm_started
    patch_kwargs = {
        "raw_history": raw_history if support_query else None,
        "input_mean": query_history_mean if support_query else None,
        "input_std": query_history_std if support_query else None,
        "max_points_per_patch": 128 if support_query else 256,
    }
    train_patches = build_raw_point_patches(
        raw_task,
        config,
        norm_stats,
        raw_output,
        split_code=0,
        raw_point_weights=train_weights,
        **patch_kwargs,
    )
    val_patches = build_raw_point_patches(
        raw_task,
        config,
        norm_stats,
        raw_output,
        split_code=1,
        **patch_kwargs,
    )
    if support_query:
        anchor_weights = None if warm_start is None else torch.tensor(warm_start.value_weights, dtype=torch.float32)
        builder = SupportAwarePointQueryModel(
            input_channels=train_patches.inputs.shape[2],
            time_steps=raw_history.shape[1],
            patch_size=config.patch_size,
            anchor_weights=anchor_weights,
            anchor_bias=0.0 if warm_start is None else warm_start.bias,
            context_frames=min(16, raw_history.shape[1]),
            use_spatial_context=support_use_spatial_context,
            use_global_coordinates=support_use_global_coordinates,
            use_local_coordinates=support_use_local_coordinates,
        )
        model = builder.to(device)
    elif model_kind == "simvp_style_residual":
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
    if not support_query:
        model = builder.build().to(device)
    disabled_components = apply_hybrid_ablation(
        model,
        disable_recent_gate=disable_recent_gate,
        disable_spatial_correction=disable_spatial_correction,
    ) if (disable_recent_gate or disable_spatial_correction) else ()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3 if support_query else config.cnn_learning_rate,
        weight_decay=config.cnn_weight_decay,
    )

    def dataset(arrays: RawPointPatchArrays) -> TensorDataset:
        if support_query:
            if arrays.point_histories is None or arrays.global_coordinates is None:
                raise ValueError("Support-aware query arrays are missing point histories.")
            return TensorDataset(
                torch.tensor(arrays.inputs, dtype=torch.float32),
                torch.tensor(arrays.sample_coordinates, dtype=torch.float32),
                torch.tensor(arrays.point_histories, dtype=torch.float32),
                torch.tensor(arrays.global_coordinates, dtype=torch.float32),
                torch.tensor(arrays.residual_targets, dtype=torch.float32),
                torch.tensor(arrays.point_masks, dtype=torch.float32),
            )
        return TensorDataset(
            torch.tensor(arrays.inputs, dtype=torch.float32),
            torch.tensor(arrays.sample_coordinates, dtype=torch.float32),
            torch.tensor(arrays.residual_targets, dtype=torch.float32),
            torch.tensor(arrays.point_masks, dtype=torch.float32),
        )

    train_loader = DataLoader(dataset(train_patches), batch_size=config.patch_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset(val_patches), batch_size=config.patch_batch_size, shuffle=False, num_workers=0)
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0
    history_rows: list[dict[str, object]] = []
    training_started = time.perf_counter()
    for epoch in range(1, config.cnn_epochs + 1):
        epoch_started = time.perf_counter()
        if support_query:
            train_loss, gradient_norm = _run_support_query_epoch(model, train_loader, optimizer, device)
            val_loss, _ = _run_support_query_epoch(model, val_loader, None, device)
        else:
            train_loss, gradient_norm = _run_raw_epoch(model, train_loader, optimizer, device)
            val_loss, _ = _run_raw_epoch(model, val_loader, None, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_raw_point_loss": float(train_loss),
                "val_raw_point_loss": float(val_loss),
                "gradient_l2_mean": float(gradient_norm),
                "epoch_seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if val_loss < best_val - 1e-9:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= config.cnn_patience:
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Raw-point supervised model did not produce a checkpoint.")
    model.load_state_dict(best_state)
    inference_started = time.perf_counter()
    if support_query:
        prediction = _predict_support_query_full_map(
            model,
            raw_task,
            config,
            norm_stats,
            device,
            history_mean=query_history_mean,
            history_std=query_history_std,
        )
        direct_payload = _predict_support_query_raw_points(
            model,
            raw_task,
            raw_history,
            config,
            norm_stats,
            raw_output,
            device,
            split_code=2,
            history_mean=query_history_mean,
            history_std=query_history_std,
        )
    else:
        prediction = _predict_full_map_for_formulation(
            model,
            raw_task,
            config,
            norm_stats,
            device,
            model_kind=model_kind,
            add_last_frame=add_last_frame,
        )
        direct_payload = None
    inference_seconds = time.perf_counter() - inference_started
    metrics: dict[str, object] = {
        "model": model_name,
        "model_kind": model_kind,
        **raw_point_metrics(prediction, raw_task, split="test"),
        **cell_aggregated_metrics(prediction, raw_task, split="test"),
        "target_supervision": "raw_observations_only",
        "interpolated_future_target_used_for_loss": False,
        "target_formulation": formulation,
        "optimization_target": (
            "standardized_future_increment"
            if support_query and formulation == "normalized_residual"
            else formulation
        ),
        "training_loss": (
            "smooth_l1_on_standardized_future_increment" if support_query else "smooth_l1"
        ),
        "anchor_coordinate_system": (
            "standardized_future_increment" if support_query and use_warm_start else None
        ),
        "physical_reconstruction": (
            "last_history_value + train_increment_mean + train_increment_std * standardized_prediction"
            if support_query
            else None
        ),
        "quality_weighted_raw_loss": bool(quality_weighted),
        "quality_weight_formula": "clip((1/max(rmse,p10_floor)^2)/train_mean,0.1,10)" if quality_weighted else None,
        **({} if direct_payload is None else direct_payload["metrics"]),
        "best_epoch": int(best_epoch),
        "best_val_raw_point_loss": float(best_val),
        "warm_start_enabled": bool(use_warm_start),
        "convlstm_num_layers": int(config.convlstm_num_layers) if model_kind == "cnn_lstm_hybrid" else None,
        "support_aware_query": bool(support_query),
        "support_context_frames": int(model.context_frames) if support_query else None,
        "support_history_encoder": "full_history_mlp" if support_query else None,
        "support_history_source": support_history_source if support_query else None,
        "spatial_context_enabled": bool(model.use_spatial_context) if support_query else None,
        "global_coordinate_conditioning": bool(model.use_global_coordinates) if support_query else None,
        "local_coordinate_conditioning": bool(model.use_local_coordinates) if support_query else None,
        "query_history_normalization": "per_time_train_mean_std" if support_query else None,
        "optimizer_learning_rate": float(1e-3 if support_query else config.cnn_learning_rate),
        "anchor_enabled": bool(model.anchor_enabled) if support_query else bool(use_warm_start),
        "anchor_preserving_initialization": (
            bool(model.anchor_preserving_initialization) if support_query else None
        ),
        "disabled_components": list(disabled_components),
        "warm_start_seconds": float(warm_start_seconds),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": trainable_parameter_count(model),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "train_patch_count": int(len(train_patches.inputs)),
        "val_patch_count": int(len(val_patches.inputs)),
        "train_raw_label_count": int(train_patches.raw_point_count),
        "val_raw_label_count": int(val_patches.raw_point_count),
        "normalization_input_mean": float(norm_stats.input_mean),
        "normalization_input_std": float(norm_stats.input_std),
        "normalization_residual_mean": float(norm_stats.residual_mean),
        "normalization_residual_std": float(norm_stats.residual_std),
        **raw_task.metadata,
    }
    torch.save(best_state, output_dir / "best_model.pth")
    if support_query:
        np.savez_compressed(
            output_dir / "query_history_normalization.npz",
            mean=np.asarray(query_history_mean, dtype=np.float32),
            std=np.asarray(query_history_std, dtype=np.float32),
        )
    np.save(output_dir / "prediction_grid.npy", prediction.astype(np.float32))
    test_prediction = sample_grid_at_raw_points(prediction, raw_task, raw_task.test_target_indices)
    np.savez_compressed(
        output_dir / "raw_test_predictions.npz",
        indices=raw_task.test_target_indices,
        points=raw_task.raw_points[raw_task.test_target_indices],
        truth=raw_task.raw_target[raw_task.test_target_indices],
        prediction=test_prediction,
        residual=test_prediction - raw_task.raw_target[raw_task.test_target_indices],
    )
    if direct_payload is not None:
        np.savez_compressed(
            output_dir / "direct_raw_test_predictions.npz",
            indices=direct_payload["indices"],
            points=raw_task.raw_points[direct_payload["indices"]],
            truth=direct_payload["truth"],
            prediction=direct_payload["prediction"],
            residual=direct_payload["prediction"] - direct_payload["truth"],
        )
    _write_rows(output_dir / "training_history.csv", history_rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics


def _direct_point_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    points: np.ndarray,
    raw_task: RawHoldoutTask,
) -> dict[str, object]:
    residual = prediction - truth
    east_index = np.rint(
        (points[:, 0] - raw_task.easting_axis[0])
        / (raw_task.easting_axis[-1] - raw_task.easting_axis[0])
        * (len(raw_task.easting_axis) - 1)
    ).astype(np.int64)
    north_index = np.rint(
        (points[:, 1] - raw_task.northing_axis[0])
        / (raw_task.northing_axis[-1] - raw_task.northing_axis[0])
        * (len(raw_task.northing_axis) - 1)
    ).astype(np.int64)
    east_index = np.clip(east_index, 0, len(raw_task.easting_axis) - 1)
    north_index = np.clip(north_index, 0, len(raw_task.northing_axis) - 1)
    cell_ids = east_index * len(raw_task.northing_axis) + north_index
    unique_cells = np.unique(cell_ids)
    cell_errors = []
    for cell_id in unique_cells:
        selected = cell_ids == cell_id
        cell_errors.append(float(prediction[selected].mean() - truth[selected].mean()))
    cell_errors_array = np.asarray(cell_errors, dtype=np.float64)
    return {
        "direct_raw_point_count": int(len(truth)),
        "direct_raw_rmse": float(np.sqrt(np.mean(residual**2))),
        "direct_raw_mae": float(np.mean(np.abs(residual))),
        "direct_raw_bias": float(np.mean(residual)),
        "direct_cell_count": int(len(unique_cells)),
        "direct_cell_mean_rmse_equal_cell": float(np.sqrt(np.mean(cell_errors_array**2))),
        "direct_cell_mean_mae_equal_cell": float(np.mean(np.abs(cell_errors_array))),
    }


def _predict_support_query_raw_points(
    model,
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    config: RevisionConfig,
    norm_stats: PatchNormStats,
    raw_output: np.ndarray,
    device,
    *,
    split_code: int,
    history_mean: np.ndarray,
    history_std: np.ndarray,
) -> dict[str, object]:
    import torch

    arrays = build_raw_point_patches(
        raw_task,
        config,
        norm_stats,
        raw_output,
        split_code=split_code,
        min_points_per_patch=1,
        max_points_per_patch=None,
        raw_history=raw_history,
        input_mean=history_mean,
        input_std=history_std,
    )
    if arrays.point_histories is None or arrays.point_indices is None or arrays.global_coordinates is None:
        raise RuntimeError("Direct support-query inference is missing point metadata.")
    predictions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arrays.inputs), max(1, config.patch_batch_size // 2)):
            end = min(start + max(1, config.patch_batch_size // 2), len(arrays.inputs))
            inputs = torch.tensor(arrays.inputs[start:end], dtype=torch.float32, device=device)
            coordinates = torch.tensor(arrays.sample_coordinates[start:end], dtype=torch.float32, device=device)
            histories = torch.tensor(arrays.point_histories[start:end], dtype=torch.float32, device=device)
            global_coordinates = torch.tensor(
                arrays.global_coordinates[start:end], dtype=torch.float32, device=device
            )
            normalized = model(inputs, coordinates, histories, global_coordinates).detach().cpu().numpy()
            mask = arrays.point_masks[start:end] > 0
            for local in range(end - start):
                predictions.append(normalized[local, mask[local]])
                indices.append(arrays.point_indices[start + local, mask[local]])
    normalized_prediction = np.concatenate(predictions).astype(np.float32)
    raw_indices = np.concatenate(indices).astype(np.int64)
    order = np.argsort(raw_indices)
    raw_indices = raw_indices[order]
    normalized_prediction = normalized_prediction[order]
    expected = np.flatnonzero(raw_task.raw_split_codes == split_code)
    if not np.array_equal(raw_indices, expected):
        raise AssertionError("Direct support-query inference did not cover each split point exactly once.")
    absolute_prediction = (
        normalized_prediction * norm_stats.residual_std
        + norm_stats.residual_mean
        + raw_history[raw_indices, -1]
    ).astype(np.float32)
    truth = raw_task.raw_target[raw_indices].astype(np.float32)
    metrics = _direct_point_metrics(
        absolute_prediction,
        truth,
        raw_task.raw_points[raw_indices],
        raw_task,
    )
    return {
        "indices": raw_indices,
        "truth": truth,
        "prediction": absolute_prediction,
        "metrics": metrics,
    }


def _predict_support_query_full_map(
    model,
    raw_task: RawHoldoutTask,
    config: RevisionConfig,
    norm_stats: PatchNormStats,
    device,
    *,
    history_mean: np.ndarray,
    history_std: np.ndarray,
) -> np.ndarray:
    import torch

    patches = _build_patch_arrays(
        task=raw_task.dense_task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=None,
        use_input_mask=True,
    )
    inputs = torch.tensor(patches.inputs, dtype=torch.float32, device=device)
    axis = torch.linspace(-1.0, 1.0, steps=config.patch_size, device=device)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
    local_rows = torch.arange(config.patch_size, device=device, dtype=torch.float32)
    local_cols = torch.arange(config.patch_size, device=device, dtype=torch.float32)
    local_row_grid, local_col_grid = torch.meshgrid(local_rows, local_cols, indexing="ij")
    height, width = raw_task.dense_task.target_map.shape
    prediction_sum = np.zeros(raw_task.dense_task.target_map.shape, dtype=np.float64)
    prediction_weight = np.zeros_like(prediction_sum)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), config.patch_batch_size):
            end = min(start + config.patch_batch_size, len(inputs))
            batch_inputs = inputs[start:end]
            batch_size = end - start
            batch_coordinates = coordinates.expand(batch_size, -1, -1)
            global_coordinate_patches = []
            for row, col in patches.positions[start:end]:
                global_x = 2.0 * (float(col) + local_col_grid) / max(width - 1, 1) - 1.0
                global_y = 2.0 * (float(row) + local_row_grid) / max(height - 1, 1) - 1.0
                global_coordinate_patches.append(
                    torch.stack([global_x, global_y], dim=-1).reshape(-1, 2)
                )
            batch_global_coordinates = torch.stack(global_coordinate_patches, dim=0)
            query_history_patches = []
            for row, col in patches.positions[start:end]:
                raw_patch = raw_task.dense_task.input_maps[
                    :,
                    int(row) : int(row) + config.patch_size,
                    int(col) : int(col) + config.patch_size,
                ]
                normalized_history = (
                    raw_patch.transpose(1, 2, 0) - np.asarray(history_mean, dtype=np.float32)
                ) / np.asarray(history_std, dtype=np.float32)
                query_history_patches.append(
                    torch.tensor(
                        normalized_history.reshape(config.patch_size * config.patch_size, -1),
                        dtype=torch.float32,
                        device=device,
                    )
                )
            query_history = torch.stack(query_history_patches, dim=0)
            normalized = model(
                batch_inputs,
                batch_coordinates,
                query_history,
                batch_global_coordinates,
            ).detach().cpu().numpy()
            residual = normalized.reshape(batch_size, config.patch_size, config.patch_size)
            residual = residual * norm_stats.residual_std + norm_stats.residual_mean
            output = residual + patches.last_frames[start:end, 0]
            for local, patch in enumerate(output):
                row, col = patches.positions[start + local]
                row_slice = slice(int(row), int(row) + config.patch_size)
                col_slice = slice(int(col), int(col) + config.patch_size)
                prediction_sum[row_slice, col_slice] += patch
                prediction_weight[row_slice, col_slice] += 1.0
    return (prediction_sum / np.clip(prediction_weight, 1.0, None)).astype(np.float32)


def run_raw_supervised_lasso(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    config: RevisionConfig,
    output_dir: Path,
) -> dict[str, object]:
    """Fit LASSO on raw observations, then apply it to the dense input grid."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(config.split_seed)
    train = raw_task.train_target_source_indices
    val = raw_task.val_target_source_indices
    test = raw_task.test_target_indices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_state = None
    best_alpha = None
    best_val = float("inf")
    best_epoch = 0
    alpha_rows: list[dict[str, object]] = []
    training_started = time.perf_counter()
    for alpha in sorted({1e-4, 5e-4, config.lasso_alpha, 5e-3, 1e-2}):
        state, val_rmse, epoch = _fit_torch_l1_regressor(
            X_train=raw_history[train],
            y_train=raw_task.raw_target[train],
            X_val=raw_history[val],
            y_val=raw_task.raw_target[val],
            alpha=alpha,
            config=config,
            device_str=device,
        )
        alpha_rows.append({"alpha": float(alpha), "val_raw_rmse": float(val_rmse), "best_epoch": int(epoch)})
        if val_rmse < best_val:
            best_state, best_alpha, best_val, best_epoch = state, alpha, val_rmse, epoch
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Raw-supervised LASSO did not produce a state.")

    direct_prediction = _predict_torch_l1_regressor(raw_history[test], best_state, device_str=device)
    flat_inputs = raw_task.dense_task.input_maps.reshape(raw_task.dense_task.input_maps.shape[0], -1).T
    inference_started = time.perf_counter()
    grid_prediction = _predict_torch_l1_regressor(flat_inputs, best_state, device_str=device).reshape(
        raw_task.dense_task.target_map.shape
    )
    inference_seconds = time.perf_counter() - inference_started
    metrics: dict[str, object] = {
        "model": "lasso_raw_supervised",
        **raw_point_metrics(grid_prediction, raw_task, split="test"),
        **cell_aggregated_metrics(grid_prediction, raw_task, split="test"),
        **_direct_point_metrics(
            direct_prediction.astype(np.float32),
            raw_task.raw_target[test].astype(np.float32),
            raw_task.raw_points[test],
            raw_task,
        ),
        "target_supervision": "raw_observations_only",
        "interpolated_future_target_used_for_loss": False,
        "best_alpha": float(best_alpha),
        "best_val_raw_rmse": float(best_val),
        "best_epoch": int(best_epoch),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "parameter_count": int(raw_history.shape[1] + 1),
        "device": device,
        **raw_task.metadata,
    }
    torch.save(best_state, output_dir / "lasso_state.pth")
    np.save(output_dir / "prediction_grid.npy", grid_prediction.astype(np.float32))
    np.savez_compressed(
        output_dir / "direct_raw_test_predictions.npz",
        indices=test.astype(np.int64),
        points=raw_task.raw_points[test].astype(np.float64),
        truth=raw_task.raw_target[test].astype(np.float32),
        prediction=direct_prediction.astype(np.float32),
        residual=(direct_prediction - raw_task.raw_target[test]).astype(np.float32),
    )
    _write_rows(output_dir / "alpha_sweep.csv", alpha_rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics


def _predict_full_map_for_formulation(
    model,
    raw_task: RawHoldoutTask,
    config: RevisionConfig,
    norm_stats: PatchNormStats,
    device,
    *,
    model_kind: str,
    add_last_frame: bool,
) -> np.ndarray:
    import torch

    patches = _build_patch_arrays(
        task=raw_task.dense_task,
        config=config,
        norm_stats=norm_stats,
        supervision_mask=None,
        use_input_mask=True,
    )
    if model_kind == "simvp_style_residual":
        input_array = patches.inputs.astype(np.float32)
    else:
        input_array = _adapt_patch_input_layout(patches.inputs, model_kind)
    inputs = torch.tensor(input_array, dtype=torch.float32, device=device)
    prediction_sum = np.zeros(raw_task.dense_task.target_map.shape, dtype=np.float64)
    prediction_weight = np.zeros_like(prediction_sum)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), config.patch_batch_size):
            end = min(start + config.patch_batch_size, len(inputs))
            normalized = model(inputs[start:end]).detach().cpu().numpy()[:, 0]
            output = normalized * norm_stats.residual_std + norm_stats.residual_mean
            if add_last_frame:
                output = output + patches.last_frames[start:end, 0]
            for local, patch in enumerate(output):
                row, col = patches.positions[start + local]
                row_slice = slice(int(row), int(row) + config.patch_size)
                col_slice = slice(int(col), int(col) + config.patch_size)
                prediction_sum[row_slice, col_slice] += patch
                prediction_weight[row_slice, col_slice] += 1.0
    return (prediction_sum / np.clip(prediction_weight, 1.0, None)).astype(np.float32)
