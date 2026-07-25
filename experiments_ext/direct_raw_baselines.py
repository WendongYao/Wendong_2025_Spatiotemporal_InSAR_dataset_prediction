"""Direct raw-history baselines for the CAGEO 300-to-one task."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Literal

import numpy as np

from raw_holdout_data import RawHoldoutTask, cell_aggregated_metrics, raw_point_metrics
from raw_point_supervision import _direct_point_metrics


TargetFormulation = Literal["auto", "absolute", "increment"]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dense_histories(raw_task: RawHoldoutTask) -> np.ndarray:
    maps = raw_task.dense_task.input_maps
    return maps.reshape(maps.shape[0], -1).T.astype(np.float32, copy=False)


def _base_metrics(
    *,
    model_name: str,
    direct_prediction: np.ndarray,
    grid_prediction: np.ndarray,
    raw_task: RawHoldoutTask,
) -> dict[str, object]:
    test = raw_task.test_target_indices
    return {
        "model": model_name,
        **raw_point_metrics(grid_prediction, raw_task, split="test"),
        **cell_aggregated_metrics(grid_prediction, raw_task, split="test"),
        **_direct_point_metrics(
            direct_prediction.astype(np.float32),
            raw_task.raw_target[test].astype(np.float32),
            raw_task.raw_points[test],
            raw_task,
        ),
        "primary_endpoint": "direct_raw_observation",
        "secondary_endpoint": "idw_input_history_dense_query",
        "target_supervision": "raw_observations_only",
        "interpolated_future_target_used_for_loss": False,
        **raw_task.metadata,
    }


def _save_predictions(
    *,
    output_dir: Path,
    raw_task: RawHoldoutTask,
    direct_prediction: np.ndarray,
    grid_prediction: np.ndarray,
) -> None:
    test = raw_task.test_target_indices
    np.save(output_dir / "prediction_grid.npy", grid_prediction.astype(np.float32))
    np.savez_compressed(
        output_dir / "direct_raw_test_predictions.npz",
        indices=test,
        points=raw_task.raw_points[test],
        truth=raw_task.raw_target[test],
        prediction=direct_prediction.astype(np.float32),
        residual=direct_prediction.astype(np.float32) - raw_task.raw_target[test],
    )


def _lightgbm_targets(
    raw_history: np.ndarray,
    raw_target: np.ndarray,
    formulation: Literal["absolute", "increment"],
) -> np.ndarray:
    if formulation == "absolute":
        return raw_target.astype(np.float32)
    return (raw_target - raw_history[:, -1]).astype(np.float32)


def _restore_lightgbm_prediction(
    prediction: np.ndarray,
    history: np.ndarray,
    formulation: Literal["absolute", "increment"],
) -> np.ndarray:
    restored = prediction.astype(np.float32)
    if formulation == "increment":
        restored = restored + history[:, -1].astype(np.float32)
    return restored


def run_direct_raw_lightgbm(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
    target_formulation: TargetFormulation = "increment",
) -> dict[str, object]:
    """Fit deterministic LightGBM directly on raw 300-step histories."""
    import lightgbm as lgb

    output_dir.mkdir(parents=True, exist_ok=True)
    train = raw_task.train_target_source_indices
    val = raw_task.val_target_source_indices
    test = raw_task.test_target_indices
    candidates: tuple[Literal["absolute", "increment"], ...] = (
        ("absolute", "increment") if target_formulation == "auto" else (target_formulation,)
    )
    candidate_rows: list[dict[str, object]] = []
    fitted: dict[str, tuple[object, float]] = {}
    selection_started = time.perf_counter()
    for formulation in candidates:
        y = _lightgbm_targets(raw_history, raw_task.raw_target, formulation)
        model = lgb.LGBMRegressor(
            objective="regression",
            metric="l2",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        fit_started = time.perf_counter()
        model.fit(
            raw_history[train],
            y[train],
            eval_set=[(raw_history[val], y[val])],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        fit_seconds = time.perf_counter() - fit_started
        val_prediction = _restore_lightgbm_prediction(
            model.predict(raw_history[val], num_iteration=model.best_iteration_),
            raw_history[val],
            formulation,
        )
        val_rmse = float(
            np.sqrt(np.mean((val_prediction - raw_task.raw_target[val].astype(np.float32)) ** 2))
        )
        candidate_rows.append(
            {
                "target_formulation": formulation,
                "val_direct_raw_rmse": val_rmse,
                "best_iteration": int(model.best_iteration_),
                "training_seconds": float(fit_seconds),
            }
        )
        fitted[formulation] = (model, fit_seconds)
    selection_seconds = time.perf_counter() - selection_started
    selected = min(candidate_rows, key=lambda row: float(row["val_direct_raw_rmse"]))
    selected_formulation = str(selected["target_formulation"])
    model, selected_training_seconds = fitted[selected_formulation]

    inference_started = time.perf_counter()
    direct_prediction = _restore_lightgbm_prediction(
        model.predict(raw_history[test], num_iteration=model.best_iteration_),
        raw_history[test],
        selected_formulation,
    )
    dense_history = _dense_histories(raw_task)
    dense_prediction = _restore_lightgbm_prediction(
        model.predict(dense_history, num_iteration=model.best_iteration_),
        dense_history,
        selected_formulation,
    )
    grid_prediction = dense_prediction.reshape(raw_task.dense_task.target_map.shape)
    inference_seconds = time.perf_counter() - inference_started

    dump = model.booster_.dump_model()
    leaf_count = int(sum(int(tree["num_leaves"]) for tree in dump["tree_info"]))
    metrics = {
        **_base_metrics(
            model_name="direct_raw_lightgbm",
            direct_prediction=direct_prediction,
            grid_prediction=grid_prediction,
            raw_task=raw_task,
        ),
        "target_formulation": selected_formulation,
        "optimization_target": (
            "absolute_future_observation"
            if selected_formulation == "absolute"
            else "future_increment"
        ),
        "physical_reconstruction": (
            "direct_model_output"
            if selected_formulation == "absolute"
            else "last_history_value + predicted_increment"
        ),
        "target_formulation_selection": (
            "seed42_validation_only" if target_formulation == "auto" else "frozen_cli_choice"
        ),
        "candidate_validation_results": candidate_rows,
        "best_val_raw_rmse": float(selected["val_direct_raw_rmse"]),
        "best_iteration": int(model.best_iteration_),
        "training_seconds": float(selected_training_seconds),
        "selection_training_seconds_total": float(selection_seconds),
        "inference_seconds": float(inference_seconds),
        "tree_count": int(model.booster_.num_trees()),
        "leaf_count_total": leaf_count,
        "parameter_count": None,
        "seed": int(seed),
        "device": "cpu",
        "configuration": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "early_stopping_rounds": 50,
            "deterministic": True,
        },
    }
    model.booster_.save_model(str(output_dir / "lightgbm_model.txt"))
    _write_rows(output_dir / "target_formulation_selection.csv", candidate_rows)
    _save_predictions(
        output_dir=output_dir,
        raw_task=raw_task,
        direct_prediction=direct_prediction,
        grid_prediction=grid_prediction,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics

class PointwiseGRU:
    """Namespace wrapper so torch is imported only when the baseline is run."""

    @staticmethod
    def build(hidden_size: int):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, history):
                sequence, _ = self.gru(history.unsqueeze(-1))
                return self.head(sequence[:, -1]).squeeze(-1)

        return _Model()


def run_pointwise_gru(
    raw_task: RawHoldoutTask,
    raw_history: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
    epochs: int = 60,
    patience: int = 12,
    batch_size: int = 1024,
    hidden_size: int = 32,
) -> dict[str, object]:
    """Fit the frozen lightweight one-layer GRU on standardized increments."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train = raw_task.train_target_source_indices
    val = raw_task.val_target_source_indices
    test = raw_task.test_target_indices
    history_mean = raw_history[train].mean(axis=0).astype(np.float32)
    history_std = np.maximum(raw_history[train].std(axis=0), 1e-6).astype(np.float32)
    increment = (raw_task.raw_target - raw_history[:, -1]).astype(np.float32)
    increment_mean = float(increment[train].mean())
    increment_std = float(max(increment[train].std(), 1e-6))
    normalized_history = ((raw_history - history_mean) / history_std).astype(np.float32)
    normalized_increment = ((increment - increment_mean) / increment_std).astype(np.float32)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_history[train]),
            torch.from_numpy(normalized_increment[train]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_history[val]),
            torch.from_numpy(normalized_increment[val]),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = PointwiseGRU.build(hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = None
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    best_epoch = -1
    stale = 0
    history_rows: list[dict[str, object]] = []
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for histories, targets in train_loader:
            histories = histories.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(histories)
            loss = F.smooth_l1_loss(prediction, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(histories)
            train_count += len(histories)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for histories, targets in val_loader:
                histories = histories.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                prediction = model(histories)
                val_loss_sum += float(F.smooth_l1_loss(prediction, targets).cpu()) * len(histories)
                val_count += len(histories)
                val_predictions.append(prediction.cpu().numpy())
        val_loss = val_loss_sum / max(val_count, 1)
        val_normalized = np.concatenate(val_predictions).astype(np.float32)
        val_absolute = (
            raw_history[val, -1] + increment_mean + increment_std * val_normalized
        ).astype(np.float32)
        val_rmse = float(
            np.sqrt(np.mean((val_absolute - raw_task.raw_target[val].astype(np.float32)) ** 2))
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_standardized_smooth_l1": train_loss_sum / max(train_count, 1),
                "val_standardized_smooth_l1": val_loss,
                "val_direct_raw_rmse": val_rmse,
                "epoch_seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if val_loss < best_val_loss - 1e-9:
            best_val_loss = float(val_loss)
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Pointwise GRU did not produce a checkpoint.")
    model.load_state_dict(best_state)

    def predict(histories: np.ndarray) -> np.ndarray:
        normalized = ((histories - history_mean) / history_std).astype(np.float32)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(normalized)),
            batch_size=max(batch_size, 2048),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        outputs: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                outputs.append(model(batch.to(device, non_blocking=True)).cpu().numpy())
        standardized = np.concatenate(outputs).astype(np.float32)
        return (histories[:, -1] + increment_mean + increment_std * standardized).astype(np.float32)

    inference_started = time.perf_counter()
    direct_prediction = predict(raw_history[test])
    dense_history = _dense_histories(raw_task)
    grid_prediction = predict(dense_history).reshape(raw_task.dense_task.target_map.shape)
    inference_seconds = time.perf_counter() - inference_started
    metrics = {
        **_base_metrics(
            model_name="pointwise_gru",
            direct_prediction=direct_prediction,
            grid_prediction=grid_prediction,
            raw_task=raw_task,
        ),
        "target_formulation": "standardized_future_increment",
        "optimization_target": "standardized_future_increment",
        "training_loss": "smooth_l1_on_standardized_future_increment",
        "physical_reconstruction": (
            "last_history_value + train_increment_mean + train_increment_std * standardized_prediction"
        ),
        "history_normalization": "per_time_train_mean_std",
        "best_epoch": int(best_epoch),
        "best_val_standardized_smooth_l1": float(best_val_loss),
        "best_val_raw_rmse": float(best_val_rmse),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
        ),
        "seed": int(seed),
        "device": str(device),
        "configuration": {
            "num_layers": 1,
            "hidden_size": int(hidden_size),
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "patience": int(patience),
        },
    }
    torch.save(
        {
            "model_state": best_state,
            "history_mean": history_mean,
            "history_std": history_std,
            "increment_mean": increment_mean,
            "increment_std": increment_std,
            "configuration": metrics["configuration"],
        },
        output_dir / "best_model.pth",
    )
    np.savez_compressed(
        output_dir / "normalization.npz",
        history_mean=history_mean,
        history_std=history_std,
        increment_mean=np.float32(increment_mean),
        increment_std=np.float32(increment_std),
    )
    _write_rows(output_dir / "training_history.csv", history_rows)
    _save_predictions(
        output_dir=output_dir,
        raw_task=raw_task,
        direct_prediction=direct_prediction,
        grid_prediction=grid_prediction,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics
