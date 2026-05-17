"""
Revision-aligned experiment implementations.

Revision skeleton alignment:
- Section 3.2 / task definition and split protocol
- Section 3.3 / models and baselines
- Section 3.4 / training protocol and hyperparameters
- Section 3.6 / evaluation metrics and comparative diagnostics
- Section 3.11 / SHAP analysis for LightGBM

These functions intentionally keep the project small and only cover the model
families that are actually available in this standalone code bundle.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from revision_config import RevisionConfig
from revision_utils import (
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

_LIGHTGBM_DEVICE_SUPPORT_CACHE: Dict[str, bool] = {}


def _model_output_dir(config: RevisionConfig, model_name: str, interpolation_method: str) -> Path:
    return ensure_dir(config.output_dir(model_name, interpolation_method) / f"split_seed_{config.split_seed}")


def _write_history_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _reconstruct_prediction_map(
    base_map: np.ndarray,
    eligible_indices: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    pred_map = np.full(base_map.shape, np.nan, dtype=np.float32)
    pred_map.reshape(-1)[eligible_indices] = predictions.astype(np.float32)
    return pred_map


def _package_metrics(
    model_name: str,
    interpolation_method: str,
    metrics: Dict[str, float],
    task,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": model_name,
        "interpolation_method": interpolation_method,
        "csv_path": str(task.csv_path),
        "grid_size": task.target_map.shape[0],
        "history_length": int(task.input_maps.shape[0]),
        "eligible_pixels": int(task.eligible_mask.sum()),
        "train_pixels": int(task.train_mask.sum()),
        "val_pixels": int(task.val_mask.sum()),
        "test_pixels": int(task.test_mask.sum()),
    }
    payload.update(metrics)
    return payload


class SimpleCNNLSTM:
    def __init__(self, hidden_dim: int, output_size: tuple[int, int]) -> None:
        import torch.nn as nn

        h, w = output_size
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError("The CNN-LSTM encoder expects spatial dimensions divisible by 8.")

        class _Model(nn.Module):
            def __init__(self, hidden_dim_: int, output_size_: tuple[int, int]) -> None:
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
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

    def build(self):
        return self.model_class(self.hidden_dim, self.output_size)


class GoodModelAlignedCNNLSTM(SimpleCNNLSTM):
    """
    Exact structural mirror of the legacy `goodmodel.py` CNN-LSTM stack.

    Revision skeleton alignment:
    - Section 2.1 / verify whether `goodmodel.py` is genuinely a different model
    - Section 5.2 / treat `goodmodel.py` as backup unless the aligned protocol proves otherwise
    """

    pass


def _default_torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _fit_torch_l1_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    config: RevisionConfig,
    device_str: str,
) -> tuple[Dict[str, object], float, int]:
    """
    Supports revision skeleton Sections 3.3 and 3.4 by training an
    L1-regularized linear regressor on the same split, while using CUDA
    whenever PyTorch can access it.
    """

    import torch

    device = torch.device(device_str)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    X_mean = X_train_t.mean(dim=0, keepdim=True)
    X_std = X_train_t.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train_t.mean()
    y_std = y_train_t.std().clamp_min(1e-6)

    X_train_n = (X_train_t - X_mean) / X_std
    X_val_n = (X_val_t - X_mean) / X_std
    y_train_n = (y_train_t - y_mean) / y_std

    weights = torch.zeros(X_train_n.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=config.lasso_learning_rate)

    best_state: Dict[str, object] | None = None
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.lasso_epochs + 1):
        optimizer.zero_grad()
        train_pred_n = X_train_n @ weights + bias
        mse_loss = torch.mean((train_pred_n - y_train_n) ** 2)
        l1_penalty = torch.mean(torch.abs(weights))
        loss = mse_loss + alpha * l1_penalty
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_pred_n = X_val_n @ weights + bias
            val_pred = val_pred_n * y_std + y_mean
            val_rmse = float(torch.sqrt(torch.mean((val_pred - y_val_t) ** 2)).item())

        if val_rmse < best_val_rmse - 1e-9:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                "weights": weights.detach().cpu().clone(),
                "bias": bias.detach().cpu().clone(),
                "X_mean": X_mean.detach().cpu().clone(),
                "X_std": X_std.detach().cpu().clone(),
                "y_mean": y_mean.detach().cpu().clone(),
                "y_std": y_std.detach().cpu().clone(),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.lasso_patience:
            break

    if best_state is None:
        raise RuntimeError("Torch L1 baseline failed to produce a valid checkpoint.")

    return best_state, best_val_rmse, best_epoch


def _predict_torch_l1_regressor(
    X_values: np.ndarray,
    state: Dict[str, object],
    device_str: str,
) -> np.ndarray:
    import torch

    device = torch.device(device_str)
    X_t = torch.tensor(X_values, dtype=torch.float32, device=device)
    weights = state["weights"].to(device)
    bias = state["bias"].to(device)
    X_mean = state["X_mean"].to(device)
    X_std = state["X_std"].to(device)
    y_mean = state["y_mean"].to(device)
    y_std = state["y_std"].to(device)

    with torch.no_grad():
        X_n = (X_t - X_mean) / X_std
        pred_n = X_n @ weights + bias
        pred = pred_n * y_std + y_mean
    return pred.detach().cpu().numpy().astype(np.float32)


def _probe_lightgbm_device(device_type: str) -> bool:
    if device_type in _LIGHTGBM_DEVICE_SUPPORT_CACHE:
        return _LIGHTGBM_DEVICE_SUPPORT_CACHE[device_type]

    probe_cache_dir = Path(__file__).resolve().parent / "revision_outputs" / "_lightgbm_probe_cache"
    probe_cache_dir.mkdir(parents=True, exist_ok=True)
    probe_code = """
import lightgbm as lgb
import numpy as np

X_probe = np.random.default_rng(42).random((128, 8), dtype=np.float32)
y_probe = np.random.default_rng(123).random(128, dtype=np.float32)
dtrain = lgb.Dataset(X_probe, label=y_probe)
params = {
    "objective": "regression",
    "metric": "l2",
    "verbosity": -1,
    "num_leaves": 7,
    "min_data_in_leaf": 1,
    "max_bin": 63,
    "device_type": "%s",
}
lgb.train(params, dtrain, num_boost_round=2)
print("ok")
""" % device_type
    env = os.environ.copy()
    env["BOOST_COMPUTE_CACHE_PATH"] = str(probe_cache_dir)
    result = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    supported = result.returncode == 0 and "ok" in result.stdout

    _LIGHTGBM_DEVICE_SUPPORT_CACHE[device_type] = supported
    return supported


def _resolve_lightgbm_device_type(requested: str) -> str:
    normalized = requested.lower()
    if normalized == "auto":
        for candidate in ("cuda", "gpu"):
            if _probe_lightgbm_device(candidate):
                return candidate
        return "cpu"
    if normalized in {"cuda", "gpu"}:
        return normalized if _probe_lightgbm_device(normalized) else "cpu"
    return "cpu"


def _run_cnnlstm_family_experiment(
    config: RevisionConfig,
    model_name: str,
    model_builder_cls,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    """
    Supports revision skeleton Sections 3.2, 3.3, 3.4, and 3.6 by providing:
    - a fixed task definition
    - explicit train/val/test masks
    - a reproducible CNN-LSTM-family training loop
    - masked-domain diagnostics and map visualizations
    """

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
    model = model_builder_cls(
        hidden_dim=config.cnn_hidden_dim,
        output_size=task.target_map.shape,
    ).build().to(device)

    x_tensor = torch.tensor(task.input_maps[np.newaxis, :, np.newaxis, :, :], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(task.target_map[np.newaxis, np.newaxis, :, :], dtype=torch.float32, device=device)
    train_mask = torch.tensor(task.train_mask, dtype=torch.float32, device=device)
    val_mask = torch.tensor(task.val_mask, dtype=torch.float32, device=device)
    test_mask = torch.tensor(task.test_mask, dtype=torch.float32, device=device)

    def masked_mse_loss(pred, target, mask):
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        squared_error = (pred - target) ** 2
        return (squared_error * mask_4d).sum() / mask_4d.sum().clamp_min(1.0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.cnn_learning_rate,
        weight_decay=config.cnn_weight_decay,
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    history_rows: List[Dict[str, float]] = []
    patience_counter = 0

    for epoch in range(1, config.cnn_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_pred = model(x_tensor)
        train_loss = masked_mse_loss(train_pred, y_tensor, train_mask)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_tensor)
            val_loss = masked_mse_loss(val_pred, y_tensor, val_mask)

        row = {
            "epoch": float(epoch),
            "train_masked_mse": float(train_loss.item()),
            "val_masked_mse": float(val_loss.item()),
        }
        history_rows.append(row)

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
        raise RuntimeError("CNN-LSTM training failed to produce a best checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "best_cnnlstm_revision_model.pth")
    _write_history_csv(history_rows, output_dir / "training_history.csv")

    model.eval()
    with torch.no_grad():
        pred_tensor = model(x_tensor)

    pred_map = pred_tensor.squeeze(0).squeeze(0).cpu().numpy()
    target_map = task.target_map

    metrics = masked_regression_metrics(target_map, pred_map, task.test_mask)
    metrics.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_masked_mse": float(best_val),
            "device": str(device),
        }
    )
    payload = _package_metrics(model_name, task.interpolation_method, metrics, task)
    save_metrics(payload, output_dir)
    save_prediction_map(pred_map, output_dir)
    save_error_diagnostics(target_map, pred_map, task.test_mask, output_dir, n_bins=config.diag_bins)
    save_map_comparison(target_map, pred_map, task.target_valid_mask, output_dir)
    np.save(output_dir / "target_map.npy", target_map.astype(np.float32))
    return payload


def run_cnnlstm_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    return _run_cnnlstm_family_experiment(
        config=config,
        model_name="cnn_lstm",
        model_builder_cls=SimpleCNNLSTM,
        interpolation_method=interpolation_method,
    )


def run_goodmodel_aligned_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    """
    Supports the revision check that asks whether the legacy `goodmodel.py`
    architecture still beats the current aligned CNN-LSTM once both use the
    same split protocol and diagnostics.
    """

    return _run_cnnlstm_family_experiment(
        config=config,
        model_name="goodmodel_aligned",
        model_builder_cls=GoodModelAlignedCNNLSTM,
        interpolation_method=interpolation_method,
    )


def run_lasso_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
) -> Dict[str, object]:
    """
    Supports revision skeleton Sections 3.3, 3.4, and 3.6 by aligning LASSO to
    the same masked pixel split used by the CNN-LSTM run.
    """

    device_str = _default_torch_device()
    task = build_dense_forecast_task(config, interpolation_method)
    output_dir = _model_output_dir(config, "lasso", task.interpolation_method)
    save_config_snapshot(config, output_dir)
    save_split_bundle(task, output_dir)

    X_all, y_all, eligible_indices = build_tabular_dataset(task)
    split_positions = split_from_eligible_indices(task, eligible_indices)

    X_train = X_all[split_positions["train"]]
    y_train = y_all[split_positions["train"]]
    X_val = X_all[split_positions["val"]]
    y_val = y_all[split_positions["val"]]

    candidate_alphas = sorted({1e-4, 5e-4, config.lasso_alpha, 5e-3, 1e-2})
    alpha_rows: List[Dict[str, float]] = []
    best_state: Dict[str, object] | None = None
    best_alpha = None
    best_val_rmse = float("inf")
    best_epoch = 0

    for alpha in candidate_alphas:
        state, val_rmse, alpha_best_epoch = _fit_torch_l1_regressor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            alpha=alpha,
            config=config,
            device_str=device_str,
        )
        alpha_rows.append(
            {
                "alpha": float(alpha),
                "val_rmse": val_rmse,
                "best_epoch": float(alpha_best_epoch),
            }
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_alpha = alpha
            best_state = state
            best_epoch = alpha_best_epoch

    if best_state is None or best_alpha is None:
        raise RuntimeError("LASSO validation sweep failed.")

    _write_history_csv(alpha_rows, output_dir / "alpha_sweep.csv")
    full_predictions = _predict_torch_l1_regressor(X_all, best_state, device_str=device_str)
    pred_map = _reconstruct_prediction_map(task.target_map, eligible_indices, full_predictions)

    metrics = masked_regression_metrics(task.target_map, pred_map, task.test_mask)
    metrics.update(
        {
            "best_alpha": float(best_alpha),
            "best_val_rmse": float(best_val_rmse),
            "best_epoch": int(best_epoch),
            "device": device_str,
        }
    )
    payload = _package_metrics("lasso", task.interpolation_method, metrics, task)
    save_metrics(payload, output_dir)
    save_prediction_map(pred_map, output_dir)
    save_error_diagnostics(task.target_map, pred_map, task.test_mask, output_dir, n_bins=config.diag_bins)
    save_map_comparison(task.target_map, pred_map, task.target_valid_mask, output_dir)
    np.save(output_dir / "target_map.npy", task.target_map.astype(np.float32))
    return payload


def run_lightgbm_experiment(
    config: RevisionConfig,
    interpolation_method: str | None = None,
    enable_shap: bool = True,
) -> Dict[str, object]:
    """
    Supports revision skeleton Sections 3.3, 3.4, 3.6, and 3.11 by aligning
    LightGBM to the same masked pixel split and exporting SHAP diagnostics.
    """

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required to run the revision-aligned tree baseline.") from exc

    task = build_dense_forecast_task(config, interpolation_method)
    output_dir = _model_output_dir(config, "lightgbm", task.interpolation_method)
    save_config_snapshot(config, output_dir)
    save_split_bundle(task, output_dir)

    X_all, y_all, eligible_indices = build_tabular_dataset(task)
    split_positions = split_from_eligible_indices(task, eligible_indices)

    X_train = X_all[split_positions["train"]]
    y_train = y_all[split_positions["train"]]
    X_val = X_all[split_positions["val"]]
    y_val = y_all[split_positions["val"]]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    resolved_device_type = _resolve_lightgbm_device_type(config.lightgbm_device_type)
    if resolved_device_type in {"gpu", "cuda"}:
        gpu_cache_dir = config.output_root / "_lightgbm_boost_compute"
        ensure_dir(gpu_cache_dir)
        os.environ["BOOST_COMPUTE_CACHE_PATH"] = str(gpu_cache_dir)

    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": config.lightgbm_learning_rate,
        "num_leaves": config.lightgbm_num_leaves,
        "feature_fraction": config.lightgbm_feature_fraction,
        "bagging_fraction": config.lightgbm_bagging_fraction,
        "bagging_freq": config.lightgbm_bagging_freq,
        "seed": config.split_seed,
        "verbosity": -1,
        "device_type": resolved_device_type,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=config.lightgbm_num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(config.lightgbm_early_stopping_rounds),
            lgb.log_evaluation(period=50),
        ],
    )

    full_predictions = model.predict(X_all, num_iteration=model.best_iteration)
    pred_map = _reconstruct_prediction_map(task.target_map, eligible_indices, full_predictions)

    metrics = masked_regression_metrics(task.target_map, pred_map, task.test_mask)
    metrics.update(
        {
            "best_iteration": int(model.best_iteration),
            "device_type": resolved_device_type,
        }
    )
    payload = _package_metrics("lightgbm", task.interpolation_method, metrics, task)
    save_metrics(payload, output_dir)
    save_prediction_map(pred_map, output_dir)
    save_error_diagnostics(task.target_map, pred_map, task.test_mask, output_dir, n_bins=config.diag_bins)
    save_map_comparison(task.target_map, pred_map, task.target_valid_mask, output_dir)
    np.save(output_dir / "target_map.npy", task.target_map.astype(np.float32))
    model.save_model(str(output_dir / "lightgbm_model.txt"))

    if enable_shap:
        try:
            import shap
            import matplotlib.pyplot as plt

            sample_size = min(2000, len(X_val))
            if sample_size > 0:
                X_shap = X_val[:sample_size]
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)
                feature_names = [f"lag_{idx + 1}" for idx in range(X_shap.shape[1])]

                shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(output_dir / "shap_summary.png", dpi=150)
                plt.close()

                try:
                    force_plot = shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        X_shap[0],
                        feature_names=feature_names,
                    )
                    shap.save_html(str(output_dir / "shap_force_sample0.html"), force_plot)
                except Exception:
                    pass
        except ImportError:
            pass

    return payload
