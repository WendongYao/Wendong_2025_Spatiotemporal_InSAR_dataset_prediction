"""Run the final all-cell SPAR architecture with its fixed LASSO anchor removed."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from revision_config import RevisionConfig  # noqa: E402

from native_pointwise_v23 import (  # noqa: E402
    DirectSPARRegressor,
    _native_metrics,
    _save_history,
    _weighted_mean,
    sampler_indices_and_weights,
    set_deterministic_seed,
)
from native_support_baselines import load_forecast_dates  # noqa: E402
from raw_holdout_data import (  # noqa: E402
    RawHoldoutSpec,
    build_raw_holdout_task,
    load_forecast_columns,
)
from raw_point_supervision import PatchNormStats  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT / "source",
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_no_anchor(
    raw_task,
    raw_history: np.ndarray,
    config,
    output_dir: Path,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_all = raw_task.train_target_source_indices
    test = raw_task.test_target_indices
    history_mean = raw_history[train_all].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    history_std = np.maximum(
        raw_history[train_all].std(axis=0, dtype=np.float64),
        1e-6,
    ).astype(np.float32)
    raw_increment = (
        raw_task.raw_target.astype(np.float32)
        - raw_history[:, -1].astype(np.float32)
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
    train_indices, train_weights = sampler_indices_and_weights(
        raw_task,
        raw_history,
        config,
        norm_stats,
        raw_increment,
        split_code=0,
        sampler="all_cells_uniform",
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
        sampler="all_cells_uniform",
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
        np.zeros(raw_history.shape[1], dtype=np.float32),
        0.0,
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
        raise RuntimeError("No-anchor ablation produced no validation checkpoint.")
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
    metrics = _native_metrics(
        model_name="direct_spar_all_cells_uniform_no_anchor",
        prediction=prediction,
        truth=truth,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
        extra={
            "sampler": "all_cells_uniform",
            "anchor": "disabled_zero_buffer",
            "anchor_enabled": False,
            "only_intervention_relative_to_locked_spar": (
                "fixed LASSO anchor weights and bias set to zero"
            ),
            "training_loss": "weighted Smooth-L1 on standardized future increment",
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
            "sampler": "all_cells_uniform",
            "anchor_enabled": False,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    code_files = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "experiments_ext" / "native_pointwise_v23.py",
        PROJECT_ROOT / "experiments_ext" / "raw_holdout_data.py",
    ]
    manifest: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "source_git_commit": source_commit(),
        "code_sha256_at_launch": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in code_files
        },
        "input": {
            "path": str(args.csv_path.resolve()),
            "size_bytes": args.csv_path.stat().st_size,
            "sha256": sha256(args.csv_path),
        },
        "tile": "E32N34",
        "seed": args.seed,
        "history_start_col": 11,
        "history_length": 300,
        "target_col": 312,
        "sampler": "all_cells_uniform",
        "anchor_enabled": False,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "protocol": "cageo-v23-final-no-anchor-ablation",
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    spec = RawHoldoutSpec(
        csv_path=args.csv_path,
        tile="E32N34",
        grid_size=256,
        history_start_col=11,
        history_length=300,
        target_col=312,
        split_seed=args.seed,
        block_side=8,
    )
    task = build_raw_holdout_task(spec, cache_dir=args.cache_dir)
    points, history, target = load_forecast_columns(spec)
    if not np.array_equal(points, task.raw_points):
        raise AssertionError("No-anchor point order differs from cached task.")
    if not np.array_equal(target, task.raw_target):
        raise AssertionError("No-anchor target differs from cached task.")
    _, _, date_metadata = load_forecast_dates(spec)
    (args.output_root / "task_metadata.json").write_text(
        json.dumps({**task.metadata, **date_metadata}, indent=2),
        encoding="utf-8",
    )
    config = RevisionConfig(
        csv_path=str(args.csv_path.resolve()),
        grid_size=256,
        split_seed=args.seed,
        split_strategy="spatial_tile",
        interpolation_method="idw",
        output_root=args.output_root,
        use_task_cache=False,
    )
    metrics = run_no_anchor(
        task,
        history,
        config,
        args.output_root / "spar_all_cells_uniform_no_anchor",
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    metrics.update({**date_metadata, "seed": args.seed})
    metrics_path = (
        args.output_root / "spar_all_cells_uniform_no_anchor" / "metrics.json"
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    manifest.update(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": float(time.perf_counter() - started),
            "output_sha256": {
                path.relative_to(args.output_root).as_posix(): sha256(path)
                for path in sorted(args.output_root.rglob("*"))
                if path.is_file() and path.name != "run_manifest.json"
            },
        }
    )
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "completed": "spar_all_cells_uniform_no_anchor",
                "seed": args.seed,
                "native_cell_rmse": metrics["native_cell_rmse"],
            }
        ),
        flush=True,
    )
if __name__ == "__main__":
    main()
