"""Evaluate frozen LASSO and SPAR models across output supports.

The square-grid endpoints use IDW-interpolated history values followed by the
frozen temporal predictor.  The native 100-m endpoint retains only valid EGMS
Level-3 product cells and introduces no additional interpolation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from raw_holdout_data import RawHoldoutSpec, build_raw_holdout_task, load_forecast_columns
from support_aware_model import SupportAwarePointQueryModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def idw_history_batch(
    tree: cKDTree,
    source_history: np.ndarray,
    queries: np.ndarray,
    *,
    neighbors: int,
    power: float,
    time_chunk: int = 32,
) -> np.ndarray:
    k = min(neighbors, len(source_history))
    distances, indices = tree.query(queries, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    exact = distances <= 1e-12
    weights = np.zeros_like(distances, dtype=np.float64)
    nonexact = ~np.any(exact, axis=1)
    weights[nonexact] = 1.0 / np.maximum(
        distances[nonexact],
        1e-12,
    ) ** float(power)
    if np.any(~nonexact):
        first_exact = np.argmax(exact[~nonexact], axis=1)
        rows = np.flatnonzero(~nonexact)
        weights[rows, first_exact] = 1.0
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    output = np.empty((len(queries), source_history.shape[1]), dtype=np.float32)
    for start in range(0, source_history.shape[1], time_chunk):
        end = min(start + time_chunk, source_history.shape[1])
        values = source_history[indices, start:end]
        output[:, start:end] = np.sum(
            values * weights[:, :, None],
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
    return output


class LassoPredictor:
    def __init__(self, state_path: Path, device: str) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        state = torch.load(state_path, map_location=self.device, weights_only=False)
        self.weights = state["weights"].to(self.device)
        self.bias = state["bias"].to(self.device)
        self.x_mean = state["X_mean"].to(self.device)
        self.x_std = state["X_std"].to(self.device)
        self.y_mean = state["y_mean"].to(self.device)
        self.y_std = state["y_std"].to(self.device)

    def __call__(self, history: np.ndarray) -> np.ndarray:
        values = self.torch.from_numpy(history).to(self.device)
        with self.torch.no_grad():
            normalized = (values - self.x_mean) / self.x_std
            prediction = (
                (normalized @ self.weights + self.bias) * self.y_std + self.y_mean
            )
        return prediction.detach().cpu().numpy().astype(np.float32)


class SparPredictor:
    def __init__(self, model_dir: Path, device: str) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        state = torch.load(
            model_dir / "best_model.pth",
            map_location=self.device,
            weights_only=False,
        )
        self.model = SupportAwarePointQueryModel(
            input_channels=1,
            time_steps=300,
            patch_size=16,
            anchor_weights=state["anchor_weights"].detach().cpu(),
            anchor_bias=float(state["anchor_bias"].detach().cpu()),
            context_frames=16,
            temporal_channels=24,
            spatial_channels=32,
            use_spatial_context=False,
            use_global_coordinates=False,
            use_local_coordinates=False,
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        normalization = np.load(
            model_dir / "query_history_normalization.npz",
            allow_pickle=False,
        )
        self.history_mean = normalization["mean"].astype(np.float32)
        self.history_std = normalization["std"].astype(np.float32)
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        self.residual_mean = float(metrics["normalization_residual_mean"])
        self.residual_std = float(metrics["normalization_residual_std"])

    def __call__(self, history: np.ndarray) -> np.ndarray:
        normalized_history = (
            (history - self.history_mean[None, :])
            / self.history_std[None, :]
        ).astype(np.float32)
        values = self.torch.from_numpy(normalized_history).to(self.device)
        count = len(history)
        dummy_context = self.torch.zeros(
            (1, 300, 1, 1, 1),
            dtype=self.torch.float32,
            device=self.device,
        )
        coordinates = self.torch.zeros(
            (1, count, 2),
            dtype=self.torch.float32,
            device=self.device,
        )
        with self.torch.no_grad():
            normalized_increment = self.model(
                dummy_context,
                coordinates,
                values[None, :, :],
                coordinates,
            )[0]
            prediction = (
                normalized_increment * self.residual_std
                + self.residual_mean
                + self.torch.from_numpy(history[:, -1]).to(self.device)
            )
        return prediction.detach().cpu().numpy().astype(np.float32)


def metric_row(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    residual = prediction - truth
    return {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "p95_absolute_error": float(np.quantile(np.abs(residual), 0.95)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--idw-neighbors", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_root.mkdir(parents=True, exist_ok=True)
    points, history, _ = load_forecast_columns(
        RawHoldoutSpec(csv_path=args.csv_path, tile="E32N34")
    )
    tree = cKDTree(points)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    predictors: dict[int, dict[str, object]] = {}
    direct_artifacts: dict[int, dict[str, np.lib.npyio.NpzFile]] = {}
    tasks = {}
    consumed: list[Path] = [args.csv_path, Path(__file__)]
    for seed in args.seeds:
        run_root = args.confirmation_root / "E32N34" / f"seed_{seed}"
        lasso_state = run_root / "lasso_raw_supervised" / "lasso_state.pth"
        spar_dir = run_root / "saqr_point_query"
        predictors[seed] = {
            "lasso": LassoPredictor(lasso_state, device),
            "spar": SparPredictor(spar_dir, device),
        }
        direct_artifacts[seed] = {
            "lasso": np.load(
                run_root
                / "lasso_raw_supervised"
                / "direct_raw_test_predictions.npz",
                allow_pickle=False,
            ),
            "spar": np.load(
                spar_dir / "direct_raw_test_predictions.npz",
                allow_pickle=False,
            ),
        }
        tasks[seed] = build_raw_holdout_task(
            RawHoldoutSpec(
                csv_path=args.csv_path,
                tile="E32N34",
                grid_size=256,
                split_seed=seed,
                block_side=8,
            ),
            cache_dir=args.cache_dir,
        )
        consumed.extend(
            [
                lasso_state,
                run_root / "lasso_raw_supervised" / "metrics.json",
                spar_dir / "best_model.pth",
                spar_dir / "query_history_normalization.npz",
                spar_dir / "metrics.json",
            ]
        )

    rows: list[dict[str, object]] = []
    # Native 100-m masked lattice: predict every valid product cell directly.
    native_shape = (
        int(round((maximum[0] - minimum[0]) / 100.0)) + 1,
        int(round((maximum[1] - minimum[1]) / 100.0)) + 1,
    )
    native_rows = np.rint((points[:, 0] - minimum[0]) / 100.0).astype(np.int64)
    native_cols = np.rint((points[:, 1] - minimum[1]) / 100.0).astype(np.int64)
    valid_mask = np.zeros(native_shape, dtype=bool)
    valid_mask[native_rows, native_cols] = True
    for seed in args.seeds:
        seed_dir = args.output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        native_maps: dict[str, np.ndarray] = {}
        for model_name in ("lasso", "spar"):
            started = time.perf_counter()
            pieces = []
            for start in range(0, len(history), args.query_batch_size):
                end = min(start + args.query_batch_size, len(history))
                pieces.append(predictors[seed][model_name](history[start:end]))
            all_prediction = np.concatenate(pieces)
            inference_seconds = time.perf_counter() - started
            native_map = np.full(native_shape, np.nan, dtype=np.float32)
            native_map[native_rows, native_cols] = all_prediction
            native_maps[model_name] = native_map
            direct = direct_artifacts[seed][model_name]
            indices = direct["indices"].astype(np.int64)
            truth = direct["truth"].astype(np.float32)
            prediction = all_prediction[indices]
            direct_prediction = direct["prediction"].astype(np.float32)
            metrics = metric_row(truth, prediction)
            direct_rmse = float(np.sqrt(np.mean((direct_prediction - truth) ** 2)))
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "support": "native_100m_masked",
                    "grid_rows": native_shape[0],
                    "grid_cols": native_shape[1],
                    "valid_product_cells": int(valid_mask.sum()),
                    **metrics,
                    "direct_native_rmse": direct_rmse,
                    "support_penalty_mm": float(metrics["rmse"]) - direct_rmse,
                    "interpolation_seconds": 0.0,
                    "inference_seconds": inference_seconds,
                    "max_abs_difference_from_saved_direct_prediction": float(
                        np.max(np.abs(prediction - direct_prediction))
                    ),
                    "max_abs_difference_from_saved_grid": None,
                    "history_support_operation": "none",
                    "missing_cell_operation": "masked_not_filled",
                }
            )
        np.savez_compressed(
            seed_dir / "native_100m_masked_maps.npz",
            lasso=native_maps["lasso"],
            spar=native_maps["spar"],
            valid_mask=valid_mask,
            easting_min=minimum[0],
            northing_min=minimum[1],
            spacing_m=100.0,
        )

    # Square grid endpoints share one IDW history construction across seeds.
    for grid_size in args.grid_sizes:
        east_axis = np.linspace(minimum[0], maximum[0], grid_size, dtype=np.float64)
        north_axis = np.linspace(minimum[1], maximum[1], grid_size, dtype=np.float64)
        east_grid, north_grid = np.meshgrid(east_axis, north_axis, indexing="ij")
        queries = np.column_stack((east_grid.ravel(), north_grid.ravel()))
        prediction_flat = {
            seed: {
                "lasso": np.empty(len(queries), dtype=np.float32),
                "spar": np.empty(len(queries), dtype=np.float32),
            }
            for seed in args.seeds
        }
        inference_time = {
            seed: {"lasso": 0.0, "spar": 0.0} for seed in args.seeds
        }
        interpolation_started = time.perf_counter()
        for start in range(0, len(queries), args.query_batch_size):
            end = min(start + args.query_batch_size, len(queries))
            history_batch = idw_history_batch(
                tree,
                history,
                queries[start:end],
                neighbors=args.idw_neighbors,
                power=args.idw_power,
            )
            for seed in args.seeds:
                for model_name in ("lasso", "spar"):
                    started = time.perf_counter()
                    prediction_flat[seed][model_name][start:end] = predictors[seed][
                        model_name
                    ](history_batch)
                    inference_time[seed][model_name] += time.perf_counter() - started
        combined_elapsed = time.perf_counter() - interpolation_started
        interpolation_seconds = max(
            0.0,
            combined_elapsed
            - sum(
                inference_time[seed][model]
                for seed in args.seeds
                for model in ("lasso", "spar")
            ),
        )
        for seed in args.seeds:
            seed_dir = args.output_root / f"seed_{seed}"
            test_points = tasks[seed].raw_points[tasks[seed].test_target_indices]
            for model_name in ("lasso", "spar"):
                grid_prediction = prediction_flat[seed][model_name].reshape(
                    grid_size, grid_size
                )
                np.save(
                    seed_dir / f"{model_name}_{grid_size}x{grid_size}.npy",
                    grid_prediction,
                )
                sampled = RegularGridInterpolator(
                    (east_axis, north_axis),
                    grid_prediction,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )(test_points).astype(np.float32)
                direct = direct_artifacts[seed][model_name]
                if not np.array_equal(
                    direct["indices"].astype(np.int64),
                    tasks[seed].test_target_indices,
                ):
                    raise AssertionError("Saved direct indices do not match the split.")
                truth = direct["truth"].astype(np.float32)
                metrics = metric_row(truth, sampled)
                direct_rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                direct["prediction"].astype(np.float32)
                                - truth
                            )
                            ** 2
                        )
                    )
                )
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "support": f"{grid_size}x{grid_size}_idw_history",
                        "grid_rows": grid_size,
                        "grid_cols": grid_size,
                        "valid_product_cells": int(len(points)),
                        **metrics,
                        "direct_native_rmse": direct_rmse,
                        "support_penalty_mm": float(metrics["rmse"]) - direct_rmse,
                        "interpolation_seconds": interpolation_seconds,
                        "inference_seconds": inference_time[seed][model_name],
                        "max_abs_difference_from_saved_direct_prediction": None,
                        "max_abs_difference_from_saved_grid": (
                            float(
                                np.max(
                                    np.abs(
                                        grid_prediction
                                        - np.load(
                                            args.confirmation_root
                                            / "E32N34"
                                            / f"seed_{seed}"
                                            / (
                                                "lasso_raw_supervised"
                                                if model_name == "lasso"
                                                else "saqr_point_query"
                                            )
                                            / "prediction_grid.npy"
                                        )
                                    )
                                )
                            )
                            if grid_size == 256
                            else None
                        ),
                        "history_support_operation": (
                            "IDW history reconstruction plus square-grid resampling"
                        ),
                        "missing_cell_operation": "IDW_filled",
                    }
                )
        print(
            json.dumps(
                {
                    "completed_grid_size": grid_size,
                    "queries": len(queries),
                    "combined_seconds": combined_elapsed,
                }
            ),
            flush=True,
        )

    write_csv(args.output_root / "multires_metrics.csv", rows)
    (args.output_root / "multires_metrics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "device": device,
        "input_csv": str(args.csv_path.resolve()),
        "input_csv_sha256": sha256(args.csv_path),
        "confirmation_root": str(args.confirmation_root.resolve()),
        "grid_sizes": args.grid_sizes,
        "native_extent_shape": native_shape,
        "represented_valid_product_cells": int(valid_mask.sum()),
        "idw_neighbors": args.idw_neighbors,
        "idw_power": args.idw_power,
        "consumed_sha256": {
            str(path.resolve()): sha256(path)
            for path in consumed
        },
        "output_sha256": {
            str(path.relative_to(args.output_root)).replace("\\", "/"): sha256(path)
            for path in sorted(args.output_root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
