"""
Generate the CAGEO experimental analysis figures from locally regenerated outputs.

This script intentionally covers only the experiment-driven figures used in the
manuscript analysis package:

- fig03_main_benchmark_rmse
- fig04_prediction_residual_maps
- fig05_error_diagnostics
- fig06_interpolation_sensitivity
- fig07_runtime_rmse_tradeoff
- fig08_persistence_similarity_or_shap
- fig09_study_area_timeseries
- figS01_resolution_scaling
- figS02_split_leakage

It does not recreate the manually curated schematic figures `fig01` and `fig02`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

from revision_config import PROJECT_ROOT


ASSET_ROOT = PROJECT_ROOT / "cageo_submission_assets"
FIG_ROOT = ASSET_ROOT / "figures"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_figure(fig: plt.Figure, stem: str) -> None:
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def first_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve {label}. Tried:\n{joined}")


def load_csv_first(candidates: list[Path], label: str) -> pd.DataFrame:
    return read_csv(first_existing_path(candidates, label))


def lookup_row(df: pd.DataFrame, key_col: str, key_value) -> pd.Series:
    row = df.loc[df[key_col] == key_value]
    if row.empty:
        raise KeyError(f"Could not find {key_value!r} in column {key_col!r}")
    return row.iloc[0]


def _round2_summary_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "revision_outputs" / "deep_model_round2" / "round2_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "deep_model_round2_multiseed" / "round2_summary.csv",
    ]


def _round2_hybrid_summary_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "revision_outputs" / "deep_model_round2_hybrid_v2_multiseed" / "round2_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "deep_model_round2" / "round2_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "deep_model_round2_multiseed" / "round2_summary.csv",
    ]


def _round2_convlstm_summary_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "revision_outputs" / "deep_model_round2_convlstm_multiseed" / "round2_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "deep_model_round2" / "round2_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "deep_model_round2_multiseed" / "round2_summary.csv",
    ]


def _round3_cnnlstm_summary_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_5seed" / "combined_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3" / "round3_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_5seed" / "round3_summary.csv",
    ]


def _round3_cnntcn_summary_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnntcn_multiseed" / "round3_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3" / "round3_summary.csv",
    ]


def _round3_fast_summary_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_bs32_lr6e4_5seed" / "round3_summary.csv",
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_fast" / "round3_summary.csv",
    ]


def _cnn_lstm_seed42_dir_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3" / "cnn_lstm_hybrid" / "linear" / "split_seed_42",
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_5seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42",
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_3seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42",
    ]


def _cnn_tcn_seed42_dir_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3" / "cnn_tcn_hybrid" / "linear" / "split_seed_42",
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnntcn_multiseed" / "cnn_tcn_hybrid" / "linear" / "split_seed_42",
    ]


def load_plan_data() -> dict[str, object]:
    canon_root = PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E2_primary_multiseed" / "spatial_tile" / "grid_256"
    data = {
        "e2_summary": read_csv(canon_root / "primary_multiseed_summary.csv"),
        "repair_summary": load_csv_first(
            [
                PROJECT_ROOT / "revision_outputs" / "deep_model_repair" / "primary_multiseed" / "deep_repair_summary.csv",
            ],
            "deep repair summary",
        ),
        "round2_summary": load_csv_first(_round2_summary_candidates(), "round-2 summary"),
        "round2_hybrid_summary": load_csv_first(_round2_hybrid_summary_candidates(), "round-2 hybrid summary"),
        "round2_convlstm_summary": load_csv_first(_round2_convlstm_summary_candidates(), "round-2 ConvLSTM summary"),
        "round3_cnnlstm_summary": load_csv_first(_round3_cnnlstm_summary_candidates(), "round-3 CNN-LSTM summary"),
        "round3_cnntcn_summary": load_csv_first(_round3_cnntcn_summary_candidates(), "round-3 CNN-TCN summary"),
        "round3_cnnlstm_fast_summary": load_csv_first(_round3_fast_summary_candidates(), "round-3 fast CNN-LSTM summary"),
        "mask_summary": read_csv(
            PROJECT_ROOT
            / "revision_outputs"
            / "cg_suite"
            / "E3_mask_ablation"
            / "spatial_tile"
            / "grid_256"
            / "mask_ablation_summary.csv"
        ),
        "interp_forecast": read_csv(
            PROJECT_ROOT
            / "revision_outputs"
            / "cg_suite"
            / "E4_interpolation_sensitivity"
            / "spatial_tile"
            / "grid_256"
            / "forecast_metric_summary.csv"
        ),
        "interp_point": read_csv(
            PROJECT_ROOT
            / "revision_outputs"
            / "cg_suite"
            / "E4_interpolation_sensitivity"
            / "spatial_tile"
            / "grid_256"
            / "point_holdout_interpolation_summary.csv"
        ),
        "split_summary": read_csv(
            PROJECT_ROOT
            / "revision_outputs"
            / "cg_suite"
            / "E5_split_comparison"
            / "grid_256"
            / "split_comparison_summary.csv"
        ),
        "scaling_summary": read_csv(
            PROJECT_ROOT
            / "revision_outputs"
            / "cg_suite"
            / "E7_resolution_scaling"
            / "resolution_scaling_summary.csv"
        ),
        "persistence_similarity": read_csv(
            PROJECT_ROOT
            / "revision_outputs"
            / "cg_suite"
            / "E10_interpretability"
            / "spatial_tile"
            / "seed_42"
            / "persistence_similarity.csv"
        ),
        "lag_concentration": read_csv(
            PROJECT_ROOT
            / "revision_outputs"
            / "cg_suite"
            / "E10_interpretability"
            / "spatial_tile"
            / "seed_42"
            / "lightgbm_lag_concentration.csv"
        ),
    }
    return data


def first_or_lookup(df: pd.DataFrame, model_name: str) -> pd.Series:
    if "model" in df.columns:
        try:
            return lookup_row(df, "model", model_name)
        except KeyError:
            pass
    return df.iloc[0]


def build_main_benchmark_table(data: dict[str, object]) -> pd.DataFrame:
    e2 = data["e2_summary"]
    round2_hybrid = data["round2_hybrid_summary"]
    round3_l1 = data["round3_cnnlstm_summary"]
    round3_cnntcn = data["round3_cnntcn_summary"]

    rows = []
    model_specs = [
        ("cnn_lstm_hybrid_l1", "Hybrid CNN-LSTM (1-layer)", "Hybrid recurrent", first_or_lookup(round3_l1, "cnn_lstm_hybrid"), "Proposed main model"),
        ("cnn_tcn_hybrid", "Hybrid CNN-TCN", "Hybrid temporal CNN", lookup_row(round3_cnntcn, "model", "cnn_tcn_hybrid"), "Strong deep comparator"),
        ("temporal_linear_hybrid", "Temporal linear hybrid v2", "Hybrid linear-spatial", lookup_row(round2_hybrid, "model", "temporal_linear_hybrid"), "Round-2 hybrid benchmark"),
        ("lasso", "LASSO", "Sparse linear baseline", lookup_row(e2, "model", "lasso"), "Strongest classical baseline"),
        ("persistence", "Persistence", "Naive temporal baseline", lookup_row(e2, "model", "persistence"), "Reference baseline"),
        ("random_forest", "Random forest", "Tree ensemble", lookup_row(e2, "model", "random_forest"), "Nonlinear baseline"),
        ("lightgbm", "LightGBM", "Boosted tree ensemble", lookup_row(e2, "model", "lightgbm"), "Tree + SHAP baseline"),
        ("linear_trend", "Linear trend", "Temporal trend baseline", lookup_row(e2, "model", "linear_trend"), "Simple extrapolation baseline"),
    ]

    lasso_rmse = float(lookup_row(e2, "model", "lasso")["rmse_mean"])
    for internal_name, display_name, family, row, role in model_specs:
        rmse_mean = float(row["rmse_mean"])
        rows.append(
            {
                "internal_name": internal_name,
                "display_name": display_name,
                "family": family,
                "rmse_mean": rmse_mean,
                "rmse_std": float(row["rmse_std"]),
                "delta_vs_lasso_pct": 100.0 * (rmse_mean / lasso_rmse - 1.0),
                "runtime_seconds_mean": float(row["runtime_seconds_mean"]),
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def generate_f3_main_benchmark(main_benchmark: pd.DataFrame) -> None:
    order = main_benchmark.sort_values("rmse_mean", ascending=True).reset_index(drop=True)
    colors = {
        "Hybrid recurrent": "#1f77b4",
        "Hybrid temporal CNN": "#ff7f0e",
        "Hybrid linear-spatial": "#2ca02c",
        "Sparse linear baseline": "#d62728",
        "Naive temporal baseline": "#9467bd",
        "Tree ensemble": "#8c564b",
        "Boosted tree ensemble": "#e377c2",
        "Temporal trend baseline": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    x = np.arange(len(order))
    y = order["rmse_mean"].to_numpy(dtype=float)
    yerr = order["rmse_std"].to_numpy(dtype=float)
    bar_colors = [colors[row["family"]] for _, row in order.iterrows()]
    ax.bar(x, y, yerr=yerr, color=bar_colors, edgecolor="black", linewidth=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(order["display_name"], rotation=30, ha="right")
    ax.set_ylabel("RMSE (mean over seeds)")
    ax.set_title("Main 5-seed benchmark under the canonical 256 x 256 spatial-tile protocol")
    ax.grid(axis="y", alpha=0.25)
    family_handles = []
    used = set()
    for fam in order["family"]:
        if fam in used:
            continue
        used.add(fam)
        family_handles.append(patches.Patch(color=colors[fam], label=fam))
    ax.legend(
        handles=family_handles,
        frameon=False,
        fontsize=8.3,
        ncols=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    ax.set_ylim(0, float(np.max(y + yerr)) + 0.45)
    save_figure(fig, "fig03_main_benchmark_rmse")


def generate_f4_prediction_maps() -> None:
    canon_root = PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E2_primary_multiseed" / "spatial_tile" / "grid_256"
    lasso_dir = canon_root / "lasso" / "linear" / "split_seed_42"
    cnn_tcn_dir = first_existing_path(_cnn_tcn_seed42_dir_candidates(), "CNN-TCN seed-42 output directory")
    cnn_lstm_dir = first_existing_path(_cnn_lstm_seed42_dir_candidates(), "CNN-LSTM seed-42 output directory")
    dirs = {
        "Reference": lasso_dir,
        "LASSO": lasso_dir,
        "CNN-TCN hybrid": cnn_tcn_dir,
        "Hybrid CNN-LSTM": cnn_lstm_dir,
    }
    target = np.load(lasso_dir / "target_map.npy").astype(np.float32)
    split = np.load(lasso_dir / "split_masks.npz")
    valid = split["target_valid_mask"].astype(bool)
    target_masked = np.where(valid, target, np.nan)

    predictions = {}
    for name, directory in dirs.items():
        if name == "Reference":
            predictions[name] = target_masked
        else:
            pred = np.load(directory / "prediction_map.npy").astype(np.float32)
            predictions[name] = np.where(valid, pred, np.nan)

    finite_values = [np.ravel(arr[np.isfinite(arr)]) for arr in predictions.values()]
    vmin = np.nanpercentile(np.concatenate(finite_values), 2)
    vmax = np.nanpercentile(np.concatenate(finite_values), 98)
    residuals = {name: predictions[name] - target_masked for name in predictions if name != "Reference"}
    resid_abs = max(np.nanmax(np.abs(arr)) for arr in residuals.values())

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8))
    titles = ["Reference", "LASSO", "CNN-TCN hybrid", "Hybrid CNN-LSTM"]
    for col, title in enumerate(titles):
        ax = axes[0, col]
        im = ax.imshow(predictions[title], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    mask_ax = axes[1, 0]
    mask_ax.imshow(valid, cmap="Greys")
    mask_ax.set_title("Valid-domain mask", fontsize=11)
    mask_ax.set_xticks([])
    mask_ax.set_yticks([])
    for col, title in enumerate(titles[1:], start=1):
        ax = axes[1, col]
        ax.imshow(residuals[title], cmap="coolwarm", vmin=-resid_abs, vmax=resid_abs)
        ax.set_title(f"Residual: {title}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar1 = fig.colorbar(im, ax=axes[0, :], shrink=0.82, location="right", pad=0.02)
    cbar1.set_label("Displacement value")
    cbar2 = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-resid_abs, vmax=resid_abs)),
        ax=axes[1, :],
        shrink=0.82,
        location="right",
        pad=0.02,
    )
    cbar2.set_label("Prediction residual")
    fig.suptitle("Prediction maps and residual structure for the canonical seed-42 split", fontsize=15, y=0.98)
    save_figure(fig, "fig04_prediction_residual_maps")


def generate_f5_error_diagnostics() -> None:
    model_dirs = {
        "LASSO": PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E2_primary_multiseed" / "spatial_tile" / "grid_256" / "lasso" / "linear" / "split_seed_42",
        "Persistence": PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E2_primary_multiseed" / "spatial_tile" / "grid_256" / "persistence" / "linear" / "split_seed_42",
        "CNN-TCN hybrid": first_existing_path(_cnn_tcn_seed42_dir_candidates(), "CNN-TCN seed-42 diagnostics directory"),
        "Hybrid CNN-LSTM": first_existing_path(_cnn_lstm_seed42_dir_candidates(), "CNN-LSTM seed-42 diagnostics directory"),
    }
    plot_names = [
        ("scatter_true_vs_pred.png", "True vs. predicted"),
        ("residual_plot.png", "Residual vs. prediction"),
        ("binned_error.png", "Binned absolute error"),
    ]

    fig, axes = plt.subplots(len(model_dirs), len(plot_names), figsize=(12.5, 11))
    for row, (model_name, directory) in enumerate(model_dirs.items()):
        for col, (filename, col_title) in enumerate(plot_names):
            ax = axes[row, col]
            image = mpimg.imread(directory / filename)
            ax.imshow(image)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_title, fontsize=11)
            if col == 0:
                ax.text(
                    -0.06,
                    0.5,
                    model_name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                    weight="bold",
                )
    fig.suptitle("Error diagnostics for final selected models (canonical seed-42 split)", fontsize=15, y=0.995)
    save_figure(fig, "fig05_error_diagnostics")


def generate_f6_interpolation(data: dict[str, object]) -> None:
    point = data["interp_point"].copy()
    forecast = data["interp_forecast"].copy()
    method_order = ["linear", "nearest", "idw", "rbf"]
    method_positions = {method: idx for idx, method in enumerate(method_order)}
    point = point.set_index("method").loc[method_order].reset_index()
    forecast = forecast[forecast["model"].isin(["lasso", "persistence", "lightgbm", "cnn_lstm_maskaware"])].copy()
    forecast["method"] = pd.Categorical(forecast["method"], categories=method_order, ordered=True)
    forecast = forecast.sort_values(["model", "method"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    ax.bar(point["method"], point["point_holdout_rmse_mean"], color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
    ax.errorbar(
        point["method"],
        point["point_holdout_rmse_mean"],
        yerr=point["point_holdout_rmse_std"],
        fmt="none",
        ecolor="black",
        capsize=4,
        lw=1,
    )
    ax.set_ylabel("Point-holdout interpolation RMSE")
    ax.set_title("Interpolation-only accuracy")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    color_map = {
        "lasso": "#d62728",
        "persistence": "#9467bd",
        "lightgbm": "#2ca02c",
        "cnn_lstm_maskaware": "#7f7f7f",
    }
    marker_map = {"lasso": "o", "persistence": "s", "lightgbm": "^", "cnn_lstm_maskaware": "D"}
    for model_name, group in forecast.groupby("model"):
        x = np.array([method_positions[str(method)] for method in group["method"]], dtype=float)
        y = group["rmse_mean"].to_numpy(dtype=float)
        yerr = group["rmse_std"].to_numpy(dtype=float)
        ax.plot(x, y, marker=marker_map[model_name], color=color_map[model_name], lw=2, label=model_name)
        ax.fill_between(x, y - yerr, y + yerr, color=color_map[model_name], alpha=0.12)
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(method_order)
    ax.set_ylabel("Forecast RMSE (mean over seeds)")
    ax.set_title("Downstream forecast sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Interpolation sensitivity in the CAGEO suite", fontsize=15, y=1.02)
    save_figure(fig, "fig06_interpolation_sensitivity")


def generate_f7_runtime_tradeoff(main_benchmark: pd.DataFrame, data: dict[str, object]) -> None:
    points = []
    for _, row in main_benchmark.iterrows():
        if row["display_name"] in {"Persistence", "Linear trend"}:
            continue
        points.append((row["display_name"], row["family"], row["runtime_seconds_mean"], row["rmse_mean"]))
    patch_unet = lookup_row(data["round2_summary"], "model", "patch_unet_residual")
    points.append(("Patch U-Net residual", "Round-2 deep", float(patch_unet["runtime_seconds_mean"]), float(patch_unet["rmse_mean"])))
    convlstm = lookup_row(data["round2_convlstm_summary"], "model", "conv_lstm_residual")
    points.append(("ConvLSTM residual", "Round-2 deep", float(convlstm["runtime_seconds_mean"]), float(convlstm["rmse_mean"])))
    fast = first_or_lookup(data["round3_cnnlstm_fast_summary"], "cnn_lstm_hybrid")
    points.append(("Hybrid CNN-LSTM fast", "Hybrid recurrent", float(fast["runtime_seconds_mean"]), float(fast["rmse_mean"])))

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    color_map = {
        "Hybrid recurrent": "#1f77b4",
        "Hybrid temporal CNN": "#ff7f0e",
        "Hybrid linear-spatial": "#2ca02c",
        "Sparse linear baseline": "#d62728",
        "Naive temporal baseline": "#9467bd",
        "Tree ensemble": "#8c564b",
        "Boosted tree ensemble": "#e377c2",
        "Temporal trend baseline": "#7f7f7f",
        "Round-2 deep": "#17becf",
    }
    annotation_offsets = {
        "Patch U-Net residual": (10, 8, "left"),
        "LASSO": (10, 4, "left"),
        "Temporal linear hybrid v2": (10, -16, "left"),
        "Hybrid CNN-TCN": (10, -14, "left"),
        "Hybrid CNN-LSTM (1-layer)": (8, 8, "left"),
        "Hybrid CNN-LSTM fast": (8, -14, "left"),
    }
    for name, family, runtime_sec, rmse in points:
        ax.scatter(runtime_sec, rmse, s=90, color=color_map.get(family, "#444444"), edgecolors="black", linewidth=0.7)
        dx, dy, ha = annotation_offsets.get(name, (6, 4, "left"))
        ax.annotate(
            name,
            (runtime_sec, rmse),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.5,
            ha=ha,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.2),
        )
    ax.set_xlabel("Runtime per seed (s)")
    ax.set_ylabel("RMSE")
    ax.set_title("Quality-cost trade-off under the canonical protocol")
    ax.grid(alpha=0.25)
    ax.set_xlim(-20, max(runtime for _, _, runtime, _ in points) + 35)
    save_figure(fig, "fig07_runtime_rmse_tradeoff")


def generate_f8_attribution(data: dict[str, object]) -> None:
    corr = data["persistence_similarity"].copy()
    lag = data["lag_concentration"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    ax.bar(corr["model"], corr["persistence_similarity_corr"], color=["#999999", "#2ca02c", "#ff7f0e", "#1f77b4"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Correlation with persistence predictions")
    ax.set_title("Persistence similarity")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.plot(lag["last_k_lags"], lag["abs_shap_share"], marker="o", color="#d62728", lw=2)
    ax.set_xlabel("Number of most recent lags retained")
    ax.set_ylabel("Cumulative absolute SHAP share")
    ax.set_title("LightGBM temporal concentration")
    ax.grid(alpha=0.25)

    fig.suptitle("Temporal attribution and persistence behavior diagnostics", fontsize=15, y=1.02)
    save_figure(fig, "fig08_persistence_similarity_or_shap")


def generate_f9_study_area_timeseries(config_snapshot_path: Path) -> None:
    config = read_json(config_snapshot_path)
    csv_path = Path(config["resolved_csv_path"])
    df = pd.read_csv(csv_path)
    easting = df.iloc[:, 1].astype(float).to_numpy()
    northing = df.iloc[:, 2].astype(float).to_numpy()
    history = df.iloc[:, config["history_start_col"] : config["history_start_col"] + config["history_length"]].astype(float).to_numpy()
    target = df.iloc[:, config["target_col"]].astype(float).to_numpy()
    center_x = np.median(easting)
    center_y = np.median(northing)
    distances = (easting - center_x) ** 2 + (northing - center_y) ** 2
    valid_rows = np.all(np.isfinite(history), axis=1) & np.isfinite(target)
    representative_idx = np.argmin(np.where(valid_rows, distances, np.inf))
    representative_series = history[representative_idx]
    representative_target = target[representative_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    sc = ax.scatter(easting, northing, c=target, s=8, cmap="viridis", linewidths=0.0)
    ax.scatter([easting[representative_idx]], [northing[representative_idx]], c="red", s=60, marker="*", label="representative point")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_title("Study-area EGMS point cloud")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Target displacement")

    ax = axes[1]
    steps = np.arange(1, representative_series.size + 1)
    ax.plot(steps, representative_series, color="#1f77b4", lw=1.8, label="history")
    ax.scatter([representative_series.size + 1], [representative_target], color="#d62728", s=55, zorder=3, label="target")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Displacement value")
    ax.set_title("Representative EGMS displacement history")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Study-area point distribution and a representative EGMS time series", fontsize=15, y=1.02)
    save_figure(fig, "fig09_study_area_timeseries")


def generate_figs_supplement(data: dict[str, object]) -> None:
    scaling = data["scaling_summary"].copy()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    order = ["lasso", "lightgbm", "persistence", "cnn_lstm_maskaware"]
    colors = {"lasso": "#d62728", "lightgbm": "#2ca02c", "persistence": "#9467bd", "cnn_lstm_maskaware": "#7f7f7f"}
    markers = {"lasso": "o", "lightgbm": "^", "persistence": "s", "cnn_lstm_maskaware": "D"}
    for model in order:
        group = scaling[scaling["model"] == model].sort_values("grid_size")
        if group.empty:
            continue
        ax.plot(group["grid_size"], group["rmse_mean"], marker=markers[model], color=colors[model], lw=2, label=model)
        if group["rmse_std"].notna().any():
            std = group["rmse_std"].fillna(0).to_numpy(dtype=float)
            mean = group["rmse_mean"].to_numpy(dtype=float)
            x = group["grid_size"].to_numpy(dtype=float)
            ax.fill_between(x, mean - std, mean + std, color=colors[model], alpha=0.12)
    ax.set_xlabel("Grid size")
    ax.set_ylabel("RMSE")
    ax.set_title("Resolution scaling under the original CAGEO suite")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)
    save_figure(fig, "figS01_resolution_scaling")

    split = data["split_summary"].copy()
    target_models = ["persistence", "lightgbm", "cnn_tcn", "cnn_lstm_maskaware"]
    spatial = split[(split["model"].isin(target_models)) & (split["split_strategy"] == "spatial_tile")].set_index("model")
    random = split[(split["model"].isin(target_models)) & (split["split_strategy"] == "random_pixel")].set_index("model")
    optimism = split[(split["model"].isin(target_models)) & (split["split_strategy"] == "comparison")].set_index("model")

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(target_models))
    width = 0.33
    ax.bar(x - width / 2, [random.loc[m, "rmse_mean"] for m in target_models], width, label="random_pixel", color="#9ecae1")
    ax.bar(x + width / 2, [spatial.loc[m, "rmse_mean"] for m in target_models], width, label="spatial_tile", color="#3182bd")
    ymax = 0.0
    for idx, model in enumerate(target_models):
        val = float(optimism.loc[model, "inflation_optimism_pct_mean"])
        bar_top = max(float(random.loc[model, "rmse_mean"]), float(spatial.loc[model, "rmse_mean"]))
        ymax = max(ymax, bar_top)
        ax.text(idx, bar_top + 0.12, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(target_models, rotation=20, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Random-pixel optimism relative to spatial-tile evaluation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, ymax + 0.8)
    save_figure(fig, "figS02_split_leakage")


def main() -> None:
    ensure_dir(ASSET_ROOT)
    ensure_dir(FIG_ROOT)
    data = load_plan_data()
    main_benchmark = build_main_benchmark_table(data)

    generate_f3_main_benchmark(main_benchmark)
    generate_f4_prediction_maps()
    generate_f5_error_diagnostics()
    generate_f6_interpolation(data)
    generate_f7_runtime_tradeoff(main_benchmark, data)
    generate_f8_attribution(data)
    generate_f9_study_area_timeseries(
        first_existing_path(
            [
                PROJECT_ROOT / "revision_outputs" / "nontransformer_round3" / "cnn_lstm_hybrid" / "linear" / "split_seed_42" / "config_snapshot.json",
                PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_5seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42" / "config_snapshot.json",
                PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_3seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42" / "config_snapshot.json",
            ],
            "CNN-LSTM seed-42 config snapshot",
        )
    )
    generate_figs_supplement(data)
    print(json.dumps({"asset_root": str(ASSET_ROOT), "figure_count": 9}, indent=2))


if __name__ == "__main__":
    main()
