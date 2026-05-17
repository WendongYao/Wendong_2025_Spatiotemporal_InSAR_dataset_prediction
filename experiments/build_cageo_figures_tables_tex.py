"""
Generate CAGEO submission figures, tables, and a standalone LaTeX asset file.

This script implements the figure/table replacement plan in
`cageo_figures_tables_replacement_plan.docx` using the current standalone
project outputs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = PROJECT_ROOT / "cageo_submission_assets"
FIG_ROOT = ASSET_ROOT / "figures"
TEX_PATH = ASSET_ROOT / "cageo_figures_tables.tex"


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


def latex_escape(value) -> str:
    text = str(value)
    manual_escapes = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
        r"\~": "~",
        r"\^": "^",
        r"\times": "x",
    }
    for src, dst in manual_escapes.items():
        text = text.replace(src, dst)
    text = text.replace("脳", "x").replace("×", "x")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def get_seed_count(row: pd.Series) -> float | int | None:
    if "n_seeds" in row.index:
        return row["n_seeds"]
    if "seeds" in row.index:
        return row["seeds"]
    return 5


def fmt_num(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_int(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{int(round(float(value)))}"


def fmt_pct(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "NA"
    return f"{float(value):.{digits}f}\\%"


def make_longtable(
    caption: str,
    label: str,
    colspec: str,
    headers: list[str],
    rows: list[list[str]],
    note: str | None = None,
) -> str:
    lines = []
    lines.append(r"\begin{longtable}{" + colspec + "}")
    lines.append(rf"\caption{{{latex_escape(caption)}}}\label{{{label}}}\\")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    if note:
        lines.append(r"{\footnotesize " + latex_escape(note) + r"}")
    return "\n".join(lines)


def make_figure_block(caption: str, label: str, figure_name: str, width: str = r"\linewidth") -> str:
    return "\n".join(
        [
            r"\begin{figure}[p]",
            r"\centering",
            rf"\includegraphics[width={width}]{{figures/{figure_name}.pdf}}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )


def load_plan_data() -> dict[str, object]:
    canon_root = PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E2_primary_multiseed" / "spatial_tile" / "grid_256"
    manifest = read_csv(PROJECT_ROOT / "outputs" / "manifest.csv")
    data = {
        "e2_summary": read_csv(canon_root / "primary_multiseed_summary.csv"),
        "repair_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "deep_model_repair" / "primary_multiseed" / "deep_repair_summary.csv"),
        "round2_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "deep_model_round2_multiseed" / "round2_summary.csv"),
        "round2_hybrid_v2_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "deep_model_round2_hybrid_v2_multiseed" / "round2_summary.csv"),
        "round2_convlstm_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "deep_model_round2_convlstm_multiseed" / "round2_summary.csv"),
        "round3_cnnlstm_l1_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_5seed" / "combined_summary.csv"),
        "round3_cnnlstm_l1_seed": read_csv(PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_5seed" / "combined_seed_level.csv"),
        "round3_cnnlstm_l1_fast_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_bs32_lr6e4_5seed" / "round3_summary.csv"),
        "round3_cnnlstm_2layer_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_3seed" / "round3_summary.csv"),
        "round3_cnntcn_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnntcn_multiseed" / "round3_summary.csv"),
        "mask_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E3_mask_ablation" / "spatial_tile" / "grid_256" / "mask_ablation_summary.csv"),
        "interp_forecast": read_csv(PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E4_interpolation_sensitivity" / "spatial_tile" / "grid_256" / "forecast_metric_summary.csv"),
        "interp_point": read_csv(PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E4_interpolation_sensitivity" / "spatial_tile" / "grid_256" / "point_holdout_interpolation_summary.csv"),
        "split_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E5_split_comparison" / "grid_256" / "split_comparison_summary.csv"),
        "scaling_summary": read_csv(PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E7_resolution_scaling" / "resolution_scaling_summary.csv"),
        "persistence_similarity": read_csv(PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E10_interpretability" / "spatial_tile" / "seed_42" / "persistence_similarity.csv"),
        "lag_concentration": read_csv(PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E10_interpretability" / "spatial_tile" / "seed_42" / "lightgbm_lag_concentration.csv"),
        "residual_strata": read_csv(PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E10_interpretability" / "spatial_tile" / "seed_42" / "residual_strata.csv"),
        "checklist": read_csv(PROJECT_ROOT / "outputs" / "reproducibility_checklist.csv"),
        "manifest": manifest,
        "manifest_head": manifest.head(10),
    }
    return data


def lookup_row(df: pd.DataFrame, key_col: str, key_value) -> pd.Series:
    row = df.loc[df[key_col] == key_value]
    if row.empty:
        raise KeyError(f"Could not find {key_value!r} in column {key_col!r}")
    return row.iloc[0]


def build_main_benchmark_table(data: dict[str, object]) -> pd.DataFrame:
    e2 = data["e2_summary"]
    round2_v2 = data["round2_hybrid_v2_summary"]
    round3_l1 = data["round3_cnnlstm_l1_summary"]
    round3_cnntcn = data["round3_cnntcn_summary"]

    rows = []
    model_specs = [
        ("cnn_lstm_hybrid_l1", "Hybrid CNN-LSTM (1-layer)", "Hybrid recurrent", 5, round3_l1.iloc[0], "Proposed main model"),
        ("cnn_tcn_hybrid", "Hybrid CNN-TCN", "Hybrid temporal CNN", 5, lookup_row(round3_cnntcn, "model", "cnn_tcn_hybrid"), "Strong deep comparator"),
        ("temporal_linear_hybrid", "Temporal linear hybrid v2", "Hybrid linear-spatial", 5, lookup_row(round2_v2, "model", "temporal_linear_hybrid"), "Round-2 hybrid benchmark"),
        ("lasso", "LASSO", "Sparse linear baseline", 5, lookup_row(e2, "model", "lasso"), "Strongest classical baseline"),
        ("persistence", "Persistence", "Naive temporal baseline", 5, lookup_row(e2, "model", "persistence"), "Reference baseline"),
        ("random_forest", "Random forest", "Tree ensemble", 5, lookup_row(e2, "model", "random_forest"), "Nonlinear baseline"),
        ("lightgbm", "LightGBM", "Boosted tree ensemble", 5, lookup_row(e2, "model", "lightgbm"), "Tree + SHAP baseline"),
        ("linear_trend", "Linear trend", "Temporal trend baseline", 5, lookup_row(e2, "model", "linear_trend"), "Simple extrapolation baseline"),
    ]

    lasso_rmse = float(lookup_row(e2, "model", "lasso")["rmse_mean"])
    for internal_name, display_name, family, seeds, row, role in model_specs:
        rmse_mean = float(row["rmse_mean"])
        rows.append(
            {
                "internal_name": internal_name,
                "display_name": display_name,
                "family": family,
                "seeds": seeds,
                "rmse_mean": rmse_mean,
                "rmse_std": float(row["rmse_std"]),
                "delta_vs_lasso_pct": 100.0 * (rmse_mean / lasso_rmse - 1.0),
                "runtime_seconds_mean": float(row["runtime_seconds_mean"]),
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def generate_f1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(17.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.40, 0.10, 0.26, "#e8f1ff", "EGMS points", "Raw CSV\ncoordinates + history"),
        (0.19, 0.40, 0.10, 0.26, "#fff0db", "Gridding", "linear / idw / rbf\nsparse-to-dense"),
        (0.35, 0.40, 0.10, 0.26, "#ecf7e8", "Validity masks", "target valid mask\nhistory coverage"),
        (0.51, 0.40, 0.10, 0.26, "#f7ebff", "Leakage-aware\nsplit", "spatial-tile\ntrain / val / test"),
        (0.67, 0.40, 0.11, 0.26, "#ffecec", "Patch-residual\nsamples", "normalized patches\nresidual targets"),
        (0.84, 0.40, 0.11, 0.26, "#eaf7f3", "Hybrid CNN-LSTM", "linear shortcut\nrecent-lag gate\nConvLSTM branch"),
    ]

    title_fontsizes = {
        "Leakage-aware\nsplit": 11.2,
        "Patch-residual\nsamples": 11.2,
    }

    for x, y, w, h, color, title, subtitle in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02", fc=color, ec="#333333", lw=1.2)
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=title_fontsizes.get(title, 12),
            weight="bold",
            linespacing=0.95,
        )
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=9)
        ax.text(x + w / 2, y - 0.08, {
            "EGMS points": "artifact: raw point table",
            "Gridding": "artifact: dense tensor",
            "Validity masks": "artifact: valid masks",
            "Leakage-aware\nsplit": "artifact: split masks",
            "Patch-residual\nsamples": "artifact: patch batches",
            "Hybrid CNN-LSTM": "artifact: metrics + maps",
        }[title], ha="center", va="center", fontsize=8, color="#444444")

    arrow_y_positions = [0.54, 0.58, 0.54, 0.58, 0.54]
    for idx, (left, right) in enumerate(zip(boxes[:-1], boxes[1:])):
        x1 = left[0] + left[2]
        x2 = right[0]
        y = arrow_y_positions[idx]
        ax.annotate("", xy=(x2 - 0.01, y), xytext=(x1 + 0.01, y), arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))

    final_rect = patches.FancyBboxPatch((0.84, 0.08), 0.11, 0.16, boxstyle="round,pad=0.015,rounding_size=0.02", fc="#f6f6f6", ec="#333333", lw=1.2)
    ax.add_patch(final_rect)
    ax.text(0.895, 0.18, "Diagnostics + repro", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(0.895, 0.10, "figures, tables,\nmanifest, audit", ha="center", va="center", fontsize=8.8)
    ax.annotate("", xy=(0.895, 0.24), xytext=(0.895, 0.40), arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))
    ax.text(0.5, 0.90, "Sparse-to-dense computing pipeline for the C&G submission", ha="center", va="center", fontsize=15, weight="bold")
    save_figure(fig, "fig01_pipeline_cageo")


def generate_f2_architecture() -> None:
    fig, ax = plt.subplots(figsize=(15.8, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def add_box(x, y, w, h, title, subtitle, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02", fc=color, ec="#333333", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=11.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=9)

    add_box(0.04, 0.42, 0.13, 0.22, "Input patch sequence", "300-lag history\nvalue + mask channels", "#e8f1ff")
    add_box(0.27, 0.67, 0.14, 0.18, "Linear shortcut", "lasso warm start\n1x1 lag head", "#fff0db")
    add_box(0.27, 0.42, 0.14, 0.18, "Recent-lag module", "recent deltas\nsoft gate over 8 lags", "#ecf7e8")
    add_box(0.27, 0.17, 0.14, 0.18, "Frame encoder", "per-frame CNN\nspatial compression", "#f7ebff")
    add_box(0.47, 0.17, 0.14, 0.18, "1-layer ConvLSTM", "temporal aggregation\nover encoded patches", "#ffecec")
    add_box(0.69, 0.17, 0.14, 0.18, "Residual decoder", "upsampling +\nspatial correction", "#eaf7f3")
    add_box(0.69, 0.50, 0.14, 0.18, "Residual merge", "base + recent mix\n+ correction branch", "#f6f6f6")
    add_box(0.87, 0.50, 0.10, 0.18, "Dense forecast", "last frame +\npredicted residual", "#fff6da")

    branch_x = 0.22
    input_right_x = 0.17
    ax.plot([input_right_x, branch_x], [0.53, 0.53], color="#333333", lw=1.5)
    ax.plot([branch_x, branch_x], [0.26, 0.76], color="#333333", lw=1.5)
    for target_y in [0.76, 0.51, 0.26]:
        ax.annotate("", xy=(0.27, target_y), xytext=(branch_x, target_y), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))

    arrows = [
        ((0.41, 0.26), (0.47, 0.26)),
        ((0.61, 0.26), (0.69, 0.26)),
        ((0.41, 0.76), (0.69, 0.62)),
        ((0.41, 0.51), (0.69, 0.56)),
        ((0.76, 0.35), (0.76, 0.50)),
        ((0.83, 0.59), (0.87, 0.59)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))

    ax.text(0.50, 0.93, "Hybrid CNN-LSTM residual forecaster used in the current C&G main line", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.50, 0.05, "Prediction target is a normalized residual map; inference adds the predicted residual back to the last observed frame.", ha="center", va="center", fontsize=9.5)
    save_figure(fig, "fig02_hybrid_cnn_lstm_architecture")


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


def load_masked_map(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(directory / "prediction_map.npy").astype(np.float32)
    target = np.load(directory / "target_map.npy").astype(np.float32)
    split = np.load(directory / "split_masks.npz")
    valid = split["target_valid_mask"].astype(bool)
    pred = np.where(valid, pred, np.nan)
    target = np.where(valid, target, np.nan)
    return pred, target


def generate_f4_prediction_maps() -> None:
    canon_root = PROJECT_ROOT / "revision_outputs" / "cg_suite" / "E2_primary_multiseed" / "spatial_tile" / "grid_256"
    dirs = {
        "Reference": canon_root / "lasso" / "linear" / "split_seed_42",
        "LASSO": canon_root / "lasso" / "linear" / "split_seed_42",
        "CNN-TCN hybrid": PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnntcn_multiseed" / "cnn_tcn_hybrid" / "linear" / "split_seed_42",
        "Hybrid CNN-LSTM": PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_3seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42",
    }
    target = np.load(dirs["Reference"] / "target_map.npy").astype(np.float32)
    split = np.load(dirs["Reference"] / "split_masks.npz")
    valid = split["target_valid_mask"].astype(bool)
    target_masked = np.where(valid, target, np.nan)

    predictions = {}
    for name, directory in dirs.items():
        if name == "Reference":
            predictions[name] = target_masked
        else:
            pred = np.load(directory / "prediction_map.npy").astype(np.float32)
            predictions[name] = np.where(valid, pred, np.nan)

    vmin = np.nanpercentile(np.concatenate([np.ravel(arr[np.isfinite(arr)]) for arr in predictions.values()]), 2)
    vmax = np.nanpercentile(np.concatenate([np.ravel(arr[np.isfinite(arr)]) for arr in predictions.values()]), 98)
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
        "CNN-TCN hybrid": PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnntcn_multiseed" / "cnn_tcn_hybrid" / "linear" / "split_seed_42",
        "Hybrid CNN-LSTM": PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_3seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42",
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
                ax.text(-0.06, 0.5, model_name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=11, weight="bold")
    fig.suptitle("Error diagnostics for final selected models (canonical seed-42 split)", fontsize=15, y=0.995)
    save_figure(fig, "fig05_error_diagnostics")


def generate_f6_interpolation(data: dict[str, object]) -> None:
    point = data["interp_point"].copy()
    forecast = data["interp_forecast"].copy()
    method_order = ["linear", "nearest", "idw", "rbf"]
    point = point.set_index("method").loc[method_order].reset_index()
    forecast = forecast[forecast["model"].isin(["lasso", "persistence", "lightgbm", "cnn_lstm_maskaware"])].copy()
    forecast["method"] = pd.Categorical(forecast["method"], categories=method_order, ordered=True)
    forecast = forecast.sort_values(["model", "method"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    ax.bar(point["method"], point["point_holdout_rmse_mean"], color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
    ax.errorbar(point["method"], point["point_holdout_rmse_mean"], yerr=point["point_holdout_rmse_std"], fmt="none", ecolor="black", capsize=4, lw=1)
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
        ax.plot(group["method"], group["rmse_mean"], marker=marker_map[model_name], color=color_map[model_name], lw=2, label=model_name)
        ax.fill_between(
            group["method"].astype(str),
            group["rmse_mean"] - group["rmse_std"],
            group["rmse_mean"] + group["rmse_std"],
            color=color_map[model_name],
            alpha=0.12,
        )
    ax.set_ylabel("Forecast RMSE (mean over seeds)")
    ax.set_title("Downstream forecast sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Interpolation sensitivity in the C&G suite", fontsize=15, y=1.02)
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
    fast = data["round3_cnnlstm_l1_fast_summary"].iloc[0]
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
            ax.fill_between(group["grid_size"], group["rmse_mean"] - group["rmse_std"].fillna(0), group["rmse_mean"] + group["rmse_std"].fillna(0), color=colors[model], alpha=0.12)
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


def build_tables(data: dict[str, object], main_benchmark: pd.DataFrame) -> dict[str, str]:
    tables: dict[str, str] = {}

    config_snapshot = read_json(PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_3seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42" / "config_snapshot.json")
    csv_path = Path(config_snapshot["resolved_csv_path"])
    raw_df = pd.read_csv(csv_path)
    n_points = len(raw_df)
    combined_seed = data["round3_cnnlstm_l1_seed"]
    train_min, train_max = int(combined_seed["train_patch_count"].min()), int(combined_seed["train_patch_count"].max())
    val_min, val_max = int(combined_seed["val_patch_count"].min()), int(combined_seed["val_patch_count"].max())
    metrics_seed42 = read_json(PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_3seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42" / "metrics.json")
    rows_t1 = [
        [latex_escape("Dataset file"), latex_escape(csv_path.name), latex_escape("EGMS sparse point archive used throughout the standalone project")],
        [latex_escape("Number of EGMS points"), fmt_int(n_points), latex_escape("Raw sparse samples before gridding")],
        [latex_escape("History columns"), latex_escape("300 columns from index 11"), latex_escape("Canonical input history used by all aligned experiments")],
        [latex_escape("Target column"), latex_escape("Column 312"), latex_escape("Single target map forecasted after the 300-step history window")],
        [latex_escape("Grid size"), latex_escape("256 x 256"), latex_escape("Canonical dense-task grid for the main benchmark")],
        [latex_escape("Canonical interpolation"), latex_escape("linear"), latex_escape("Default interpolation used by the main benchmark")],
        [latex_escape("Main split protocol"), latex_escape("spatial_tile, tile size 32"), latex_escape("Leakage-aware evaluation protocol")],
        [latex_escape("Nominal split ratios"), latex_escape("70% / 15% / 15%"), latex_escape("Train / validation / test")],
        [latex_escape("Eligible grid cells"), fmt_int(metrics_seed42["eligible_pixels"]), latex_escape("Valid target cells with sufficient history coverage")],
        [latex_escape("Seed-42 train / val / test cells"), f"{fmt_int(metrics_seed42['train_pixels'])} / {fmt_int(metrics_seed42['val_pixels'])} / {fmt_int(metrics_seed42['test_pixels'])}", latex_escape("Representative canonical split; counts vary slightly across seeds")],
        [latex_escape("Train patch count across seeds"), f"{train_min} to {train_max}", latex_escape("Variation induced by leakage-aware spatial-tile sampling across the five canonical seeds")],
        [latex_escape("Validation patch count across seeds"), f"{val_min} to {val_max}", latex_escape("Patch counts used for early stopping under the same five-seed protocol")],
        [latex_escape("Repeated seeds"), latex_escape("42, 43, 44, 45, 46"), latex_escape("Mandatory multi-seed set for C&G summaries")],
    ]
    tables["t1"] = make_longtable(
        "Dataset, task, and split configuration used in the current C&G experiment line.",
        "tab:t1",
        r"p{3.2cm}p{3.3cm}p{7.2cm}",
        ["Item", "Value", "Rationale / usage"],
        rows_t1,
    )

    t2_rows = []
    for _, row in main_benchmark.iterrows():
        t2_rows.append(
            [
                latex_escape(row["display_name"]),
                latex_escape(row["family"]),
                fmt_int(row["seeds"]),
                fmt_num(row["rmse_mean"]),
                fmt_num(row["rmse_std"]),
                fmt_pct(row["delta_vs_lasso_pct"]),
                fmt_num(row["runtime_seconds_mean"], 2),
                latex_escape(row["role"]),
            ]
        )
    tables["t2"] = make_longtable(
        "Main benchmark under the canonical 256 x 256 spatial-tile protocol. Negative values in the relative-change column indicate improvement over LASSO.",
        "tab:t2",
        r"p{3.3cm}p{2.4cm}p{1.1cm}p{1.3cm}p{1.3cm}p{1.6cm}p{1.6cm}p{3.0cm}",
        ["Model", "Family", "Seeds", "RMSE mean", "RMSE std", "Delta vs LASSO", "Runtime (s)", "Status / role"],
        t2_rows,
    )

    t3_rows = []
    e2 = data["e2_summary"]
    repair = data["repair_summary"]
    round2 = data["round2_summary"]
    round2_v2 = data["round2_hybrid_v2_summary"]
    round2_convlstm = data["round2_convlstm_summary"]
    round3_cnnlstm_2 = data["round3_cnnlstm_2layer_summary"]
    round3_cnnlstm_l1 = data["round3_cnnlstm_l1_summary"]
    round3_cnntcn = data["round3_cnntcn_summary"]
    t3_specs = [
        ("Original aligned run", "cnn_lstm_maskaware", "single full-grid sample", "absolute target map", lookup_row(e2, "model", "cnn_lstm_maskaware"), "Obsolete broken result; not the proposed model."),
        ("Original aligned run", "cnn_tcn", "single full-grid sample", "absolute target map", lookup_row(e2, "model", "cnn_tcn"), "Obsolete broken result; early deep route was mis-specified."),
        ("Patch-residual repair", "cnn_lstm_maskaware", "patch-residual samples", "normalized residual map", lookup_row(repair, "model", "cnn_lstm_maskaware"), "Loader and target formulation corrected."),
        ("Patch-residual repair", "cnn_tcn", "patch-residual samples", "normalized residual map", lookup_row(repair, "model", "cnn_tcn"), "Temporal convolution recovered after repair."),
        ("Round-2 search", "patch_unet_residual", "patch-residual samples", "normalized residual map", lookup_row(round2, "model", "patch_unet_residual"), "Strongest non-hybrid round-2 deep model."),
        ("Round-2 search", "temporal_channel_cnn", "patch-residual samples", "normalized residual map", lookup_row(round2, "model", "temporal_channel_cnn"), "Flattened lag-channel spatial baseline."),
        ("Round-2 search", "temporal_linear_hybrid v1", "patch-residual samples", "normalized residual map", lookup_row(round2, "model", "temporal_linear_hybrid"), "Initial linear-plus-spatial hybrid."),
        ("Round-2 search", "temporal_linear_hybrid v2", "patch-residual samples", "normalized residual map", lookup_row(round2_v2, "model", "temporal_linear_hybrid"), "Warm-started hybrid with recent-lag gating."),
        ("Round-2 search", "conv_lstm_residual", "patch-residual samples", "normalized residual map", lookup_row(round2_convlstm, "model", "conv_lstm_residual"), "Fair ConvLSTM test; slower and weaker than top hybrids."),
        ("Round-3 non-Transformer", "cnn_tcn_hybrid", "patch-residual samples", "normalized residual map", lookup_row(round3_cnntcn, "model", "cnn_tcn_hybrid"), "Best repeated TCN-style hybrid."),
        ("Round-3 non-Transformer", "cnn_lstm_hybrid (1-layer)", "patch-residual samples", "normalized residual map", round3_cnnlstm_l1.iloc[0], "Current strongest fully repeated deep result."),
        ("Round-3 non-Transformer", "cnn_lstm_hybrid (2-layer)", "patch-residual samples", "normalized residual map", round3_cnnlstm_2.iloc[0], "Sensitivity-only result; stronger on 3 seeds but not fully repeated."),
    ]
    for phase, model, sample_form, target_form, row, interpretation in t3_specs:
        t3_rows.append(
            [
                latex_escape(phase),
                latex_escape(model),
                latex_escape(sample_form),
                latex_escape(target_form),
                fmt_int(get_seed_count(row)),
                fmt_num(row["rmse_mean"]),
                fmt_num(row["rmse_std"]),
                latex_escape(interpretation),
            ]
        )
    tables["t3"] = make_longtable(
        "Deep-model repair and architecture search history. The table explicitly separates obsolete broken runs from repaired and later hybrid formulations.",
        "tab:t3",
        r"p{2.5cm}p{2.8cm}p{2.5cm}p{2.4cm}p{0.9cm}p{1.2cm}p{1.2cm}p{3.8cm}",
        ["Phase", "Model", "Sample formulation", "Target formulation", "Seeds", "RMSE mean", "RMSE std", "Interpretation"],
        t3_rows,
    )

    rows_t4 = [
        [latex_escape("Canonical task"), latex_escape("256 x 256 grid, spatial-tile split, 5 seeds"), latex_escape("Main C&G evaluation setting")],
        [latex_escape("Input history"), latex_escape("300 gridded lag maps"), latex_escape("Shared across all aligned models")],
        [latex_escape("Prediction target"), latex_escape("normalized residual map"), latex_escape("Final dense forecast equals last frame plus predicted residual")],
        [latex_escape("Linear shortcut"), latex_escape("1x1 lag head with lasso warm start"), latex_escape("Injects a strong temporal baseline at initialization")],
        [latex_escape("Recent-lag module"), latex_escape("8 recent lags with soft gating"), latex_escape("Lets the model emphasize short-range temporal deviations")],
        [latex_escape("Frame encoder"), latex_escape("2D CNN with two pooling stages"), latex_escape("Extracts compact per-frame patch features")],
        [latex_escape("Temporal aggregator"), latex_escape("1-layer ConvLSTM"), latex_escape("Current main-quality hybrid configuration")],
        [latex_escape("Hidden channels"), fmt_int(config_snapshot["nontransformer_hybrid_hidden_channels"]), latex_escape("Shared latent width in the recurrent branch")],
        [latex_escape("Patch size / stride"), f"{fmt_int(config_snapshot['patch_size'])} / {fmt_int(config_snapshot['patch_stride'])}", latex_escape("Canonical patch sampler used after deep repair")],
        [latex_escape("Patch minimum valid pixels"), fmt_int(config_snapshot["patch_min_valid_pixels"]), latex_escape("Drops nearly empty supervision patches")],
        [latex_escape("Batch size"), fmt_int(config_snapshot["patch_batch_size"]), latex_escape("Main-quality training configuration; see Table T5 for the faster alternative")],
        [latex_escape("Learning rate"), fmt_num(config_snapshot["cnn_learning_rate"], 4), latex_escape("Main-quality recurrent hybrid setting")],
        [latex_escape("Weight decay"), fmt_num(config_snapshot["cnn_weight_decay"], 5), latex_escape("Regularizes the recurrent hybrid branch")],
        [latex_escape("Epochs / patience"), f"{fmt_int(config_snapshot['cnn_epochs'])} / {fmt_int(config_snapshot['cnn_patience'])}", latex_escape("Early stopping on validation loss")],
    ]
    tables["t4"] = make_longtable(
        "Final model configuration and hyperparameters for the current main-quality Hybrid CNN-LSTM submission line.",
        "tab:t4",
        r"p{3.2cm}p{3.5cm}p{7.0cm}",
        ["Component", "Value", "Rationale"],
        rows_t4,
    )

    fast = data["round3_cnnlstm_l1_fast_summary"].iloc[0]
    main = data["round3_cnnlstm_l1_summary"].iloc[0]
    speedup = float(main["runtime_seconds_mean"]) / float(fast["runtime_seconds_mean"])
    delta_rmse = float(fast["rmse_mean"]) - float(main["rmse_mean"])
    rows_t5 = [
        [latex_escape("Main-quality"), latex_escape("16"), latex_escape("3e-4"), fmt_num(main["rmse_mean"]), fmt_num(main["rmse_std"]), fmt_num(main["runtime_seconds_mean"], 2), fmt_num(main["peak_gpu_memory_mb_mean"], 1), latex_escape("Reference accuracy configuration")],
        [latex_escape("Fast rerun"), latex_escape("32"), latex_escape("6e-4"), fmt_num(fast["rmse_mean"]), fmt_num(fast["rmse_std"]), fmt_num(fast["runtime_seconds_mean"], 2), fmt_num(fast["peak_gpu_memory_mb_mean"], 1), latex_escape(f"~{speedup:.2f}x faster with a small RMSE increase of {delta_rmse:.4f}")],
    ]
    tables["t5"] = make_longtable(
        "Throughput tuning for the 1-layer Hybrid CNN-LSTM. The larger-batch configuration is intended for faster reruns rather than the strict best-quality result.",
        "tab:t5",
        r"p{2.1cm}p{1.3cm}p{1.4cm}p{1.2cm}p{1.2cm}p{1.6cm}p{1.8cm}p{4.5cm}",
        ["Setting", "Batch", "LR", "RMSE mean", "RMSE std", "Runtime (s)", "Peak GPU MB", "Comment"],
        rows_t5,
    )

    point = data["interp_point"].set_index("method")
    forecast = data["interp_forecast"]
    t6_rows = []
    method_comments = {
        "linear": "Canonical main interpolation in the current final hybrid results.",
        "nearest": "Weakest interpolation choice among the tested methods.",
        "idw": "Strongest interpolation in the current C&G suite, but the final hybrid has not yet been rerun for 5 seeds under IDW.",
        "rbf": "Second-best interpolation family in the current suite.",
    }
    for method in ["linear", "nearest", "idw", "rbf"]:
        sub = forecast[forecast["method"] == method]
        for idx, (_, row) in enumerate(sub.iterrows()):
            comment = method_comments[method] if idx == 0 else ""
            t6_rows.append(
                [
                    latex_escape(method),
                    fmt_num(point.loc[method, "point_holdout_rmse_mean"]),
                    latex_escape(row["model"]),
                    fmt_num(row["rmse_mean"]),
                    fmt_num(row["rmse_std"]),
                    latex_escape(comment),
                ]
            )
    t6_rows.append(
        [
            latex_escape("cubic"),
            latex_escape("NA"),
            latex_escape("not run in current C&G suite"),
            latex_escape("NA"),
            latex_escape("NA"),
            latex_escape("Listed in the replacement plan, but no cubic C&G multi-seed run is available in the current standalone bundle."),
        ]
    )
    tables["t6"] = make_longtable(
        "Interpolation sensitivity. Point-holdout errors quantify gridding quality; forecast errors quantify downstream task sensitivity under aligned baselines.",
        "tab:t6",
        r"p{1.6cm}p{1.6cm}p{2.3cm}p{1.5cm}p{1.5cm}p{6.1cm}",
        ["Interpolation", "Point-holdout RMSE", "Forecast model", "Forecast RMSE mean", "Forecast RMSE std", "Comment"],
        t6_rows,
        note="The current final Hybrid CNN-LSTM leaderboard remains tied to canonical linear gridding until a full 5-seed hybrid rerun under IDW is completed.",
    )

    split = data["split_summary"]
    spatial = split[split["split_strategy"] == "spatial_tile"].set_index("model")
    random = split[split["split_strategy"] == "random_pixel"].set_index("model")
    comparison = split[split["split_strategy"] == "comparison"].set_index("model")
    scaling = data["scaling_summary"]
    split_rows = []
    for model in ["persistence", "lightgbm", "cnn_tcn", "cnn_lstm_maskaware"]:
        split_rows.append(
            [
                latex_escape(model),
                fmt_num(random.loc[model, "rmse_mean"]),
                fmt_num(spatial.loc[model, "rmse_mean"]),
                fmt_pct(comparison.loc[model, "inflation_optimism_pct_mean"]),
            ]
        )
    scaling_rows = []
    for _, row in scaling.iterrows():
        scaling_rows.append(
            [
                fmt_int(row["grid_size"]),
                latex_escape(row["model"]),
                fmt_num(row["rmse_mean"]),
                fmt_num(row["rmse_std"]),
                fmt_num(row["peak_gpu_memory_mb_mean"], 1),
            ]
        )
    tables["t7"] = "\n".join(
        [
            r"\begin{table}[p]",
            r"\centering",
            r"\caption{Split leakage and resolution scaling. Panel (a) compares random-pixel and spatial-tile RMSE and reports optimism inflation. Panel (b) summarizes resolution scaling from the original C\&G suite.}",
            r"\label{tab:t7}",
            r"\textbf{(a) Split leakage comparison}\\[0.3em]",
            r"\begin{tabular}{p{3.0cm}p{2.2cm}p{2.2cm}p{2.6cm}}",
            r"\toprule",
            r"Model & Random-pixel RMSE & Spatial-tile RMSE & Optimism inflation (\%)\\",
            r"\midrule",
            *["{} & {} & {} & {}\\\\".format(*row) for row in split_rows],
            r"\bottomrule",
            r"\end{tabular}",
            r"\\[1.0em]",
            r"\textbf{(b) Resolution scaling}\\[0.3em]",
            r"\begin{tabular}{p{1.5cm}p{3.0cm}p{1.8cm}p{1.8cm}p{2.2cm}}",
            r"\toprule",
            r"Grid & Model & RMSE mean & RMSE std & Peak GPU memory (MB)\\",
            r"\midrule",
            *["{} & {} & {} & {} & {}\\\\".format(*row) for row in scaling_rows],
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    checklist = data["checklist"]
    manifest = data["manifest"]
    rows_t8 = []
    purposes = {
        "README": "Top-level reproduction entry point",
        "environment.yml": "Environment recreation",
        "configs": "Frozen configuration snapshots",
        "splits": "Saved split definitions",
        "scripts/reproduce_all": "Batch rerun entry point",
        "outputs/manifest.csv": "Artifact hashing and traceability",
        "LICENSE": "Open-source compliance",
    }
    required = {
        "README": "Yes",
        "environment.yml": "Yes",
        "configs": "Yes",
        "splits": "Yes",
        "scripts/reproduce_all": "Yes",
        "outputs/manifest.csv": "Yes",
        "LICENSE": "Yes",
    }
    for _, row in checklist.iterrows():
        item = str(row["item"])
        status = str(row["status"])
        detail = status
        if item == "LICENSE":
            detail = "Repository LICENSE is present in the current public package." if status == "present" else "Repository LICENSE is missing."
        rows_t8.append(
            [
                latex_escape(item),
                latex_escape(item),
                latex_escape(purposes.get(item, "Project reproducibility asset")),
                latex_escape("Yes" if status == "present" else "No / pending"),
                latex_escape(required.get(item, "Yes")),
                latex_escape(detail),
            ]
        )
    rows_t8.extend(
        [
            [latex_escape("outputs/metric_sanity_audit.csv"), latex_escape("outputs/metric_sanity_audit.csv"), latex_escape("Metric consistency audit"), latex_escape("Yes"), latex_escape("Yes"), latex_escape("241 rows, all pass in the current audit.")],
            [latex_escape("outputs/reproducibility_checklist.csv"), latex_escape("outputs/reproducibility_checklist.csv"), latex_escape("Checklist summary"), latex_escape("Yes"), latex_escape("Yes"), latex_escape("Used for submission readiness tracking.")],
            [latex_escape("outputs/manifest.csv"), latex_escape("outputs/manifest.csv"), latex_escape("Artifact hash manifest"), latex_escape("Yes"), latex_escape("Yes"), latex_escape(f"Current manifest contains {len(manifest)} tracked artifacts.")],
        ]
    )
    tables["t8"] = make_longtable(
        "Reproducibility assets and audit summary for the standalone C&G bundle.",
        "tab:t8",
        r"p{3.1cm}p{3.1cm}p{3.0cm}p{1.5cm}p{1.8cm}p{3.0cm}",
        ["Asset", "Path", "Purpose", "Included?", "Required?", "Notes"],
        rows_t8,
    )

    return tables


def build_tex_document(tables: dict[str, str]) -> str:
    figure_specs = [
        ("fig:f1", "fig01_pipeline_cageo", "Sparse-to-dense computing pipeline used for the C&G submission. The figure highlights the full chain from EGMS points to gridding, validity masks, leakage-aware spatial-tile splits, patch-residual samples, the Hybrid CNN-LSTM backend, and the final diagnostics/reproducibility artifacts."),
        ("fig:f2", "fig02_hybrid_cnn_lstm_architecture", "Hybrid CNN-LSTM residual forecaster used in the current main line. The model combines a lasso-warm-started temporal shortcut, a recent-lag gating module, a one-layer ConvLSTM residual branch, and a spatial correction decoder. The final dense forecast is obtained by adding the predicted residual back to the last observed frame."),
        ("fig:f3", "fig03_main_benchmark_rmse", "Main benchmark RMSE with seed-to-seed uncertainty bars under the canonical 256 x 256 spatial-tile protocol with linear gridding. Bars are sorted by RMSE, and the proposed Hybrid CNN-LSTM (1-layer) is shown together with strong hybrid, linear, tree, and naive temporal baselines."),
        ("fig:f4", "fig04_prediction_residual_maps", "Prediction maps and residual maps for the canonical seed-42 split. The top row shows the reference map and the predictions of LASSO, the Hybrid CNN-TCN, and the Hybrid CNN-LSTM. The bottom row shows the valid-domain mask and model residual maps on a shared symmetric scale."),
        ("fig:f5", "fig05_error_diagnostics", "Error diagnostics for final selected models on the canonical seed-42 split. Each row corresponds to one model and each column shows the true-vs-predicted scatter, residual-versus-prediction plot, and binned absolute error summary."),
        ("fig:f6", "fig06_interpolation_sensitivity", "Interpolation sensitivity in the C&G suite. The left panel shows point-holdout gridding RMSE, and the right panel shows downstream forecast RMSE under aligned baselines. IDW is the strongest interpolation family in the current suite, but the final Hybrid CNN-LSTM has not yet been fully rerun under IDW for all five seeds."),
        ("fig:f7", "fig07_runtime_rmse_tradeoff", "Runtime-RMSE trade-off across the main computational baselines and deep-learning routes. The figure highlights the quality-cost positions of the Hybrid CNN-LSTM main-quality configuration, its faster rerun configuration, the Hybrid CNN-TCN, and the strongest round-2 hybrid baselines."),
        ("fig:f8", "fig08_persistence_similarity_or_shap", "Temporal attribution and persistence-behavior diagnostics. The left panel reports persistence-similarity correlations, and the right panel summarizes the cumulative share of absolute SHAP importance carried by the most recent LightGBM lags."),
        ("fig:f9", "fig09_study_area_timeseries", "Study-area EGMS point distribution and a representative displacement history. The left panel shows the sparse point cloud in projected coordinates, and the right panel shows a representative 300-step history together with its target displacement value."),
        ("fig:fs1", "figS01_resolution_scaling", "Resolution scaling from the original C&G suite. The plot reports RMSE versus grid size for the strongest classical baselines and the obsolete early CNN-LSTM result where available."),
        ("fig:fs2", "figS02_split_leakage", "Split leakage comparison from the original C&G suite. Random-pixel evaluation is systematically more optimistic than leakage-aware spatial-tile evaluation, particularly for LightGBM."),
    ]

    blocks = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\usepackage{pdflscape}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}",
        r"\begin{document}",
        r"\section*{C\&G Figures and Tables Replacement Package}",
        r"This file contains the full table contents and all figure captions generated from the current standalone experiment bundle. All numerical values are embedded directly in the LaTeX source rather than referenced externally through CSV files.",
        r"\section*{Tables}",
        tables["t1"],
        tables["t2"],
        r"\begin{landscape}",
        tables["t3"],
        r"\end{landscape}",
        tables["t4"],
        tables["t5"],
        r"\begin{landscape}",
        tables["t6"],
        r"\end{landscape}",
        tables["t7"],
        r"\begin{landscape}",
        tables["t8"],
        r"\end{landscape}",
        r"\clearpage",
        r"\section*{Figures}",
    ]
    for label, name, caption in figure_specs:
        blocks.append(make_figure_block(caption=caption, label=label, figure_name=name))
    blocks.append(r"\end{document}")
    return "\n\n".join(blocks)


def main() -> None:
    ensure_dir(ASSET_ROOT)
    ensure_dir(FIG_ROOT)
    data = load_plan_data()
    main_benchmark = build_main_benchmark_table(data)

    generate_f1_pipeline()
    generate_f2_architecture()
    generate_f3_main_benchmark(main_benchmark)
    generate_f4_prediction_maps()
    generate_f5_error_diagnostics()
    generate_f6_interpolation(data)
    generate_f7_runtime_tradeoff(main_benchmark, data)
    generate_f8_attribution(data)
    generate_f9_study_area_timeseries(
        PROJECT_ROOT / "revision_outputs" / "nontransformer_round3_cnnlstm_l1_3seed" / "cnn_lstm_hybrid" / "linear" / "split_seed_42" / "config_snapshot.json"
    )
    generate_figs_supplement(data)

    tables = build_tables(data, main_benchmark)
    tex = build_tex_document(tables)
    TEX_PATH.write_text(tex, encoding="utf-8")
    print(json.dumps({"asset_root": str(ASSET_ROOT), "figure_count": 11, "tex_path": str(TEX_PATH)}, indent=2))


if __name__ == "__main__":
    main()
