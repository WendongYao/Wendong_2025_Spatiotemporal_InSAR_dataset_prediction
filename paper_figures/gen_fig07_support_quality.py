import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import COLORS, locate_result, save_figure


support = pd.read_csv(
    locate_result(
        "results/R093_v22_aggregates/multires_support_summary.csv",
        "results/spar_v2/aggregates/native_support_v2_2/multires_support_summary.csv",
    )
)
quality = pd.read_csv(
    locate_result(
        "results/R093_v22_aggregates/quality_stratified_summary.csv",
        "results/spar_v2/aggregates/native_support_v2_2/quality_stratified_summary.csv",
    )
)

support_order = [
    "native_100m_masked",
    "512x512_idw_history",
    "256x256_idw_history",
    "128x128_idw_history",
]
support_labels = ["Native 100 m", "512×512", "256×256", "128×128"]
quality_order = ["Q1", "Q2", "Q3", "Q4"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32})

ax = axes[0]
x = np.arange(len(support_order))
width = 0.36
for offset, model, label, color in [
    (-width / 2, "lasso", "LASSO", COLORS["baseline"]),
    (width / 2, "spar", "SPAR", COLORS["ours"]),
]:
    selected = (
        support[support["model"] == model]
        .set_index("support")
        .loc[support_order]
    )
    ax.bar(
        x + offset,
        selected["mean_rmse"],
        width,
        yerr=selected["std_rmse"],
        capsize=2,
        color=color,
        label=label,
        zorder=2,
    )
ax.set_xticks(x, support_labels, rotation=16, ha="right")
ax.set_ylabel("Held-out native-cell RMSE (mm)")
ax.legend(frameon=False, ncol=2, loc="upper center")
ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")

ax = axes[1]
x = np.arange(len(quality_order))
width = 0.36
for offset, model, label, color in [
    (-width / 2, "lasso", "LASSO", COLORS["baseline"]),
    (width / 2, "spar", "SPAR", COLORS["ours"]),
]:
    selected = (
        quality[
            (quality["model"] == model)
            & (quality["quality_bin"].isin(quality_order))
        ]
        .set_index("quality_bin")
        .loc[quality_order]
    )
    ax.bar(
        x + offset,
        selected["mean_rmse"],
        width,
        yerr=selected["std_rmse"],
        capsize=2,
        color=color,
        label=label,
        zorder=2,
    )
ax.set_xticks(x, quality_order)
ax.set_xlabel("Within-partition EGMS rmse-attribute quartile")
ax.set_ylabel("Native-cell RMSE (mm)")
ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")

save_figure(fig, "fig07_support_quality")
