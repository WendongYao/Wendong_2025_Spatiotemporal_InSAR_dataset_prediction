import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, locate_result, save_figure


analytic = pd.read_csv(
    locate_result(
        "results/R098_v23_aggregates/analytic_multiseed_rows.csv",
        "results/spar_v2/aggregates/native_support_v2_3_core/analytic_multiseed_rows.csv",
    )
)
support = pd.read_csv(
    locate_result(
        "results/R091_multires_support/multires_metrics.csv",
        "results/spar_v2/multiresolution_support/multires_metrics.csv",
    )
)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.34})

ax = axes[0]
for _, row in analytic.iterrows():
    ax.plot(
        [0, 1],
        [row["pseudo_target_test_rmse"], row["analytic_test_rmse"]],
        color="#B8B8B8",
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
    )
ax.scatter(
    [0] * len(analytic),
    analytic["pseudo_target_test_rmse"],
    color=COLORS["baseline"],
    edgecolor="white",
    linewidth=0.35,
    s=32,
    zorder=2,
)
ax.scatter(
    [1] * len(analytic),
    analytic["analytic_test_rmse"],
    color=COLORS["warning"],
    edgecolor="white",
    linewidth=0.35,
    s=32,
    zorder=2,
)
ax.set_xticks([0, 1], ["Matched pseudo-target", "Analytic truth"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("Test RMSE (mm)")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(
    0.98,
    0.95,
    "Mean optimism gap 0.3115 mm\n10/10 positive",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=7,
    color=COLORS["warning"],
)

ax = axes[1]
support_order = [
    "native_100m_masked",
    "512x512_idw_history",
    "256x256_idw_history",
    "128x128_idw_history",
]
support_labels = ["Native 100 m", r"$512^2$", r"$256^2$", r"$128^2$"]
x = np.arange(len(support_order))
width = 0.36
for offset, model, label, color in [
    (-width / 2, "lasso", "LASSO", COLORS["baseline"]),
    (width / 2, "spar", "SPAR (capped dev.)", COLORS["ours"]),
]:
    selected = (
        support.loc[(support["model"] == model) & support["support"].isin(support_order)]
        .groupby("support")["rmse"]
        .agg(["mean", "std"])
        .loc[support_order]
    )
    ax.bar(
        x + offset,
        selected["mean"],
        width,
        yerr=selected["std"],
        capsize=2,
        color=color,
        label=label,
        zorder=2,
    )
ax.set_xticks(x, support_labels, rotation=12, ha="right")
ax.set_ylabel("Held-out native-cell RMSE (mm)")
ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=7)
ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
save_figure(fig, "fig08_support_diagnostics")
