import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, locate_result, save_figure


temporal = pd.read_csv(
    locate_result(
        "results/R098_v23_aggregates/temporal_origin_rows.csv",
        "results/spar_v2/aggregates/native_support_v2_3_core/temporal_origin_rows.csv",
    )
)
analytic = pd.read_csv(
    locate_result(
        "results/R098_v23_aggregates/analytic_multiseed_rows.csv",
        "results/spar_v2/aggregates/native_support_v2_3_core/analytic_multiseed_rows.csv",
    )
)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.36})

ax = axes[0]
method_styles = [
    ("DLinear", COLORS["warning"], "o"),
    ("LASSO", COLORS["baseline"], "s"),
    ("Causal TCN", COLORS["secondary"], "^"),
    ("SPAR", COLORS["ours"], "D"),
]
date_order = sorted(temporal["target_date"].unique())
x = np.arange(len(date_order))
for method, color, marker in method_styles:
    frame = temporal.loc[temporal["model"] == method].set_index("target_date").loc[date_order]
    ax.plot(x, frame["rmse"], marker=marker, color=color, linewidth=1.2, markersize=4.5, label=method)
ax.set_xticks(x, [str(value)[:4] + "-" + str(value)[4:6] for value in date_order])
ax.set_xlabel("Forecast origin")
ax.set_ylabel("Native-cell RMSE (mm)")
ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7)
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(0.02, 0.91, "Shorter histories: SPAR lowest on 1/4 origins", transform=ax.transAxes, fontsize=7, color="#555555")

ax = axes[1]
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
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
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

save_figure(fig, "fig05_temporal_analytic")
