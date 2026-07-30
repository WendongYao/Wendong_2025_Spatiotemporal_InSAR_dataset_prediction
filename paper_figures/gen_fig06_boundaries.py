import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, locate_result, save_figure


external = pd.read_csv(
    locate_result(
        "results/R100_v23_final_aggregates/external_region_rows.csv",
        "results/spar_v2/aggregates/native_support_v2_3_final/external_region_rows.csv",
    )
)
temporal = pd.read_csv(
    locate_result(
        "results/R098_v23_aggregates/temporal_origin_rows.csv",
        "results/spar_v2/aggregates/native_support_v2_3_core/temporal_origin_rows.csv",
    )
)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.34})

ax = axes[0]
tiles = ["E29N33", "E36N31", "E37N41"]
tile_colors = [COLORS["secondary"], COLORS["ours"], COLORS["baseline"]]
for position, (tile, color) in enumerate(zip(tiles, tile_colors)):
    wide = external.loc[external["tile"] == tile].pivot(index="seed", columns="model", values="rmse")
    reductions = 100 * (wide["DLinear"] - wide["SPAR"]) / wide["DLinear"]
    jitter = np.linspace(-0.075, 0.075, len(reductions))
    ax.scatter(
        position + jitter,
        reductions,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        s=34,
        zorder=2,
    )
    mean = 100 * (wide["DLinear"].mean() - wide["SPAR"].mean()) / wide["DLinear"].mean()
    ax.hlines(mean, position - 0.18, position + 0.18, color="#333333", linewidth=1.2, zorder=3)
    ax.text(position, mean + (2 if mean > 5 else 1), f"{mean:.1f}%", ha="center", fontsize=7.5)
ax.axhline(0, color="#444444", linewidth=0.8)
ax.set_xticks(range(len(tiles)), tiles)
ax.set_ylabel("SPAR RMSE reduction vs DLinear (%)")
ax.set_xlabel("Same-origin within-tile replication")
ax.set_ylim(-5, 55)
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")

ax = axes[1]
method_styles = [
    ("Persistence", COLORS["neutral"], "X"),
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
ax.set_xlabel("Shortened-history forecast origin")
ax.set_ylabel("Native-cell RMSE (mm)")
ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
ax.legend(
    frameon=False,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    fontsize=6.7,
    columnspacing=0.8,
    handletextpad=0.3,
)
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(0.02, 0.05, "SPAR lowest on 1/4 origins", transform=ax.transAxes, fontsize=7, color="#555555")
ax.set_ylim(1.14, 2.02)

save_figure(fig, "fig06_boundaries")
