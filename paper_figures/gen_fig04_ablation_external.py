import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, locate_result, save_figure


primary = pd.read_csv(
    locate_result(
        "results/R083_priority_aggregates/primary_models_multiseed.csv",
        "results/spar_v2/aggregates/priority_v2_1/primary_models_multiseed.csv",
    )
)
ext = pd.read_csv(
    locate_result(
        "results/R083_priority_aggregates/external_regions_multiseed.csv",
        "results/spar_v2/aggregates/priority_v2_1/external_regions_multiseed.csv",
    )
)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), gridspec_kw={"wspace": 0.38})

ax = axes[0]
for _, row in primary.iterrows():
    ax.plot(
        [0, 1],
        [row["no_anchor_direct_raw_rmse"], row["spar_direct_raw_rmse"]],
        color="#B8B8B8",
        linewidth=1.0,
        zorder=1,
    )
ax.scatter(
    [0] * len(primary),
    primary["no_anchor_direct_raw_rmse"],
    color=COLORS["purple"],
    edgecolor="white",
    linewidth=0.35,
    s=38,
    zorder=2,
)
ax.scatter(
    [1] * len(primary),
    primary["spar_direct_raw_rmse"],
    color=COLORS["ours"],
    edgecolor="white",
    linewidth=0.35,
    s=38,
    zorder=2,
)
ax.set_xticks([0, 1], ["No anchor", "Anchored SPAR"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("Native-cell RMSE (mm)")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(0.98, 0.05, "6.34% lower; 5/5 wins", transform=ax.transAxes, ha="right", color=COLORS["ours"])

ax = axes[1]
tiles = ["E29N33", "E36N31", "E37N41"]
tile_colors = [COLORS["secondary"], COLORS["ours"], COLORS["baseline"]]
for position, (tile, color) in enumerate(zip(tiles, tile_colors)):
    values = ext.loc[ext["tile"] == tile, "reduction_percent"].to_numpy()
    jitter = [-0.08, -0.04, 0.0, 0.04, 0.08]
    ax.scatter(
        [position + value for value in jitter],
        values,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        s=34,
        zorder=2,
    )
    mean = values.mean()
    ax.hlines(mean, position - 0.18, position + 0.18, color="#333333", linewidth=1.2, zorder=3)
    ax.text(position, mean + 2.0, f"mean {mean:.1f}%", ha="center", va="bottom", fontsize=7.2)
ax.axhline(0, color="#444444", linewidth=0.8)
ax.set_ylabel("RMSE reduction vs LASSO (%)")
ax.set_xlabel("Independently trained external tile")
ax.set_xticks(range(len(tiles)), tiles)
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.set_ylim(0, 61)

save_figure(fig, "fig04_ablation_external")
