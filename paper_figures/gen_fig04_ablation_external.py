import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, locate_result, save_figure


sampler = pd.read_csv(
    locate_result(
        "results/R098_v23_aggregates/sampler_ablation.csv",
        "results/spar_v2/aggregates/native_support_v2_3_core/sampler_ablation.csv",
    )
)
ext = pd.read_csv(
    locate_result(
        "results/R100_v23_final_aggregates/external_region_rows.csv",
        "results/spar_v2/aggregates/native_support_v2_3_final/external_region_rows.csv",
    )
)
anchor = pd.read_csv(
    locate_result(
        "results/R100_v23_final_aggregates/anchor_ablation_rows.csv",
        "results/spar_v2/aggregates/native_support_v2_3_final/anchor_ablation_rows.csv",
    )
)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={"wspace": 0.48})

ax = axes[0]
sampler_order = [
    ("legacy_capped_selection", "Capped", COLORS["neutral"]),
    ("all_cells_uniform", "Uniform", COLORS["ours"]),
    ("all_cells_density_balanced", "Density-\nbalanced", COLORS["cyan"]),
]
for position, (key, label, color) in enumerate(sampler_order):
    row = sampler.loc[sampler["sampler"] == key].iloc[0]
    ax.bar(position, row["native_cell_rmse"], color=color, width=0.68, zorder=2)
    ax.text(
        position,
        row["native_cell_rmse"] + 0.012,
        f"{100 * row['train_cell_coverage']:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7,
    )
ax.set_xticks(range(len(sampler_order)), [label for _, label, _ in sampler_order])
ax.set_ylabel("Native-cell RMSE (mm)")
ax.set_ylim(0, 0.95)
ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(0.5, 0.02, "Labels: cell coverage", transform=ax.transAxes, ha="center", fontsize=7.5)

ax = axes[1]
for seed, seed_frame in anchor.groupby("seed"):
    no_anchor = seed_frame.loc[seed_frame["variant"] == "SPAR without anchor", "rmse"].iloc[0]
    anchored = seed_frame.loc[seed_frame["variant"] == "SPAR", "rmse"].iloc[0]
    ax.plot(
        [0, 1],
        [no_anchor, anchored],
        color="#B8B8B8",
        linewidth=1.0,
        zorder=1,
    )
ax.scatter(
    [0] * anchor["seed"].nunique(),
    anchor.loc[anchor["variant"] == "SPAR without anchor", "rmse"],
    color=COLORS["purple"],
    edgecolor="white",
    linewidth=0.35,
    s=38,
    zorder=2,
)
ax.scatter(
    [1] * anchor["seed"].nunique(),
    anchor.loc[anchor["variant"] == "SPAR", "rmse"],
    color=COLORS["ours"],
    edgecolor="white",
    linewidth=0.35,
    s=38,
    zorder=2,
)
ax.set_xticks([0, 1], ["No anchor", "Anchored SPAR"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("Native-cell RMSE (mm)")
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(0.98, 0.05, "4.10% lower; 4/4", transform=ax.transAxes, ha="right", color=COLORS["ours"], fontsize=8)

ax = axes[2]
tiles = ["E29N33", "E36N31", "E37N41"]
tile_colors = [COLORS["secondary"], COLORS["ours"], COLORS["baseline"]]
for position, (tile, color) in enumerate(zip(tiles, tile_colors)):
    tile_frame = ext.loc[ext["tile"] == tile]
    wide = tile_frame.pivot(index="seed", columns="model", values="rmse")
    values = 100 * (wide["DLinear"] - wide["SPAR"]) / wide["DLinear"]
    jitter = np.linspace(-0.075, 0.075, len(values))
    ax.scatter(
        [position + value for value in jitter],
        values,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        s=34,
        zorder=2,
    )
    mean = 100 * (wide["DLinear"].mean() - wide["SPAR"].mean()) / wide["DLinear"].mean()
    ax.hlines(mean, position - 0.18, position + 0.18, color="#333333", linewidth=1.2, zorder=3)
    offset = 2.0 if mean > 5 else 1.0
    ax.text(position, mean + offset, f"{mean:.1f}%", ha="center", va="bottom", fontsize=7.2)
ax.axhline(0, color="#444444", linewidth=0.8)
ax.set_ylabel("RMSE reduction vs DLinear (%)")
ax.set_xticks(range(len(tiles)), tiles)
ax.text(0.02, 0.97, "(c)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.set_ylim(-5, 55)

save_figure(fig, "fig04_ablation_external")
