import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, locate_result, save_figure


df = pd.read_csv(
    locate_result(
        "results/R093_v22_aggregates/native_primary_models_multiseed.csv",
        "results/spar_v2/aggregates/native_support_v2_2/native_primary_models_multiseed.csv",
        "results/R083_priority_aggregates/primary_models_multiseed.csv",
        "results/spar_v2/aggregates/priority_v2_1/primary_models_multiseed.csv",
    )
)

methods = [
    ("Persistence", "persistence", COLORS["neutral"]),
    ("Linear trend", "linear_trend", COLORS["cyan"]),
    ("DLinear", "dlinear", COLORS["warning"]),
    ("LASSO", "lasso", COLORS["baseline"]),
    ("LightGBM", "lightgbm", COLORS["secondary"]),
    ("GRU", "gru", COLORS["purple"]),
    ("SPAR", "spar", COLORS["ours"]),
]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.34})

ax = axes[0]
positions = list(range(len(methods)))
for _, row in df.iterrows():
    values = [row[f"{key}_native_cell_rmse"] for _, key, _ in methods]
    ax.plot(positions, values, color="#B8B8B8", linewidth=0.9, alpha=0.75, zorder=1)
for position, (label, key, color) in enumerate(methods):
    ax.scatter(
        [position] * len(df),
        df[f"{key}_native_cell_rmse"],
        color=color,
        edgecolor="white",
        linewidth=0.35,
        s=34,
        label=label,
        zorder=2,
    )
ax.set_xticks(positions, [label for label, _, _ in methods], rotation=15, ha="right")
ax.set_xlim(-0.4, len(methods) - 0.6)
ax.set_ylabel("Native-cell RMSE (mm)")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(0.98, 0.05, "SPAR lowest on all 5 partitions", transform=ax.transAxes, ha="right", color=COLORS["ours"])

ax = axes[1]
for label, key, color in methods:
    rmse = df[f"{key}_native_cell_rmse"]
    runtime = df[f"{key}_core_seconds"]
    ax.errorbar(
        runtime.mean(),
        rmse.mean(),
        xerr=runtime.std(ddof=1),
        yerr=rmse.std(ddof=1),
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=0.9,
        capsize=2.0,
        markersize=6,
        label=label,
        zorder=2,
    )
ax.set_xscale("log")
ax.set_xlabel("Mean core time (s, log scale)")
ax.set_ylabel("Mean native-cell RMSE (mm)")
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.grid(axis="both", color="#E5E5E5", linewidth=0.5, zorder=0)
ax.legend(
    frameon=False,
    ncol=2,
    loc="lower left",
    fontsize=6.7,
    columnspacing=0.7,
    handletextpad=0.3,
    borderaxespad=0.2,
)

save_figure(fig, "fig03_primary_results")
