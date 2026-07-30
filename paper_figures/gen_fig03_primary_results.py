import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, locate_result, save_figure


df = pd.read_csv(
    locate_result(
        "results/R098_v23_aggregates/locked_confirmation_rows.csv",
        "results/spar_v2/aggregates/native_support_v2_3_core/locked_confirmation_rows.csv",
    )
)

methods = [
    ("Persistence", COLORS["neutral"], "X"),
    ("DLinear", COLORS["warning"], "o"),
    ("LASSO", COLORS["baseline"], "s"),
    ("Causal TCN", COLORS["secondary"], "^"),
    ("SPAR", COLORS["ours"], "D"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.34})

ax = axes[0]
positions = list(range(len(methods)))
for _, seed_frame in df.groupby("seed"):
    values = [
        seed_frame.loc[seed_frame["model"] == label, "rmse"].iloc[0]
        for label, _, _ in methods
    ]
    ax.plot(positions, values, color="#B8B8B8", linewidth=0.9, alpha=0.75, zorder=1)
for position, (label, color, marker) in enumerate(methods):
    values = df.loc[df["model"] == label, "rmse"]
    ax.scatter(
        [position] * len(values),
        values,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        s=34,
        marker=marker,
        label=label,
        zorder=2,
    )
ax.set_xticks(positions, [label for label, _, _ in methods], rotation=15, ha="right")
ax.set_xlim(-0.4, len(methods) - 0.6)
ax.set_ylabel("Native-cell RMSE (mm)")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(0.98, 0.05, "SPAR lowest on all 4 locked partitions", transform=ax.transAxes, ha="right", color=COLORS["ours"])

ax = axes[1]
for label, color, marker in methods:
    method_frame = df.loc[df["model"] == label]
    rmse = method_frame["rmse"]
    runtime = method_frame["core_seconds"]
    ax.errorbar(
        runtime.mean(),
        rmse.mean(),
        xerr=runtime.std(ddof=1),
        yerr=rmse.std(ddof=1),
        fmt=marker,
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
    loc="upper right",
    fontsize=6.7,
    columnspacing=0.7,
    handletextpad=0.3,
    borderaxespad=0.2,
)

save_figure(fig, "fig03_primary_results")
