from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_figure


def box(ax, xy, wh, text, face, edge, fontsize=8.5, lw=1.2):
    patch = FancyBboxPatch(
        xy,
        *wh,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color="#555555", style="-", connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            linestyle=style,
            connectionstyle=connectionstyle,
        )
    )


fig, ax = plt.subplots(figsize=(7.2, 3.7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.01, 0.89, "(a) Non-deployable matched-pseudo-target control", weight="bold", va="center")
ax.text(0.01, 0.39, "(b) Support-preserving formulation", weight="bold", va="center")

box(ax, (0.02, 0.63), (0.18, 0.16), "Analytic support\nhistories + future", "#EAF1F8", COLORS["baseline"])
box(ax, (0.27, 0.63), (0.20, 0.16), "Matched gridding of\ninputs and target", "#FCE8E6", COLORS["warning"])
box(ax, (0.54, 0.63), (0.17, 0.16), "Dense sequence\nmodel", "#F2F2F2", "#666666")
box(ax, (0.78, 0.63), (0.19, 0.16), "Pseudo-target score\nand analytic error", "#FCE8E6", COLORS["warning"])
arrow(ax, (0.20, 0.71), (0.27, 0.71))
arrow(ax, (0.47, 0.71), (0.54, 0.71))
arrow(ax, (0.71, 0.71), (0.78, 0.71))
ax.text(0.37, 0.58, "deliberately includes held-out future support", color=COLORS["warning"], ha="center", fontsize=8)

box(ax, (0.02, 0.12), (0.20, 0.16), "Native 300-step history\nfor each valid L3 cell", "#EAF1F8", COLORS["baseline"])
box(ax, (0.30, 0.12), (0.20, 0.16), "All-cell SPAR\nresidual forecaster", "#FFF1E6", COLORS["ours"], lw=1.5)
box(ax, (0.59, 0.12), (0.18, 0.16), "Native-cell forecast\nand primary error", "#E7F4EA", COLORS["secondary"])
box(ax, (0.81, 0.12), (0.17, 0.16), "Optional dense query\nfrom gridded histories", "#F2F2F2", "#666666")
arrow(ax, (0.22, 0.20), (0.30, 0.20))
arrow(ax, (0.50, 0.20), (0.59, 0.20), color=COLORS["secondary"])
arrow(ax, (0.49, 0.14), (0.82, 0.14), color="#777777", style="--", connectionstyle="arc3,rad=0.42")
ax.text(
    0.72,
    0.055,
    "apply fitted model to interpolated histories (support-dependent)",
    color="#444444",
    ha="center",
    fontsize=8.5,
)

save_figure(fig, "fig01_support_workflow")
