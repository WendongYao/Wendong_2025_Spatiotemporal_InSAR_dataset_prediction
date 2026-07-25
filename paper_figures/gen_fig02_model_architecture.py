from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_figure


def add_box(ax, x, y, w, h, text, face, edge, fs=8.2):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=1.2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def add_arrow(ax, x1, y1, x2, y2, color="#555555", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=color, linestyle=ls))


fig, ax = plt.subplots(figsize=(7.2, 3.1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

add_box(ax, 0.02, 0.39, 0.15, 0.22, "Point history\n$\\mathbf{x}_i\\in\\mathbb{R}^{300}$", "#EAF1F8", COLORS["baseline"])
add_box(ax, 0.23, 0.62, 0.18, 0.20, "Direct LASSO anchor\n$f_{\\mathrm{LASSO}}(\\mathbf{x}_i)$", "#E7F4EA", COLORS["secondary"])
add_box(ax, 0.23, 0.18, 0.18, 0.20, "Per-time train-only\nnormalization", "#F2F2F2", "#777777")
add_box(ax, 0.47, 0.18, 0.20, 0.20, "Full-history encoder\n$300\\rightarrow96\\rightarrow24$\nGELU + LayerNorm", "#FFF1E6", COLORS["ours"])
add_box(ax, 0.73, 0.18, 0.18, 0.20, "Residual decoder\n$24\\rightarrow64\\rightarrow1$\nzero-initialized head", "#FFF1E6", COLORS["ours"])
add_box(ax, 0.73, 0.62, 0.18, 0.20, "Anchored forecast\n$\\hat y_i=f_{\\mathrm{LASSO}}+\\alpha r_\\theta$", "#F7EAF4", COLORS["purple"])

add_arrow(ax, 0.17, 0.52, 0.23, 0.72, color=COLORS["secondary"])
add_arrow(ax, 0.17, 0.48, 0.23, 0.28)
add_arrow(ax, 0.41, 0.28, 0.47, 0.28, color=COLORS["ours"])
add_arrow(ax, 0.67, 0.28, 0.73, 0.28, color=COLORS["ours"])
add_arrow(ax, 0.82, 0.38, 0.82, 0.62, color=COLORS["ours"])
add_arrow(ax, 0.41, 0.72, 0.73, 0.72, color=COLORS["secondary"])

ax.text(0.50, 0.04, "33,210 parameters; no context CNN and no coordinate branch in the frozen model", ha="center", fontsize=8.3, color="#444444")

save_figure(fig, "fig02_model_architecture")
