from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent

COLORS = {
    "baseline": "#4C78A8",
    "ours": "#F58518",
    "secondary": "#54A24B",
    "warning": "#E45756",
    "neutral": "#9D9D9D",
    "purple": "#B279A2",
    "cyan": "#72B7B2",
}

mpl.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220)
    plt.close(fig)
