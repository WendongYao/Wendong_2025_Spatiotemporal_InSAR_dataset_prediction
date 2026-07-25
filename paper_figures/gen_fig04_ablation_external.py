from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_figure


ROOT = Path(__file__).resolve().parents[1]
ab = pd.read_csv(ROOT / "results" / "spar_v2" / "aggregates" / "ablations_seed42.csv")
ext = pd.read_csv(ROOT / "results" / "spar_v2" / "aggregates" / "external_regions.csv")

order = ["frozen_full", "no_lasso_anchor", "with_context_and_coordinates", "grid_history_only"]
labels = ["Frozen", "No anchor", "+ context/coords", "Grid history only"]
ab = ab.set_index("variant").loc[order]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), gridspec_kw={"wspace": 0.38})

ax = axes[0]
colors = [COLORS["ours"], COLORS["purple"], COLORS["neutral"], COLORS["warning"]]
bars = ax.bar(range(len(ab)), ab["direct_raw_rmse"], color=colors, width=0.68)
ax.set_xticks(range(len(ab)), labels, rotation=22, ha="right")
ax.set_ylabel("Direct raw RMSE (mm)")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
for bar, val in zip(bars, ab["direct_raw_rmse"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.045, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
ax.set_ylim(0, max(ab["direct_raw_rmse"]) * 1.18)

ax = axes[1]
bars = ax.bar(ext["tile"], ext["relative_improvement_percent"], color=[COLORS["secondary"], COLORS["ours"], COLORS["baseline"]], width=0.62)
ax.axhline(0, color="#444444", linewidth=0.8)
ax.set_ylabel("RMSE reduction vs LASSO (%)")
ax.set_xlabel("External EGMS tile (seed 42)")
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
for bar, val in zip(bars, ext["relative_improvement_percent"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
ax.set_ylim(0, max(ext["relative_improvement_percent"]) * 1.22)

save_figure(fig, "fig04_ablation_external")
