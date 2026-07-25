from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_figure


ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "spar_v2" / "aggregates" / "multiseed.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.34})

ax = axes[0]
for _, row in df.iterrows():
    ax.plot([0, 1], [row["lasso_direct_rmse"], row["spar_direct_rmse"]], color="#B8B8B8", linewidth=1.0, zorder=1)
ax.scatter([0] * len(df), df["lasso_direct_rmse"], color=COLORS["baseline"], s=34, label="Direct raw LASSO", zorder=2)
ax.scatter([1] * len(df), df["spar_direct_rmse"], color=COLORS["ours"], s=34, label="Support-preserving model", zorder=2)
ax.set_xticks([0, 1], ["LASSO", "SPAR"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("Direct raw RMSE (mm)")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.text(0.50, 0.06, "5/5 paired wins", transform=ax.transAxes, ha="center", color=COLORS["ours"])

ax = axes[1]
ax.scatter(df["cost_ratio"], df["relative_improvement_percent"], color=COLORS["ours"], s=42)
for _, row in df.iterrows():
    ax.annotate(str(int(row["seed"])), (row["cost_ratio"], row["relative_improvement_percent"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.axhline(df["relative_improvement_percent"].mean(), color=COLORS["ours"], linestyle="--", linewidth=1.0, label="Mean improvement")
ax.axvline(df["cost_ratio"].mean(), color="#777777", linestyle=":", linewidth=1.0, label="Mean cost ratio")
ax.set_xlabel("Core-time ratio to LASSO")
ax.set_ylabel("RMSE reduction (%)")
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.legend(frameon=False, loc="lower right")

save_figure(fig, "fig03_primary_results")
