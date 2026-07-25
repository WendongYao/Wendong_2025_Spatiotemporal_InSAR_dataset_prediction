from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_figure


ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "spar_v2" / "aggregates" / "synthetic_truth.csv")
df = df[(df["scenario"] == "composite") & (df["operator"].isin(["idw", "linear", "nearest"]))].copy()
df["operator"] = pd.Categorical(df["operator"], ["idw", "linear", "nearest"], ordered=True)
df = df.sort_values("operator")
labels = ["IDW", "Linear", "Nearest"]
x = np.arange(len(df))
w = 0.34

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.34})

ax = axes[0]
ax.bar(x - w / 2, df["lasso_direct_rmse"], width=w, color=COLORS["baseline"], label="LASSO")
ax.bar(x + w / 2, df["spar_direct_rmse"], width=w, color=COLORS["ours"], label="SPAR")
ax.set_xticks(x, labels)
ax.set_ylabel("Direct analytic RMSE (mm)")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.legend(frameon=False)

ax = axes[1]
ax.bar(x - w / 2, df["lasso_dense_analytic_rmse"], width=w, color=COLORS["baseline"], label="LASSO")
ax.bar(x + w / 2, df["spar_dense_analytic_rmse"], width=w, color=COLORS["ours"], label="SPAR")
ax.set_xticks(x, labels)
ax.set_ylabel("Dense analytic RMSE (mm)")
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", weight="bold")
ax.legend(frameon=False)

save_figure(fig, "fig05_operator_support")
