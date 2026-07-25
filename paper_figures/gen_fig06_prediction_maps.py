import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from paper_plot_style import save_figure


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "experiments_ext"
SOURCE_EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from raw_holdout_data import RawHoldoutSpec, load_forecast_columns  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--csv-path", type=Path, required=True, help="Path to the E32N34 EGMS CSV from Zenodo.")
parser.add_argument(
    "--spar-predictions",
    type=Path,
    default=ROOT / "results" / "spar_v2" / "predictions" / "E32N34_seed42.npz",
)
parser.add_argument(
    "--lasso-state",
    type=Path,
    default=ROOT / "results" / "spar_v2" / "checkpoints" / "E32N34_seed42_lasso_state.pth",
)
args = parser.parse_args()

spar_path = args.spar_predictions
lasso_path = args.lasso_state
csv_path = args.csv_path

spar = np.load(spar_path)
indices = spar["indices"].astype(np.int64)
points = spar["points"].astype(np.float32)
truth = spar["truth"].astype(np.float32)
spar_pred = spar["prediction"].astype(np.float32)

spec = RawHoldoutSpec(csv_path=csv_path, tile="E32N34", grid_size=256, split_seed=42, block_side=8, buffer_blocks=0)
_, raw_history, _ = load_forecast_columns(spec)
state = torch.load(lasso_path, map_location="cpu", weights_only=False)
x = torch.tensor(raw_history[indices], dtype=torch.float32)
x_norm = (x - state["X_mean"]) / state["X_std"]
lasso_pred = ((x_norm @ state["weights"] + state["bias"]) * state["y_std"] + state["y_mean"]).numpy()

rmse = float(np.sqrt(np.mean((lasso_pred - truth) ** 2)))
if not np.isclose(rmse, 1.2151602506637573, atol=1e-5):
    raise RuntimeError(f"Reconstructed LASSO RMSE mismatch: {rmse}")

value_pool = np.concatenate([truth, lasso_pred, spar_pred])
vmin, vmax = np.quantile(value_pool, [0.01, 0.99])
err_pool = np.concatenate([lasso_pred - truth, spar_pred - truth])
elim = np.quantile(np.abs(err_pool), 0.99)

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), constrained_layout=True)
value_titles = ["Observed target", "Direct raw LASSO", "SPAR"]
value_arrays = [truth, lasso_pred, spar_pred]
value_scatter = None
for ax, title, values in zip(axes[0], value_titles, value_arrays):
    value_scatter = ax.scatter(points[:, 0], points[:, 1], c=values, s=1.2, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

error_titles = ["LASSO error", "SPAR error", "Absolute-error reduction"]
error_arrays = [lasso_pred - truth, spar_pred - truth, np.abs(lasso_pred - truth) - np.abs(spar_pred - truth)]
error_scatter = None
for ax, title, values in zip(axes[1], error_titles, error_arrays):
    error_scatter = ax.scatter(points[:, 0], points[:, 1], c=values, s=1.2, cmap="RdBu_r", vmin=-elim, vmax=elim, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(value_scatter, ax=axes[0, :], location="right", shrink=0.83, label="Displacement (mm)")
fig.colorbar(error_scatter, ax=axes[1, :], location="right", shrink=0.83, label="Error or reduction (mm)")

save_figure(fig, "fig06_prediction_maps")
