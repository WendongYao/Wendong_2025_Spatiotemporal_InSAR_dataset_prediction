import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import locate_result, save_figure


spar_path = locate_result(
    "results/R069_saqr_frozen_seed42/saqr_point_query/direct_raw_test_predictions.npz",
    "results/spar_v2/predictions/E32N34_seed42.npz",
)
lasso_path = locate_result(
    "results/R069_saqr_frozen_seed42/lasso_raw_supervised/direct_raw_test_predictions.npz",
    "results/spar_v2/lasso_backfill/E32N34_seed42_spatial_block/direct_raw_test_predictions.npz",
)

spar = np.load(spar_path)
lasso = np.load(lasso_path)
for field in ("indices", "points", "truth"):
    if not np.array_equal(spar[field], lasso[field]):
        raise RuntimeError(f"SPAR/LASSO prediction artifacts disagree for {field}.")
indices = spar["indices"].astype(np.int64)
points = spar["points"].astype(np.float32)
truth = spar["truth"].astype(np.float32)
spar_pred = spar["prediction"].astype(np.float32)
lasso_pred = lasso["prediction"].astype(np.float32)

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
