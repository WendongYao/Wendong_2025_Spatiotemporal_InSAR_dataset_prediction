# Release v2.2.0

This release aligns the public package with the manuscript
*Native-Support Sparse-to-Dense EGMS Forecasting with Anchored Neural Residuals*.

## New evidence

- Exact audit of the E32N34 EGMS L3 product geometry, dates, valid native
  100-m product cells, and product `rmse` attribute.
- Persistence, dated linear-trend, and DLinear direct-forecast baselines.
- Provenance-complete five-split reruns of LASSO and SPAR.
- A pre-declared development split (seed 42) and four frozen confirmation
  splits (seeds 43--46), with corrected repeated-holdout statistics.
- Native-support and 512, 256, and 128 raster evaluations showing the
  change-of-support limit.
- Product-quality-stratified paired errors using within-split quartiles of the
  EGMS L3 product `rmse` attribute.
- Updated figures and regression tests.

## Interpretation

The primary endpoint is forecast error at held-out valid EGMS L3 product cells.
It measures product-level predictive consistency, not independent geodetic
truth. Dense maps remain conditional query-support reconstructions. The
multi-resolution experiment demonstrates that raster coarsening compresses the
observable model difference, so dense-grid scores are not substituted for the
native-support claim.

Transformer, STGCN, graph-model, and RSASE experiments remain outside this
repository and manuscript.
