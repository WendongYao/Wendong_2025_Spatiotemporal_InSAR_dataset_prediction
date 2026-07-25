# v2.0.0 - Support-preserving SPAR release

This release accompanies the rebuilt manuscript *Support-Preserving
Sparse-to-Dense InSAR Forecasting with Anchored Neural Residuals*.

## Added

- SPAR: a 33,210-parameter, full-history anchored neural residual forecaster.
- Raw-observation supervision and primary evaluation before target gridding.
- Five paired E32N34 spatial splits and four-seed frozen confirmation analysis.
- Three exploratory external-region evaluations.
- ConvLSTM and SimVP-style dense baseline comparisons.
- Raw-history, anchor, context, and coordinate ablations.
- Analytic known-truth tests under IDW, linear, and nearest-neighbour input interpolation.
- Strict-buffer stress test and LightGBM tail-variance diagnostic.
- Fourteen verified direct-prediction artifacts, sanitized manifests, checkpoints, and SHA256 inventory.
- Publication-quality figure scripts and rendered PDF figures.

## Main result

Across E32N34 seeds 42--46, direct raw-observation RMSE decreases from
1.5075 +/- 0.2194 mm for LASSO to 1.1669 +/- 0.1758 mm for SPAR: a 22.59%
mean reduction with five wins, paired t-test p=0.000413, and exact one-sided
Wilcoxon p=0.03125. The mean core-time ratio is 3.04x LASSO.

## Scope and limitations

Dense maps are interpolation-conditional secondary products, not uniquely true
dense deformation fields. External-region results use one seed, strict-buffer
results are confounded by severe training-volume loss, and simple analytic
regimes show that the neural correction does not always improve on LASSO.

The historical machine identifier `saqr_point_query` is retained in run
manifests for traceability; the public method name is SPAR.
