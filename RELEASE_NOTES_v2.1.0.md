# v2.1.0 — Reviewer-priority validation and provenance release

This release accompanies the revised manuscript *Support-Preserving
Sparse-to-Dense InSAR Forecasting with Anchored Neural Residuals*.

## Added

- Five-partition direct raw LightGBM, GRU, no-anchor, and external-tile evidence.
- Nadeau--Bengio corrected resampled statistics with actual test/train ratios.
- A training-count-matched E32N34 seed-42 buffer comparison at 100 m and 2 km.
- Eighteen no-refit LASSO direct-prediction backfills from frozen states.
- A deliberately leaky analytic pseudo-target control that separates
  pseudo-target RMSE from independent analytic-truth RMSE.
- Temporal-origin, resolved-configuration, dense-diagnostic authority, and
  Figure 6 plotting manifests.
- Refreshed integrity and paper-claim audit artifacts.
- Portable figure scripts that use the public result tree directly.

## Main result

Across five overlapping E32N34 spatial partitions at one forecast origin, SPAR
reduces mean direct raw-observation RMSE from `1.5075 +/- 0.2194 mm` for LASSO to
`1.1669 +/- 0.1758 mm`, a `22.59%` mean reduction with five wins. The corrected
resampled two-sided p value is `0.001679`, and the corrected interval for the
LASSO-minus-SPAR RMSE difference is `0.2147--0.4664 mm`. These are
repeated-holdout diagnostics, not inference from five independent geophysical cases.

The mean SPAR core time is `22.70 s`, compared with `7.49 s` for LASSO,
`23.09 s` for LightGBM, and `106.09 s` for the direct GRU. Across independently retrained E29N33, E36N31, and
E37N41 partitions, the mean reductions are `21.15%`, `52.92%`, and `1.48%`.

The frozen query-sampling rule is now explicit. Training and validation exclude
patches with fewer than eight labels and cap retained patches at 128
target-blind queries; seed-42 optimization therefore uses 58,103/89,865
training labels and 14,128/18,656 validation labels. Direct test inference is
uncapped and covers all 20,236 held-out measurements. Seed 42 is the disclosed
development partition, while seeds 43--46 are frozen confirmation partitions.

## Interpretation controls

- The matched-count 2-km buffer retains an `18.83%` reduction versus `20.05%`
  in the matched-unbuffered condition.
- In the analytic pseudo-target control, matched IDW supervision gives Hybrid
  CNN-LSTM RMSE `0.1846` against the IDW pseudo-target but `0.5232` against
  independent analytic truth, exposing interpolation self-consistency optimism.
- Dense ConvLSTM and SimVP-style rows are identified as raw-label-supervised,
  seed-42 fixed-resolution diagnostics; similarly named earlier cell-level
  residual runs are not the table sources.

## Scope

The release covers E32N34, E29N33, E36N31, and E37N41 at one shared
300-history-to-one-target forecast origin. Each external tile is trained
independently. It does not claim Europe-wide transfer, temporal generalization,
blind-site forecasting, or model superiority outside the stated task.

Transformer, STGCN, graph-model, and RSASE experiments are outside this release.
