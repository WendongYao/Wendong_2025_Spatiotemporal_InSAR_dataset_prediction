# CAGEO v2.2 Result Summary

Generated from the provenance-complete and hashed artifacts in R087--R093.
All real-data values below are predictive errors against held-out EGMS
Level-3 product-cell values, not absolute geodetic truth.

## Product support

- E32N34 contains 128,757 rows and 128,757 unique coordinate pairs.
- The physical native 100-m extent is 692 x 1000 cells.
- 128,757 of 692,000 possible cells are populated (18.6065%).
- Inputs span 2018-01-06 through 2022-12-05; the target is 2022-12-17.
- The CSV `rmse` field is a product attribute used only for post-hoc strata.

Source: `../R087_egms_l3_product_audit/egms_l3_product_audit.json`.

## Primary native-cell result

| Model | RMSE, mean +/- SD (mm) | MAE, mean +/- SD (mm) | Core time, mean +/- SD (s) |
|---|---:|---:|---:|
| Persistence | 1.7341 +/- 0.2507 | 1.1074 +/- 0.1882 | <0.001 |
| Dated linear trend | 2.1362 +/- 0.3561 | 1.4678 +/- 0.2841 | 0.045 +/- 0.006 |
| DLinear | 1.4142 +/- 0.2282 | 0.9038 +/- 0.1713 | 38.83 +/- 1.49 |
| LASSO | 1.5075 +/- 0.2194 | 0.9793 +/- 0.1738 | 7.29 +/- 0.19 |
| LightGBM | 1.5212 +/- 0.2404 | 0.9696 +/- 0.1782 | 23.09 +/- 3.80 |
| Pointwise GRU | 1.4642 +/- 0.2393 | 0.9321 +/- 0.1776 | 106.09 +/- 23.85 |
| **SPAR** | **1.1669 +/- 0.1758** | **0.7660 +/- 0.1372** | **25.40 +/- 1.15** |

SPAR wins all five paired partitions against every direct baseline. DLinear is
the strongest new sequence baseline: the ratio-of-means RMSE reduction is
17.4868%, the mean paired reduction is 0.2473 mm, ordinary paired
`p=0.001296`, corrected repeated-holdout `p=0.005047`, corrected 95% interval
0.1243--0.3703 mm, and exact one-sided Wilcoxon `p=0.03125`.

Source: `native_primary_model_summary.csv` and
`native_primary_paired_statistics.csv`.

## Frozen confirmation result

For seeds 43--46, DLinear and SPAR obtain 1.4892 and 1.2318 mm mean RMSE.
The reduction is 17.2828% with four paired wins, corrected `p=0.0150`, and
corrected interval 0.0950--0.4197 mm. With only four overlapping partitions,
the exact Wilcoxon diagnostic is limited to `p=0.0625`.

Source: `native_confirmation_paired_statistics.csv`.

## Multi-resolution support result

| Input histories | LASSO RMSE (mm) | SPAR RMSE (mm) | SPAR reduction |
|---|---:|---:|---:|
| Native 100-m valid cells | 1.5075 | 1.1669 | 22.59% |
| 512 x 512 IDW grid | 3.0178 | 2.9400 | 2.58% |
| 256 x 256 IDW grid | 3.7168 | 3.6877 | 0.78% |
| 128 x 128 IDW grid | 4.4489 | 4.4410 | 0.18% |

The square-grid result is a composite change-of-support effect: resolution
change, support aggregation, missing-cell filling, and smoothing occur
together. It is not interpreted as evidence that IDW is universally inferior.

Source: `multires_support_summary.csv` and `multires_support_paired.csv`.

## Product-quality strata

Within each partition, held-out cells are divided into quartiles of the EGMS
CSV `rmse` attribute. SPAR beats LASSO in all five partitions in Q1--Q4.
The mean paired reductions are 0.0815, 0.2087, 0.3504, and 0.5231 mm from Q1
through Q4. The attribute is not independent ground truth and the result is
not a calibration claim.

Source: `quality_stratified_summary.csv`.

## Reproducibility verdict

- The five provenance-complete LASSO and SPAR reruns reproduce every v2.1 RMSE
  exactly.
- Each R090 launch records the full input SHA256, source hashes, environment,
  clean source worktree status, and output hashes.
- R091--R093 output hashes recompute successfully.
- Project tests: 13/13 passed.
- Public-source tests: 11/11 passed.

## Paper claim gate

**Supported:** SPAR is a task-specific native-support forecasting method with
repeatable product-level gains over six direct baselines.

**Not supported:** universal superiority, recovery of the true deformation
gradient, absolute geodetic accuracy, or an operator-independent dense field.
