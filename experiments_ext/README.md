# SPAR extension

This directory contains the public implementation and validation code for the
support-preserving residual forecaster (SPAR).

## Core protocol

The primary endpoint predicts the future displacement directly at each held-out
valid cell of the native 100-m EGMS L3 Ortho product. Spatial blocks are
assigned before any target gridding,
and test future targets never enter an interpolation surface or fitting step.
Validation labels may be used for early stopping and model selection.
Partition 42 is disclosed development; partitions 47--50 are the pre-specified
held-out evaluation. A dense map is an optional query obtained by interpolating
the input histories and applying the fitted point forecaster.

The final SPAR implementation uses every available training cell exactly once
per epoch with uniform objective weight and uses all validation cells for
stopping. Each query retains all 300 temporal lags. Patch-density exclusions
and caps are retained only in the archived development implementation.

## Main files

- `support_aware_model.py`: `300 -> 96 -> 24 -> 64 -> 1` anchored residual model.
- `raw_holdout_data.py`: label-independent spatial partitions and native-cell metrics.
- `raw_point_supervision.py`: direct LASSO and SPAR training/prediction paths.
- `direct_raw_baselines.py`: direct LightGBM and GRU baselines.
- `native_support_baselines.py`: persistence, dated trend, and DLinear baselines.
- `run_raw_holdout_pilot.py`: one-tile/one-partition experiment runner.
- `run_v22_native_baselines.py`: native-support baseline runner.
- `audit_egms_l3_product.py`: exact EGMS L3 product and date audit.
- `run_v22_multires_support.py`: native-to-raster change-of-support evaluation.
- `run_v22_quality_stratification.py`: product-`rmse` quartile analysis.
- `aggregate_v22_results.py`: v2.2 paper-facing tables and paired statistics.
- `native_pointwise_v23.py`: final all-cell SPAR, DLinear, and causal TCN paths.
- `run_v23_native_suite.py`: final native-support experiment runner.
- `run_v23_no_anchor.py`: otherwise identical final zero-anchor ablation.
- `aggregate_v23_results.py`: sampler, primary, temporal, and analytic tables.
- `aggregate_v23_external.py`: regional and anchor final aggregates.
- `run_multiregion_suite.py`: independently trained multi-tile/multi-seed runner.
- `run_controlled_buffer_suite.py`: matched-count 100-m versus 2-km target buffer.
- `synthetic_truth_data.py` and `run_saqr_synthetic_truth.py`: analytic known-truth tasks.
- `run_interpolated_target_confound.py`: deliberately non-deployable pseudo-target control.
- `audit_interpolated_target_confound.py`: recomputation audit for that control.
- `backfill_lasso_predictions.py`: no-refit prediction backfill from frozen LASSO states.
- `aggregate_priority_results.py`: repeated-holdout corrected statistics and runtime summaries.

## Real-data example

```powershell
python .\experiments_ext\run_raw_holdout_pilot.py `
  --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv `
  --tile E32N34 `
  --output-root .\results\_local\E32N34_seed42 `
  --cache-dir .\results\_cache `
  --seed 42 --epochs 60 --patience 12 --batch-size 16 `
  --models lasso_raw_supervised saqr_point_query
```

## Analytic known-truth example

```powershell
python .\experiments_ext\run_saqr_synthetic_truth.py `
  --output-root .\results\_local\synthetic_composite_idw `
  --scenario composite --input-interpolation idw `
  --grid-size 64 --support-points 1024 --noise 0.35 `
  --seed 42 --epochs 60 --patience 12
```

Repeat with `linear` and `nearest` for the input-interpolation sensitivity
diagnostic. Analytic future truth is evaluated independently from the input
interpolator.

## Interpolated-target confound control

```powershell
python .\experiments_ext\run_interpolated_target_confound.py `
  --output-root .\results\_local\interpolated_target_confound
python .\experiments_ext\audit_interpolated_target_confound.py `
  --result-root .\results\_local\interpolated_target_confound
```

This control intentionally constructs future pseudo-targets from all future
support values. It is not a deployable forecast protocol; it quantifies how a
matched input/target interpolation operator can make pseudo-target RMSE
optimistic relative to independent analytic truth.

## Final v2.3 identity and historical identifiers

The final implementation is `direct_spar_all_cells_uniform`. Archived machine
identifiers are retained: older seed-42/external/analytic artifacts use
`saqr_point_query`, and v2.2 confirmation artifacts use
`saqr_no_global_coord`. They remain traceable but are not pooled with the final
v2.3 primary evaluation.

Use `experiments/environment.yml` or
`experiments/requirements-revision.txt`. GPU execution is recommended for neural
training; released aggregate and prediction artifacts allow inspection without
rerunning the full experiments.
