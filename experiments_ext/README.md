# SPAR extension

This directory contains the public implementation and validation code for the
support-preserving anchored residual forecaster (SPAR).

## Core protocol

The primary endpoint predicts the future displacement directly at each original
measurement support. Spatial blocks are assigned before any target gridding,
and test future targets never enter an interpolation surface or fitting step.
Validation labels may be used for early stopping and model selection. Seed 42
is the disclosed development partition because its test scores informed the
frozen architecture; seeds 43--46 are the confirmation partitions. A `256 x
256` dense map is a secondary product obtained by interpolating the 300 input
histories and applying the frozen point forecaster.

The frozen SPAR implementation assigns training and validation queries to
`16 x 16` patches, excludes patches with fewer than eight assigned labels, and
caps each retained patch at 128 target-blind selected queries. All 300 temporal
lags are retained for every selected query. Direct inference uses a minimum of
one point and no cap, so every held-out measurement receives one prediction.
The seed-42 metrics record 58,103/89,865 training labels and 14,128/18,656
validation labels in the optimization batches, with 20,236/20,236 test points
predicted.

## Main files

- `support_aware_model.py`: `300 -> 96 -> 24 -> 64 -> 1` anchored residual model.
- `raw_holdout_data.py`: label-independent spatial partitions and measurement-level metrics.
- `raw_point_supervision.py`: direct LASSO and SPAR training/prediction paths.
- `direct_raw_baselines.py`: direct raw LightGBM and GRU baselines.
- `run_raw_holdout_pilot.py`: one-tile/one-partition experiment runner.
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

## Historical identifiers

Archived machine identifiers are retained. The seed-42, original external, and
analytic artifacts use `saqr_point_query`; the authoritative E32N34 seed-43--46
confirmation artifacts use `saqr_no_global_coord`. Both implement the paper's
SPAR family, and the latter is the frozen no-coordinate architecture reported in
the five-partition aggregate.

Use `experiments/environment.yml` or
`experiments/requirements-revision.txt`. GPU execution is recommended for neural
training; released aggregate and prediction artifacts allow inspection without
rerunning the full experiments.
