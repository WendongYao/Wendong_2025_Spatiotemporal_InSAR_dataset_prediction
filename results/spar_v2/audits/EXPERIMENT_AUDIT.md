# Experiment Audit Report

**Date:** 2026-07-25
**Auditor:** independent read-only Codex reviewer agent
**Project:** CAGEO support-preserving sparse-to-dense InSAR forecasting rebuild

## Overall Verdict: WARN

No integrity failure was found. The raw-data target path, metric definitions,
result existence, aggregation, and manuscript-facing comparisons are
internally consistent. The remaining warning concerns incomplete historical
extension-code provenance for the already frozen SPAR runs; this limitation is
now disclosed in the manuscript.

## Checks

### A. Ground-truth provenance: PASS

- Real targets are loaded from the dataset forecast column, not generated from
  model outputs (`experiments_ext/raw_holdout_data.py:102-115`).
- Validation and test targets do not enter a future target grid
  (`raw_holdout_data.py:321-355,390`).
- Direct metrics compare persisted predictions with raw observations
  (`raw_holdout_data.py:420-433`; `raw_point_supervision.py:675-708`).
- EGMS experiments are classified as `real_gt`; analytic experiments use
  independent known truth and are classified as `simulation_only`.

### B. Score normalization: PASS

- RMSE, MAE, bias, and R-squared use prediction residuals and truth statistics;
  no score is divided by a statistic of the model's own predictions.
- GRU normalization is fitted from training data and predictions are restored
  to physical units (`direct_raw_baselines.py:285-294,392-407`).
- Percentage reductions use the baseline RMSE as denominator
  (`aggregate_priority_results.py:74-79,254`).

### C. Result existence and consistency: WARN

- All R083 primary rows and paired statistics match their frozen raw metrics.
- All 15 external tile/seed rows match R070/R081 raw metrics.
- All 47 R083 input hashes match the files recorded in `summary.json`.
- Aggregation reads the authoritative public frozen SPAR CSV and excludes the
  superseded local coordinate-branch development result
  (`aggregate_priority_results.py:122-135,185-198`).
- R084 closes the prediction-persistence gap. It reconstructs predictions from
  18 frozen LASSO state files without refitting and requires reconstructed RMSE
  to match the recorded value within `2e-6`
  (`experiments_ext/backfill_lasso_predictions.py:113-162`). All 52 direct
  metric runs in the audit set can now be recomputed from persisted predictions.
- Remaining warning: `source/results/spar_v2/integrity_checks.json` records that
  the historical extension-code source snapshot for the frozen SPAR runs is
  incomplete. Numerical artifacts, states, predictions, and hashes exist, but
  the unavailable historical source tree cannot be reconstructed exactly.

### D. Dead code and execution path: PASS

- SPAR inference and metrics are called by the executed path
  (`raw_point_supervision.py:547-581,660-672`).
- LASSO now persists direct predictions
  (`raw_point_supervision.py:904-944`).
- LightGBM and GRU use the common direct-result writer
  (`direct_raw_baselines.py:61-77,229-234,469-476`).
- The R083 aggregator reads raw metrics and writes all reported aggregate and
  statistical files (`aggregate_priority_results.py:180-290`).

### E. Scope: WARN

- E32N34 uses five spatial splits; seed 42 is developmental and seeds 43--46
  are the frozen confirmation set.
- External evidence is three independently trained tiles times five seeds. The
  reductions are positive but heterogeneous: 21.15%, 52.92%, and 1.48%.
- The controlled 2-km target-supervision buffer is one tile and one seed.
- The real-data study uses one forecast origin.
- The paper has been revised to avoid Europe-wide, temporal-generalization, and
  cross-region-transfer claims.

### F. Protocol-specific checks: WARN

- LightGBM target selection uses seed-42 validation only and the increment
  target is frozen for later splits.
- GRU uses training-only normalization and reconstructs physical raw targets.
- The no-anchor run changes the anchor/warm start while retaining the residual
  network capacity.
- The controlled buffer conditions share validation/test indices and each use
  81,466 training targets; their minimum separations are 100 m and 2,000 m.
- Validation labels participate in hyperparameter choice and early stopping,
  although they do not receive gradient updates. The manuscript now states
  this explicitly and reserves final evaluation for test labels.

## Claim impact

1. **Five-split E32N34 comparisons:** supported. SPAR wins 5/5 against LASSO,
   LightGBM, GRU, and the no-anchor ablation; mean reductions are 22.59%,
   23.29%, 20.30%, and 6.34%.
2. **Multi-region replication:** supported with a scope qualifier. The fixed
   architecture replicates positively in three independently trained tiles,
   but the evidence is not Europe-wide or temporal generalization.
3. **Controlled buffer:** supported as a seed-42 target-supervision result. The
   2-km condition retains 18.83% improvement versus 20.05% at 100 m.
4. **Interpolation-sensitive dense reconstruction:** supported by analytic
   known-truth experiments; it does not establish recovery of a unique physical
   dense field or deformation gradients.

## Remaining action

- Preserve the historical extension-code provenance warning in the manuscript,
  release notes, and Zenodo description. Do not imply that the missing source
  snapshot has been recovered.
