# SPAR v2.1 result package

This directory contains compact, sanitized evidence for the support-preserving
CAGEO rebuild. Source EGMS CSVs, caches, unrelated projects, and ephemeral queue
files are excluded.

## Contents

- `aggregates/`: original v2 aggregates plus `priority_v2_1/` corrected
  repeated-holdout statistics and five-seed external replications.
- `predictions/`: verified SPAR direct-prediction artifacts.
- `lasso_backfill/`: 18 frozen-state, no-refit LASSO prediction artifacts.
- `replications/external_seeds43_46/`: sanitized raw run evidence for the 12
  additional external-tile partitions.
- `checkpoints/`: frozen real-data SPAR checkpoints and LASSO state.
- `manifests/`: sanitized run metadata plus `release_evidence_v2_1/` for the
  shared forecast origin, resolved configuration, dense-run identity, and map scales.
- `diagnostics/controlled_buffer_E32N34_seed42/`: matched-count buffer evidence.
- `diagnostics/interpolated_target_confound/`: analytic pseudo-target control and audit.
- `diagnostics/dense_raw_label_seed42/`: authoritative raw-label-supervised
  Hybrid, ConvLSTM, and SimVP-style diagnostic artifacts plus the disambiguated
  earlier cell-level metric files.
- `audits/`: experiment-integrity and paper-claim audit artifacts.
- `release_manifest.json`: SHA256 and byte size for every released result file.

## Endpoint convention

`direct_raw_rmse` is the primary endpoint: one future prediction per original
measurement history. `grid_sampled_raw_rmse` bilinearly samples a dense field at
measurement coordinates and is a secondary fixed-resolution diagnostic.
`nearest_cell_point_rmse` is the separate nearest-cell endpoint to which the
within-cell MSE decomposition applies. These endpoints are not interchangeable.

## Statistical convention

Seeds 42--46 are overlapping repeated spatial holdouts of the same tile and
forecast origin, not independent geophysical cases. The package reports win
counts and a Nadeau--Bengio corrected resampled statistic using the recorded
test/train ratio. These quantities diagnose split sensitivity; they do not
constitute population inference across independent regions or dates.

## Training-query coverage

SPAR preserves the full 300-lag history of each selected query but does not use
every spatial training or validation label. The frozen query builder excludes
patches with fewer than eight assigned labels and caps retained patches at 128
target-blind selected queries. On E32N34 seed 42 this gives 58,103/89,865
training labels and 14,128/18,656 validation labels. Direct inference removes
both restrictions and predicts all 20,236 held-out points exactly once. The
resolved configuration manifest records this distinction explicitly.

## Provenance qualification

Predictions, model states, metrics, backfills, aggregate inputs, and release
hashes are auditable. Some original frozen SPAR manifests predate complete
launch-time hashing of every evolving extension source file. Exact historical
extension-source reconstruction is therefore incomplete and is disclosed in
`integrity_checks.json` and the release evidence manifests.
