# SPAR v2 result package

This directory contains the compact public evidence for the support-preserving
CAGEO rebuild. Source EGMS CSVs, caches, and unrelated project artifacts are not
included here.

## Contents

- `aggregates/`: five-seed primary results, external regions, ablations, dense baselines, known-truth tests, and summary statistics.
- `predictions/`: 14 compressed direct-prediction artifacts verified by `integrity_checks.json`.
- `checkpoints/`: frozen real-data SPAR checkpoints and the seed-42 LASSO state used by the map script.
- `manifests/`: sanitized run manifests, raw-task metadata, and model metrics.
- `diagnostics/`: change-of-support, LightGBM variance, and strict-buffer diagnostics.
- `release_manifest.json`: SHA256 and byte size for every released result file.

## Endpoint convention

`direct_raw_rmse` evaluates one future prediction per original measurement
history and is the primary endpoint. `grid_sampled_raw_rmse` samples a
cell-constant dense field at measurement coordinates and is a secondary,
change-of-support diagnostic. These endpoints must not be compared as though
they were the same estimand.

## Historical identifier

Sanitized historical manifests retain `saqr_point_query`, while public
aggregate columns use `spar_*`. Both refer to SPAR. This mapping is deliberate
and preserves traceability to the immutable run artifacts.

## Provenance note

Prediction artifacts and reported metrics were verified, and the release
records current code hashes. Some historical run manifests predate launch-time
hashing of every evolving `experiments_ext` file; exact historical extension
source provenance is therefore incomplete. The baseline repository commit and
data fingerprints are retained, and the limitation is disclosed rather than
silently reconstructed.
