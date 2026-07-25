# Paper Claim Audit Report

**Date:** 2026-07-25
**Paper:** *Support-Preserving Sparse-to-Dense InSAR Forecasting with Anchored Neural Residuals*

## Overall verdict: WARN

A fresh zero-context reviewer audited the current manuscript, all six rendered
figures, and 53 evidence/code files. Every material number, aggregation,
configuration, split/leakage statement, runtime qualification, figure/caption
mapping, and scope claim reconciled with the declared evidence set.

The WARN is an assurance qualification, not a result correction. Frozen
numerical predictions are integrity-checked, but launch-time hashes of the
evolving extension source are unavailable. The manuscript consistently states
that current source hashes support reproduction rather than exact historical
source reconstruction.

## Independent-review accounting

| Check | Result |
|---|---:|
| declared inputs | 60 |
| material numeric mismatches | 0 |
| aggregation mismatches | 0 |
| configuration mismatches | 0 |
| split or leakage mismatches | 0 |
| figure/caption mismatches | 0 |
| scope overclaims | 0 |
| disclosed provenance warnings | 1 |

## Verified areas

- Primary five-partition and seeds 43--46 confirmation results.
- Three independently retrained external-tile replications.
- Anchor, support, dense-backbone, known-truth, pseudo-target, and buffer diagnostics.
- Training/validation/test coverage and target-gridding separation.
- LightGBM validation-only target selection and frozen seeds 43--46 configuration.
- Runtime environment and all six figure/caption mappings.

## Submission interpretation

All paper numbers are suitable for release. The repeated spatial partitions are
explicitly treated as overlapping holdout diagnostics, not independent
geophysical cases. Dense-map results remain secondary and
interpolation-conditional. The canonical JSON binds this report to the current
paper and evidence files with SHA256 hashes. The independent review trace is
stored at `.aris/traces/paper-claim-audit/2026-07-25_run09/`.
