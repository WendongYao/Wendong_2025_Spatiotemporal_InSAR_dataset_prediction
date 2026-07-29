# SPAR v2.2 result package

This directory contains compact, sanitized evidence for the native-support
CAGEO rebuild. Source EGMS CSVs, caches, unrelated projects, and ephemeral queue
files are excluded.

## Native-support additions

- `product_audit/`: exact E32N34 EGMS L3 product geometry, temporal support,
  valid-cell count, and product `rmse` distribution.
- `native_baselines/`: persistence, dated ordinary-least-squares trend, and
  DLinear results for the development split and four frozen confirmation splits.
- `provenance_complete_reruns/`: LASSO and SPAR reruns with launch-time source,
  input, configuration, output, and environment hashes.
- `multiresolution_support/`: native-support and 512, 256, and 128 raster
  evaluation artifacts for all five frozen splits.
- `product_quality_strata/`: paired LASSO--SPAR errors by within-split quartile
  of the EGMS L3 product `rmse` attribute.
- `aggregates/native_support_v2_2/`: paper-facing tables, paired statistics,
  confirmation-only analyses, and result-to-claim summary.

The v2.0 and v2.1 subdirectories remain present so that the development history
and reviewer-priority controls stay auditable.

## Primary endpoint

The primary endpoint is one forecast at every held-out valid 100-m EGMS L3
product cell. It evaluates product-level predictive consistency and is not
presented as validation against independent geodetic truth. Dense raster maps
are conditional query-support reconstructions and are secondary diagnostics.

## Statistical convention

Seed 42 is the development split. Seeds 43--46 are frozen confirmation splits.
All five splits share one tile and forecast origin, so they are not independent
geophysical replicates. We report split-wise wins, paired differences, an
exact one-sided Wilcoxon statistic, and a Nadeau--Bengio corrected resampled
statistic. The confirmation-only result is distinguished from the inclusive
five-split descriptive result.

## Scope and provenance

The package covers the E32N34 EGMS L3 product only. The 302 dated displacement
columns support a fixed 300-observation history, one skipped acquisition, and
one 12-day target; this does not provide three comparable rolling origins
without changing the task definition. Transformer, STGCN, graph-model, and
RSASE experiments are outside this release.

The v2.2 LASSO/SPAR confirmation reruns record complete launch-time provenance.
Earlier frozen artifacts are retained for traceability and preserve their
previously disclosed extension-source provenance qualification.
