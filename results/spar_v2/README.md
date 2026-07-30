# SPAR v2.3 result package

This directory contains the compact, sanitized evidence for the final
native-support CAGEO submission package. Source EGMS CSVs, caches, unrelated
projects, and ephemeral queue files are excluded.

## v2.3 additions

- `development/sampler_ablation/`: the pre-lock comparison that removed the
  inherited capped sampler and selected uniform all-cell training.
- `primary/pre_specified_partitions_47_50/`: persistence, DLinear, LASSO,
  causal TCN, and final SPAR results, predictions, task metadata, and hashes.
- `boundaries/shortened_history_temporal/`: four 240-history forecast origins
  on E32N34 partition 47, including the negative DLinear/TCN comparison.
- `diagnostics/analytic_multirealization/`: ten matched-IDW pseudo-target
  realizations scored against both pseudo-targets and independent analytic
  truth.
- `replication/same_origin_within_tile/`: independently trained E29N33,
  E36N31, and E37N41 models under the final sampler.
- `development/anchor_ablation/`: otherwise identical anchored and zero-anchor
  networks across partitions 47--50.
- `aggregates/native_support_v2_3_core/` and
  `aggregates/native_support_v2_3_final/`: paper-facing rows, summaries,
  paired diagnostics, and consumed/output SHA256 manifests.

The v2.0--v2.2 directories remain for traceability. Their capped-sampler
development results are not pooled with the v2.3 final configuration.

## Evidence hierarchy and scope

Partition 42 is development only. Partitions 47--50 are the pre-specified
held-out E32N34 evaluation, but they overlap and are not independent
geophysical replicates. The additional tiles are same-origin within-tile
replications, not cross-region transfer. The four temporal origins shorten the
history from 300 to 240 values and therefore form a counter-test rather than
additional confirmation. The analytic field diagnoses matched-interpolation
optimism and is not a physical Earth model.

The primary endpoint is one future delivered value for every held-out valid
EGMS Level-3 product cell. It measures product-level predictive consistency,
not error against independent geodetic truth. Optional raster queries remain
conditional on interpolation operator and resolution.

## Integrity

Every released result tree includes task/configuration metadata and SHA256
records. The aggregate manifests enumerate their consumed files and generated
outputs. Use `release_manifest.json` for a file-level inventory of this result
payload.
