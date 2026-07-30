# Release v2.3.0

This release aligns the public package with the manuscript
*Native-Support Forecasting of Delivered EGMS Level-3 Displacement Values*.

## Final-configuration evidence

- Uniform all-cell training replaces the inherited capped patch sampler.
- Partitions 47--50 provide a pre-specified held-out E32N34 evaluation of
  persistence, DLinear, LASSO, a causal TCN, and SPAR.
- Three additional tiles are rerun at the same forecast origin with the final
  sampler and independently trained within-tile models.
- Four 240-history origins provide a negative temporal counter-test.
- Ten analytic realizations quantify matched-IDW pseudo-target optimism.
- A final-configuration anchor ablation reports a modest, directionally
  consistent effect whose corrected interval crosses zero.
- New aggregate manifests, figure scripts, direct predictions, resolved
  configurations, and source/output hashes are included.

## Interpretation

The primary endpoint is the next delivered EGMS Level-3 value at a valid
product cell. It is not independent geodetic truth. Partitions 47--50 overlap,
the external tile experiments are not transfer tests, and the temporal
counter-test changes both origin and history length. The release therefore does
not claim Europe-wide, temporal, or physical dense-field superiority.

The v2.0--v2.2 history remains auditable. Transformer, STGCN, graph-model, and
RSASE experiments remain outside this release and manuscript.
