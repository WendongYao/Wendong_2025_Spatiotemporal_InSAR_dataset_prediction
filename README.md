# Native-Support Forecasting of Delivered EGMS Level-3 Values

This repository hosts the reproducibility package for:

`Native-Support Forecasting of Delivered EGMS Level-3 Displacement Values`

Release `v2.3.0` adds the final all-cell SPAR configuration, pre-specified
partitions 47--50, a same-support causal TCN, same-origin within-tile
replication, a shortened-history temporal counter-test, ten analytic
matched-interpolation realizations, and final sampler/anchor ablations. The
original gridding-first package remains under `experiments/` for traceability.

## Repository layout

- `experiments/`: original CAGEO code, environment, documentation, and smoke data.
- `experiments_ext/`: support-preserving model, native-product-cell evaluation,
  baselines, controls, aggregation, and audit utilities.
- `results/spar_v2/`: sanitized aggregates, manifests, predictions, checkpoints,
  controlled diagnostics, and integrity hashes.
- `paper_figures/`: publication figure scripts and rendered PDF/PNG figures.
- `tests/`: boundary and protocol regression tests for the extension.

## Start here

- `experiments/REPRODUCTION_INSTRUCTIONS.md`
- `experiments_ext/README.md`
- `results/spar_v2/README.md`
- `paper_figures/README.md`
- `RELEASE_NOTES_v2.3.0.md`

## Quick start

```powershell
cd experiments
conda env create -f environment.yml
conda activate found_training_project
pip install -r requirements-revision.txt
python .\smoke_test_revision.py --skip-csv-check
python .\run_synthetic_smoke_case.py
```

From the repository root, the support-preserving analytic task can be exercised
without an EGMS CSV:

```powershell
python .\experiments_ext\run_saqr_synthetic_truth.py `
  --output-root .\results\_local\synthetic_composite `
  --scenario composite --input-interpolation idw `
  --grid-size 64 --support-points 1024 --seed 42
```

Historical CLI names beginning with `saqr` are retained so archived artifacts
remain verifiable. The final v2.3 implementation is
`direct_spar_all_cells_uniform`; in the paper and documentation it is named
**SPAR**. The release manifests preserve both current and historical identities.

## Data and scope

The primary study uses E32N34 at one 300-history-to-one-target origin. Rows are
valid cells of a partially populated native 100-m product lattice, not raw
persistent scatterers. The primary endpoint is product-level predictive
consistency, not independent geodetic truth. Partitions 47--50 overlap; the
three additional tiles are independently trained same-origin replications, not
transfer tests; and the 240-history origins are a counter-test, not temporal
confirmation. Transformer, STGCN, graph-model, and RSASE experiments are
outside this release.

The source EGMS products are not redistributed in the Git repository. The
versioned archive is available through the Zenodo concept DOI
[`10.5281/zenodo.20299645`](https://doi.org/10.5281/zenodo.20299645).

## License

See `LICENSE`.
