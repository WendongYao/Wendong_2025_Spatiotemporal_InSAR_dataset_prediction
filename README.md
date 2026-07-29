# Native-Support Sparse-to-Dense EGMS Forecasting

This repository hosts the reproducibility package for:

`Native-Support Sparse-to-Dense EGMS Forecasting with Anchored Neural Residuals`

Release `v2.2.0` makes the EGMS L3 native product-cell task explicit and adds:
an exact product audit; persistence, dated linear-trend, and DLinear baselines;
provenance-complete five-split LASSO/SPAR reruns; multi-resolution support
evaluation; product-quality stratification; corrected repeated-holdout
statistics; and refreshed paper figures. The original gridding-first CAGEO
package remains under `experiments/` for traceability.

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
- `RELEASE_NOTES_v2.2.0.md`

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

The historical CLI names beginning with `saqr` are retained so archived
artifacts remain verifiable. In the paper and documentation the method is named
**SPAR**. The authoritative E32N34 seed-43--46 frozen variant is
`saqr_no_global_coord`; seed 42 and the original external/analytic artifacts use
`saqr_point_query`. The release manifests preserve these identities explicitly.

## Data and scope

The primary study uses the E32N34 EGMS L3 Ortho product at one shared
300-history-to-one-target forecast origin. Rows are valid cells of a partially
populated native 100-m product lattice, not raw persistent scatterers. The
primary endpoint is product-level predictive consistency at held-out valid
cells; it is not independent validation of geodetic truth. Earlier external-tile
controls remain in the package as development history but are not used to make
a cross-region generalization claim. Transformer, STGCN, graph-model, and RSASE
experiments are outside this release.

The source EGMS products are not redistributed in the Git repository. The
versioned archive is available through the Zenodo concept DOI
[`10.5281/zenodo.20299645`](https://doi.org/10.5281/zenodo.20299645).

## License

See `LICENSE`.
