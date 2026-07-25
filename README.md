# Support-Preserving Sparse-to-Dense InSAR Forecasting

This repository hosts the reproducibility package for:

`Support-Preserving Sparse-to-Dense InSAR Forecasting with Anchored Neural Residuals`

Release `v2.1.0` adds the reviewer-priority validation and provenance package:
corrected repeated-holdout statistics, five-partition external-tile replications,
a training-count-matched 2-km buffer control, LASSO prediction backfills, an
interpolated-pseudo-target confound experiment, resolved configuration and plot
manifests, and refreshed paper figures. The original gridding-first CAGEO package
remains under `experiments/` for traceability.

## Repository layout

- `experiments/`: original CAGEO code, environment, documentation, and smoke data.
- `experiments_ext/`: support-preserving model, raw-observation evaluation,
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
- `RELEASE_NOTES_v2.1.0.md`

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

The study uses the original E32N34 task and external tiles E29N33, E36N31, and
E37N41 at one shared 300-history-to-one-target forecast origin. Each tile is
trained independently; these are same-origin spatial replications, not transfer
experiments. Transformer, STGCN, graph-model, and RSASE experiments are outside
this release.

The source EGMS products are not redistributed in the Git repository. The
versioned archive is available through the Zenodo concept DOI
[`10.5281/zenodo.20299645`](https://doi.org/10.5281/zenodo.20299645).

## License

See `LICENSE`.
