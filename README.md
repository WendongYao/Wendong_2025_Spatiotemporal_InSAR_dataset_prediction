# Support-Preserving Sparse-to-Dense InSAR Forecasting

This repository hosts the reproducibility package for:

`Support-Preserving Sparse-to-Dense InSAR Forecasting with Anchored Neural Residuals`

Release `v2.0.0` adds SPAR, the measurement-support-first evaluation protocol,
frozen/sanitized run manifests, compact prediction artifacts, known-truth tests,
reviewer-requested ablations, and the final paper figure scripts. The original
gridding-first CAGEO package remains under `experiments/` for traceability.

## License

This repository includes a clear open-source license:

- `LICENSE`

## Repository layout

- `experiments/`
  Original CAGEO code, documentation, environment files, and synthetic smoke case
- `experiments_ext/`
  SPAR model, raw-observation protocol, baselines, ablations, and known-truth experiments
- `results/spar_v2/`
  Sanitized aggregates, manifests, prediction artifacts, checkpoints, and integrity hashes
- `paper_figures/`
  Publication figure scripts and rendered PDF figures

## Start here

- `experiments/README.md`
- `experiments/REPRODUCTION_INSTRUCTIONS.md`
- `experiments/USER_GUIDE.md`
- `experiments/TUTORIAL.md`
- `experiments/COMPUTATIONAL_REQUIREMENTS.md`
- `experiments/CAGEO_COMPLETE_EXPERIMENT_REPORT.md`
- `experiments_ext/README.md`
- `results/spar_v2/README.md`
- `RELEASE_NOTES_v2.0.0.md`

## Quick start

```powershell
cd experiments
conda env create -f environment.yml
conda activate found_training_project
pip install -r requirements-revision.txt
python .\smoke_test_revision.py --skip-csv-check
python .\run_synthetic_smoke_case.py
```

To exercise the support-preserving extension without the EGMS CSV:

```powershell
python .\experiments_ext\run_saqr_synthetic_truth.py `
  --output-root .\results\_local\synthetic_composite `
  --scenario composite --input-interpolation idw `
  --grid-size 64 --support-points 1024 --seed 42
```

The command retains the historical `saqr` filename and machine identifier used
by the frozen manifests. In the paper and public documentation, the method is
named **SPAR** (support-preserving anchored residual forecasting).

The synthetic smoke case is included so that the software can be exercised even
when the external EGMS source CSV is not bundled with the repository.

## Data availability note

The paper uses an external EGMS CSV that is not redistributed in this
repository. The repository therefore includes:

- the full code used for the paper workflow
- documentation for where the real CSV should be placed
- a versioned synthetic CSV and generator for test cases

See `experiments/datasets/README.md` and `experiments/examples/README.md`.

The versioned data archive is available through the Zenodo concept DOI
[`10.5281/zenodo.20299645`](https://doi.org/10.5281/zenodo.20299645).
