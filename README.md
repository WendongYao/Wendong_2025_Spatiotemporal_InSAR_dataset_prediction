# Wendong_2025_Spatiotemporal_InSAR_dataset_prediction

This repository hosts the CAGEO reproduction package for:

`A Reproducible Sparse-to-Dense InSAR Computing Pipeline with Hybrid CNN-LSTM Forecasting`

The versioned repository contents are intentionally limited to the files needed
to install the software, understand the workflow, run the paper experiments, and
recreate the manuscript analysis figures.

## License

This repository includes a clear open-source license:

- `LICENSE`

## Repository layout

- `experiments/`
  CAGEO code, documentation, environment files, synthetic test case, and figure-generation utilities

## Start here

- `experiments/README.md`
- `experiments/REPRODUCTION_INSTRUCTIONS.md`
- `experiments/USER_GUIDE.md`
- `experiments/TUTORIAL.md`
- `experiments/COMPUTATIONAL_REQUIREMENTS.md`
- `experiments/CAGEO_COMPLETE_EXPERIMENT_REPORT.md`

## Quick start

```powershell
cd experiments
conda env create -f environment.yml
conda activate found_training_project
pip install -r requirements-revision.txt
python .\smoke_test_revision.py --skip-csv-check
python .\run_synthetic_smoke_case.py
```

The synthetic smoke case is included so that the software can be exercised even
when the external EGMS source CSV is not bundled with the repository.

## Data availability note

The paper uses an external EGMS CSV that is not redistributed in this
repository. The repository therefore includes:

- the full code used for the paper workflow
- documentation for where the real CSV should be placed
- a versioned synthetic CSV and generator for test cases

See `experiments/datasets/README.md` and `experiments/examples/README.md`.
