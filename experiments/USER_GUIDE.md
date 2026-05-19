# User Guide

This guide describes the main inputs, outputs, options, and expected behavior
of the scripts included in the CAGEO reproduction package.

## Common input

Most runners accept:

- `--csv-path`
  Explicit path to the real EGMS CSV or to the bundled synthetic CSV

If `--csv-path` is omitted, the code searches for:

- `experiments/EGMS_L3_E32N34_100km_U_2018_2022_1.csv`
- `experiments/datasets/EGMS_L3_E32N34_100km_U_2018_2022_1.csv`
- a small set of neighboring fallback locations defined in `revision_config.py`

## Common output behavior

Most experiment runners create:

- model-specific output directories under `revision_outputs/` or a user-selected `--output-root`
- `metrics.json` and `metrics.csv`
- split masks or cached task files when needed
- diagnostic plots such as scatter plots, residual plots, and binned-error summaries

The scripts do not modify the input CSV.

## Script reference

### `smoke_test_revision.py`

Purpose:
- Check whether core and optional Python dependencies are available
- Optionally test whether a real or synthetic CSV path resolves correctly

Key options:
- `--strict-core`
- `--strict-all`
- `--csv-path`
- `--skip-csv-check`

Expected behavior:
- prints a JSON payload describing imports and data-path readiness
- exits with code `1` only when strict mode is requested and required dependencies are missing

### `preflight_revision.py`

Purpose:
- Build the task once and report the dense-map geometry and split sizes before long runs

Key options:
- `--csv-path`
- `--interpolation`
- `--split-seed`

Expected behavior:
- prints a JSON payload with input shape, target shape, and train/validation/test pixel counts
- creates task-cache files if caching is enabled

### `run_cg_additional_suite.py`

Purpose:
- Run the main additional-suite experiments described in the report

Key options:
- `--phase`
- `--csv-path`
- `--interpolation`
- `--grid-size`
- `--epochs`
- `--patience`
- `--lasso-epochs`
- `--num-boost-round`
- `--early-stopping-rounds`
- `--random-forest-n-estimators`
- `--lightgbm-device-type`
- `--tile-size`

Expected behavior:
- runs one or more paper experiment phases
- writes summaries under `revision_outputs/cg_suite/`
- writes audit and reproducibility files under `outputs/`

### `run_deep_model_repair.py`

Purpose:
- Reproduce the repaired patch-residual deep-learning experiments

Key options:
- `--csv-path`
- `--epochs`
- `--patience`
- `--learning-rate`
- `--patch-size`
- `--patch-stride`
- `--patch-batch-size`

Expected behavior:
- runs the repaired deep models and writes summaries under `revision_outputs/deep_model_repair/`

### `run_deep_model_round2.py`

Purpose:
- Run the round-2 architecture search on the repaired deep pipeline

Key options:
- `--csv-path`
- `--epochs`
- `--patience`
- `--models`
- `--seeds`
- `--output-root`

Expected behavior:
- writes `round2_seed_level.csv` and `round2_summary.csv`
- creates model-specific subdirectories containing metrics and plots

### `run_nontransformer_round3.py`

Purpose:
- Run the round-3 non-Transformer exploration, including Hybrid CNN-LSTM and Hybrid CNN-TCN

Key options:
- `--csv-path`
- `--epochs`
- `--patience`
- `--learning-rate`
- `--patch-batch-size`
- `--models`
- `--seeds`
- `--output-root`

Expected behavior:
- writes `round3_seed_level.csv` and `round3_summary.csv`
- saves model metrics under the selected output root

### `run_synthetic_smoke_case.py`

Purpose:
- Run a small end-to-end test on the bundled synthetic dataset

Key options:
- `--csv-path`
- `--output-root`
- `--interpolation`
- `--grid-size`
- `--epochs`

Expected behavior:
- validates task construction
- runs a small one-seed benchmark with persistence, LASSO, LightGBM, and Hybrid CNN-LSTM
- writes compact outputs under `synthetic_smoke_outputs/`

### `build_cageo_analysis_figures.py`

Purpose:
- Recreate the manuscript analysis figures from existing experiment outputs

Expected inputs:
- the relevant `revision_outputs/` summaries and diagnostics already exist locally

Expected behavior:
- writes figures under `cageo_submission_assets/figures/`
- does not recreate `fig01` or `fig02`

## Main configuration file

`revision_config.py` centralizes:

- dataset column assumptions
- grid size
- split ratios
- patch extraction parameters
- classical-model hyperparameters
- deep-model hyperparameters
- interpolation settings

If you need to change the global task definition, start there.
