# CAGEO Experiment Package

This directory is the paper-aligned experiment bundle for:

`A Reproducible Sparse-to-Dense InSAR Computing Pipeline with Hybrid CNN-LSTM Forecasting`

It is a curated upload from the local `found_training_project` workspace and keeps only the assets needed to match the current manuscript line.

## What is included

- Core experiment code:
  - `revision_config.py`
  - `revision_utils.py`
  - `revision_experiments.py`
  - `deep_patch_models.py`
  - `cg_additional_experiments.py`
- Main experiment runners:
  - `run_cg_additional_suite.py`
  - `run_deep_model_repair.py`
  - `run_deep_model_round2.py`
  - `run_nontransformer_round3.py`
- Reproducibility support:
  - `configs/`
  - `splits/`
  - `outputs/`
  - `scripts/`
  - `environment.yml`
  - `requirements-revision.txt`
  - `requirements-revision-optional.txt`
- Paper assets:
  - `cageo_submission_assets/`
  - `build_cageo_figures_tables_tex.py`
- Curated result bundle:
  - `revision_outputs/`
  - representative seed-42 outputs used by the manuscript figures

## Canonical manuscript settings

The uploaded defaults are aligned to the current manuscript main line:

- `grid_size = 256`
- `split_strategy = spatial_tile`
- `tile_size = 32`
- `history_length = 300`
- `target_col = 312`
- `interpolation_method = linear`
- `patch_size = 16`
- `patch_stride = 8`
- `patch_min_valid_pixels = 24`
- `patch_batch_size = 16`
- `cnn_learning_rate = 3e-4`
- `cnn_weight_decay = 1e-5`
- `cnn_epochs = 60`
- `cnn_patience = 12`
- `convlstm_num_layers = 1`

## Main manuscript results represented here

- Hybrid CNN-LSTM (1-layer, 5 seeds): `RMSE 1.3077 +/- 0.0927`
- Hybrid CNN-TCN (5 seeds): `RMSE 1.3782 +/- 0.0685`
- LASSO (5 seeds): `RMSE 1.3836 +/- 0.0603`
- Fast Hybrid CNN-LSTM rerun (`bs32`, `lr=6e-4`): `RMSE 1.3140 +/- 0.0654`

## Important consistency note

The current manuscript text still contains one wording item that should be revised before submission:

- the manuscript describes the `LICENSE` status as pending, but this public repository already contains a `LICENSE` file

Archived `config_snapshot.json` files inside `revision_outputs/` are preserved as exact historical run artifacts. The authoritative manuscript rerun defaults for this upload are `revision_config.py` and `configs/base_revision_config.json`.

Large deep-model checkpoint binaries (`*.pth`) are intentionally omitted from the GitHub upload to keep the public package within standard repository size limits. All manuscript-facing metrics, summaries, figures, split definitions, and configuration snapshots are retained.

## Data note

The EGMS CSV used by the current manuscript is not committed here. See `datasets/README.md` for the expected filename and placement.
