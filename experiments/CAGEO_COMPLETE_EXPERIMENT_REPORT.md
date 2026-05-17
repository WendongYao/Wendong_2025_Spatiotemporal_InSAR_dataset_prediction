# CAGEO Complete Experiment Report

Date: 2026-05-16
Project root: `C:\Users\Jupyter\Desktop\found_training_project`

## 1. Purpose

This report records the full experiment line executed in the standalone `found_training_project` bundle from the start of the CAGEO additional-experiment phase up to the current non-Transformer deep-learning exploration.

The report covers:

- dataset source and loading pipeline
- interpolation and split construction
- baseline and deep-model configurations
- executed experiment phases
- main quantitative results
- current recommended training configurations
- reproducibility assets and remaining gaps

This document is intended to be the single complete reference for the current experiment state.

## 2. Project Scope

The active project scope is the standalone bundle rooted at `found_training_project`, not the full original repository.

The active code and result pipeline has been built around these core files:

- `revision_config.py`
- `revision_utils.py`
- `revision_experiments.py`
- `deep_patch_models.py`
- `cg_additional_experiments.py`
- `run_cg_additional_suite.py`
- `run_deep_model_repair.py`
- `run_deep_model_round2.py`
- `run_nontransformer_round3.py`

The experiment line developed in five stages:

1. CAGEO additional suite execution
2. Deep-model repair after diagnosing broken early deep results
3. Round-2 deep architecture expansion
4. Round-3 non-Transformer hybrid exploration
5. Throughput-oriented tuning of the strongest CNN-LSTM hybrid

## 3. Dataset and Task Definition

### 3.1 Data source

The project uses the EGMS CSV:

- filename: `EGMS_L3_E32N34_100km_U_2018_2022_1.csv`
- resolved path during current runs:
  `C:\Users\Jupyter\Desktop\pytorch-tcn-main\EGMS_L3_E32N34_100km_U_2018_2022_1.csv`

This path is resolved through `RevisionConfig.resolve_csv_path()`.

### 3.2 Column usage

The core task definition is controlled by `RevisionConfig`:

- `history_start_col = 11`
- `history_length = 300`
- `target_col = 312`

That means:

- input history uses 300 displacement columns from 0-based columns `11:311`
- target uses 0-based column `312`

Spatial coordinates are read from:

- column `1`: easting
- column `2`: northing

### 3.3 Spatial task construction

The main dense-task builder is `build_dense_forecast_task()` in `revision_utils.py`.

Pipeline:

1. Read the CSV into a dataframe.
2. Read easting and northing coordinate columns.
3. Create a regular `grid_size x grid_size` query grid.
4. Interpolate each history column onto the grid.
5. Interpolate the target column onto the same grid.
6. Build validity masks for every interpolated input frame and the target frame.
7. Compute `history_coverage` per grid cell.
8. Build the eligible mask and then train/val/test split masks.

### 3.4 Default dense-task settings

From `revision_config.py`, the default project-wide task settings are:

| Item | Value |
|---|---|
| `grid_size` | `256` |
| `history_length` | `300` |
| `target_col` | `312` |
| default interpolation | `linear` |
| `min_history_coverage` | `0.0` |
| default split strategy | `spatial_tile` in the uploaded manuscript package; `random_pixel` is retained only for the E5 leakage probe |
| `tile_size` | `32` |
| split ratios | `train 0.70 / val 0.15 / test 0.15` |

### 3.5 Interpolation methods

Implemented interpolation methods:

- `linear`
- `nearest`
- `cubic`
- `idw`
- `rbf`

Implementation details:

- `linear`, `nearest`, `cubic` use `scipy.interpolate.griddata`
- `idw` uses a KD-tree nearest-neighbor weighted interpolation
- `rbf` uses `scipy.interpolate.RBFInterpolator`

Key interpolation settings in `RevisionConfig`:

| Item | Value |
|---|---|
| `idw_power` | `2.0` |
| `idw_neighbors` | `8` |
| `rbf_neighbors` | `64` |
| `rbf_smoothing` | `0.0` |
| `rbf_kernel` | `linear` |
| holdout fraction for interpolation test | `0.02` |
| holdout max points | `4000` |

### 3.6 Task caching

Dense tasks are cached under:

- `revision_outputs/_task_cache/`

The cache key depends on:

- csv path
- grid size
- history start column
- history length
- target column
- interpolation method
- minimum history coverage

This prevents repeated full-grid interpolation when rerunning the same task definition.

## 4. Split Construction and Data Loading Modes

### 4.1 Split strategies

Two split strategies are implemented:

- `random_pixel`
- `spatial_tile`

`spatial_tile` is the main leakage-aware evaluation protocol used in most serious experiments.

For `spatial_tile`:

- the grid is partitioned into non-empty `tile_size x tile_size` tiles
- tiles are shuffled by seed
- tiles are assigned to train, val, test to approximate `70/15/15`

### 4.2 Eligible mask and split masks

The dense task stores:

- `input_valid_mask`
- `target_valid_mask`
- `history_coverage`
- `eligible_mask`
- `train_mask`
- `val_mask`
- `test_mask`

These masks are saved to `split_masks.npz` in every experiment output folder.

### 4.3 Tabular loading mode

For `lasso`, `lightgbm`, `random_forest`, and some diagnostics:

- the dense input history tensor is flattened to per-pixel rows
- shape logic: `time x H x W -> (eligible_pixels, time)`
- the target is flattened to one scalar per eligible pixel

This is implemented by `build_tabular_dataset()` and `split_from_eligible_indices()`.

### 4.4 Patch loading mode

For repaired and later deep models:

- the dense task is converted into many patch-level training samples
- each patch stores:
  - normalized input sequence
  - normalized residual target
  - absolute target
  - last input frame
  - supervision mask
  - patch position

The deep models predict residuals rather than absolute target values.

### 4.5 Normalization

Patch-based deep models use train-domain normalization:

- input mean and std are computed from training pixels across all input history frames
- residual target mean and std are computed from the train-domain residual map

Residual target means:

- `residual = target_map - last_input_frame`

This normalized residual-learning setup was one of the key corrections that fixed the early deep-model failures.

## 5. Global Experiment Configuration

### 5.1 Base config defaults

Important shared defaults from `revision_config.py`:

| Item | Value |
|---|---|
| `cnn_hidden_dim` | `128` |
| `cnn_learning_rate` | `3e-4` |
| `cnn_weight_decay` | `1e-5` |
| `cnn_epochs` | `60` |
| `cnn_patience` | `12` |
| `patch_size` | `16` |
| `patch_stride` | `8` |
| `patch_min_valid_pixels` | `24` |
| `patch_batch_size` | `16` |
| `temporal_hybrid_recent_lags` | `8` |
| `temporal_hybrid_recent_scale_init` | `1.0` |
| `temporal_hybrid_correction_scale_init` | `0.1` |
| `lasso_alpha` | `1e-3` |
| `lasso_epochs` | `600` |
| `lasso_patience` | `60` |
| `lightgbm_num_boost_round` | `300` |
| `lightgbm_early_stopping_rounds` | `30` |
| `tcn_hidden_channels` | `64` |
| `tcn_num_layers` | `4` |
| `tcn_kernel_size` | `3` |
| `tcn_dropout` | `0.1` |
| `convlstm_hidden_dim` | `64` |
| `convlstm_num_layers` | `1` |
| `convlstm_kernel_size` | `3` |
| `nontransformer_hybrid_hidden_channels` | `64` |

Authoritative rerun defaults for the uploaded manuscript package are defined by `revision_config.py` and `configs/base_revision_config.json`. Some archived `config_snapshot.json` files under `revision_outputs/` are preserved as exact historical run artifacts from earlier exploration rounds and therefore may show earlier settings.

### 5.2 Device usage

Torch backends use:

- `cuda` if available
- otherwise `cpu`

LightGBM device type is resolved from:

- `auto`
- falls back to `cuda`, `gpu`, or `cpu` depending on support probe

The project also exports peak GPU memory where available.

### 5.3 Seeds

The main multi-seed set used throughout CAGEO additional experiments is:

- `42, 43, 44, 45, 46`

## 6. Experiment Scripts and Their Effective Settings

### 6.1 `run_cg_additional_suite.py`

Purpose:

- execute the CAGEO-aligned additional suite phases `E0` to `E7`, `E10`, `E11`

Key defaults:

| Item | Value |
|---|---|
| interpolation | `linear` |
| grid size | `256` |
| torch epochs | `50` |
| torch patience | `10` |
| lasso epochs | `600` |
| LightGBM rounds | `150` |
| LightGBM early stopping | `20` |
| RF trees | `100` |
| tile size | `32` |

Output root:

- `revision_outputs/cg_suite/`

### 6.2 `run_deep_model_repair.py`

Purpose:

- rerun the repaired `cnn_lstm_maskaware` and `cnn_tcn`

Key defaults:

| Item | Value |
|---|---|
| epochs | `80` |
| patience | `12` |
| patch size | `32` |
| patch stride | `16` |
| patch batch size | `8` |
| split | `spatial_tile` |

Output root:

- `revision_outputs/deep_model_repair/primary_multiseed/`

### 6.3 `run_deep_model_round2.py`

Purpose:

- second-round deep architecture search

Default model set:

- `temporal_channel_cnn`
- `patch_unet_residual`
- `conv3d_residual`
- `temporal_linear_hybrid`
- `conv_lstm_residual`

Key defaults:

| Item | Value |
|---|---|
| epochs | `60` |
| patience | `12` |
| learning rate | `3e-4` |
| weight decay | `1e-5` |
| patch size | `16` |
| patch stride | `8` |
| patch min valid pixels | `24` |
| patch batch size | `16` |
| split | `spatial_tile` |

### 6.4 `run_nontransformer_round3.py`

Purpose:

- focused follow-up on `CNN-LSTM / TCN` non-Transformer routes

Default model set:

- `cnn_lstm_hybrid`
- `cnn_tcn_hybrid`

Key defaults:

| Item | Value |
|---|---|
| epochs | `60` |
| patience | `12` |
| learning rate | `3e-4` |
| weight decay | `1e-5` |
| patch size | `16` |
| patch stride | `8` |
| patch min valid pixels | `24` |
| patch batch size | `16` |
| hidden channels | `64` |
| ConvLSTM layers | `2` by default |
| split | `spatial_tile` |

Later throughput tuning also tested:

- `cnn_lstm_hybrid`
- `convlstm_num_layers = 1`
- `patch_batch_size = 32`
- `learning_rate = 6e-4`

## 7. Model Configuration Summary

### 7.1 Classical and tree baselines

#### Persistence

- predicts the last observed frame as the target
- no trainable parameters

#### Linear trend

- simple temporal extrapolation baseline
- included in the main CAGEO suite

#### LASSO

Implementation:

- torch-based L1-regularized linear regressor
- trained on the same pixel split as all other aligned models
- standardizes train features and target internally
- performs alpha sweep across a small candidate set
- early stops on validation RMSE

Key config:

- `lasso_alpha = 1e-3`
- `lasso_epochs = 600`
- `lasso_patience = 60`
- `lasso_learning_rate = 2e-2`

#### Random forest

Key config:

- `n_estimators = 300` in `RevisionConfig`
- `100` trees in the default `run_cg_additional_suite.py` invocation

#### LightGBM

Implementation:

- same tabular split as LASSO
- early stopping on validation set
- optional SHAP export

Key config:

- `learning_rate = 0.05`
- `num_leaves = 31`
- `feature_fraction = 0.8`
- `bagging_fraction = 0.8`
- `bagging_freq = 5`
- `num_boost_round = 300` in config, `150` in default CAGEO suite invocation
- `early_stopping_rounds = 30` in config, `20` in default CAGEO suite invocation

### 7.2 Early aligned deep baselines

#### `cnn_lstm_maskaware`

- original aligned CNN-LSTM route used in the first CAGEO suite
- later shown to be unreliable due to sample-construction mismatch

#### `cnn_tcn`

- original aligned CNN + TCN route
- suffered from the same early sample-construction problem

### 7.3 Repaired patch-residual deep baselines

The key fix was:

- build many patch-level training samples instead of treating the full grid as one monolithic sample
- predict normalized residuals instead of raw absolute target maps
- preserve split masks consistently

These repaired models were:

- `cnn_lstm_maskaware`
- `cnn_tcn`

### 7.4 Round-2 architectures

#### `temporal_channel_cnn`

- flatten time and channel dimensions into 2D feature channels
- use residual 2D refinement blocks

#### `patch_unet_residual`

- U-Net-like encoder-decoder over flattened time-channel input
- predicts residual map directly

#### `conv3d_residual`

- 3D convolutional temporal encoder
- 2D spatial refinement head

#### `conv_lstm_residual`

- explicit ConvLSTM temporal aggregation over encoded patch frames
- 2D refinement head

#### `temporal_linear_hybrid`

Core idea:

- lasso-like linear shortcut over lag channels
- recent-lag gating over recent residual dynamics
- spatial correction branch

This model exists in two practical states:

- v1: early round-2 hybrid
- v2: warm-started and improved hybrid with recency-biased gating

### 7.5 Round-3 non-Transformer hybrids

#### `cnn_lstm_hybrid`

Core design:

- per-frame CNN encoder
- ConvLSTM temporal aggregation over encoded patch features
- lasso-warm-started linear shortcut
- recent-lag gating
- spatial correction decoder

Variants explored:

- 2-layer ConvLSTM
- 1-layer ConvLSTM
- faster 1-layer config with larger batch size and higher learning rate

#### `cnn_tcn_hybrid`

Core design:

- per-frame CNN encoder
- temporal TCN over encoded features
- lasso-warm-started linear shortcut
- recent-lag gating
- spatial correction decoder

## 8. Executed Experiment Phases and Main Results

### 8.1 E0 / E11 reproducibility and audit layer

Artifacts present:

- `README.md`
- `environment.yml`
- `configs/`
- `splits/`
- `scripts/reproduce_all.ps1`
- `scripts/reproduce_all.sh`
- `outputs/manifest.csv`

Checklist status:

- all expected items present, including `LICENSE` in the public repository package

Audit status:

- `metric_sanity_audit.csv`: `241` rows
- all `241` rows are `pass`

Manifest status:

- `outputs/manifest.csv` contains `3092` artifacts with SHA256 and size metadata

### 8.2 E1 / E2 main CAGEO multi-seed comparison

Summary file:

- `revision_outputs/cg_suite/E2_primary_multiseed/spatial_tile/grid_256/primary_multiseed_summary.csv`

Results:

| Model | RMSE mean | RMSE std | Notes |
|---|---:|---:|---|
| lasso | 1.3836 | 0.0603 | strongest baseline in the original suite |
| random_forest | 1.5229 | 0.1421 | decent baseline |
| persistence | 1.5589 | 0.0954 | strong naive baseline |
| linear_trend | 2.0025 | 0.0733 | weaker |
| lightgbm | 2.0629 | 0.4739 | weaker than expected |
| cnn_tcn | 7.3370 | 0.6128 | broken early deep result |
| cnn_lstm_maskaware | 7.3435 | 0.6093 | broken early deep result |

Interpretation:

- the first aligned deep results were not scientifically usable
- `lasso` emerged as the strongest baseline

### 8.3 E3 mask ablation

Summary file:

- `revision_outputs/cg_suite/E3_mask_ablation/spatial_tile/grid_256/mask_ablation_summary.csv`

Key findings:

- `noinputmask` was nearly identical to `maskaware`
- `nolossmask` looked numerically stronger but is not methodologically fair

### 8.4 E4 interpolation sensitivity

Summary files:

- `forecast_metric_summary.csv`
- `point_holdout_interpolation_summary.csv`

Forecast-table highlights:

| Method | Model | RMSE mean |
|---|---|---:|
| idw | lasso | 0.9034 |
| idw | persistence | 0.9967 |
| idw | lightgbm | 1.2821 |
| linear | lasso | 1.3836 |
| nearest | lasso | 1.7505 |
| rbf | lasso | 1.2349 |

Point-holdout interpolation RMSE:

| Method | RMSE mean |
|---|---:|
| idw | 4.2907 |
| rbf | 4.4116 |
| linear | 4.6574 |
| nearest | 5.3347 |

Interpretation:

- `idw` is the strongest interpolation choice observed so far

### 8.5 E5 split comparison

Summary file:

- `revision_outputs/cg_suite/E5_split_comparison/grid_256/split_comparison_summary.csv`

Key findings:

- random-pixel split was optimistic relative to spatial-tile split
- optimism inflation:
  - `lightgbm`: about `19.6%`
  - old `cnn_lstm_maskaware`: about `7.4%`
  - old `cnn_tcn`: about `7.4%`

Interpretation:

- `spatial_tile` should remain the main protocol

### 8.6 E7 resolution scaling

Summary file:

- `revision_outputs/cg_suite/E7_resolution_scaling/resolution_scaling_summary.csv`

Key findings:

- `lasso` remained strong across `128`, `256`, and `512`
- `lightgbm` remained weaker than `lasso`
- the old `cnn_lstm_maskaware` failed to provide a valid `512` result

Selected values:

| Grid | Model | RMSE mean |
|---|---|---:|
| 128 | lasso | 1.3160 |
| 256 | lasso | 1.3836 |
| 512 | lasso | 1.3686 |
| 128 | persistence | 1.4751 |
| 256 | persistence | 1.5589 |
| 512 | persistence | 1.5657 |

### 8.7 E10 interpretability

Summary file:

- `revision_outputs/cg_suite/E10_interpretability/spatial_tile/seed_42/persistence_similarity.csv`

Persistence similarity:

| Model | Correlation with persistence |
|---|---:|
| persistence | 1.0000 |
| lightgbm | 0.9680 |
| cnn_tcn | -0.0017 |
| cnn_lstm_maskaware | -0.0053 |

Interpretation:

- the old poor CNN results were structurally wrong, not just noisy persistence variants
- LightGBM behaved much more like a persistence refinement

### 8.8 Deep repair phase

Summary file:

- `revision_outputs/deep_model_repair/primary_multiseed/deep_repair_summary.csv`

Results:

| Model | RMSE mean | RMSE std |
|---|---:|---:|
| cnn_lstm_maskaware | 1.5591 | 0.0950 |
| cnn_tcn | 1.5593 | 0.0952 |

Interpretation:

- the deep repair fixed the major implementation problem
- repaired models recovered from `~7.34` RMSE to `~1.56`

### 8.9 Round-2 architecture search

Summary files:

- `revision_outputs/deep_model_round2_multiseed/round2_summary.csv`
- `revision_outputs/deep_model_round2_hybrid_v2_multiseed/round2_summary.csv`
- `revision_outputs/deep_model_round2_convlstm_multiseed/round2_summary.csv`

Round-2 results:

| Model | RMSE mean | RMSE std |
|---|---:|---:|
| temporal_linear_hybrid v2 | 1.3602 | 0.0917 |
| patch_unet_residual | 1.4165 | 0.0825 |
| temporal_linear_hybrid v1 | 1.4361 | 0.0835 |
| temporal_channel_cnn | 1.4503 | 0.0830 |
| conv_lstm_residual | 1.5067 | 0.0936 |

Interpretation:

- warm-started hybrid modeling was clearly useful
- `temporal_linear_hybrid v2` was the first deep model to beat the current `lasso` baseline on 5 seeds

### 8.10 Round-3 non-Transformer exploration

Summary files:

- `revision_outputs/nontransformer_round3_cnnlstm_l1_5seed/combined_summary.csv`
- `revision_outputs/nontransformer_round3_cnntcn_multiseed/round3_summary.csv`
- `revision_outputs/nontransformer_round3_cnnlstm_3seed/round3_summary.csv`

Results:

| Model | Seed count | RMSE mean | RMSE std | Runtime mean (s) |
|---|---:|---:|---:|---:|
| cnn_lstm_hybrid, 2-layer | 3 | 1.3049 | 0.0714 | 1074.5 |
| cnn_lstm_hybrid, 1-layer | 5 | 1.3077 | 0.0927 | 529.9 |
| cnn_tcn_hybrid | 5 | 1.3782 | 0.0685 | 61.6 |

Interpretation:

- this phase established the current strongest deep route
- `cnn_lstm_hybrid (1-layer)` is the strongest fully repeated deep result
- `cnn_tcn_hybrid` is nearly as strong and dramatically cheaper

### 8.11 Throughput tuning of `cnn_lstm_hybrid (1-layer)`

Summary files:

- `revision_outputs/nontransformer_round3_cnnlstm_l1_5seed/combined_summary.csv`
- `revision_outputs/nontransformer_round3_cnnlstm_l1_bs32_lr6e4_5seed/round3_summary.csv`

Comparison:

| Config | RMSE mean | RMSE std | Runtime mean (s) | Peak GPU memory (MB) |
|---|---:|---:|---:|---:|
| `bs16, lr=3e-4` | 1.3077 | 0.0927 | 529.9 | 1545.7 |
| `bs32, lr=6e-4` | 1.3140 | 0.0654 | 309.9 | 3065.3 |

Interpretation:

- the `bs32, lr=6e-4` setting is about `1.71x` faster
- the RMSE penalty is very small
- this is the current recommended fast experiment configuration

## 9. Current Leaderboard

Under the aligned `grid_256 + spatial_tile` setting, the strongest current results are:

| Rank | Model | RMSE mean | RMSE std | Status |
|---|---|---:|---:|---|
| 1 | `cnn_lstm_hybrid` (1-layer, `bs16, lr=3e-4`) | 1.3077 | 0.0927 | strongest 5-seed repeated deep result |
| 2 | `cnn_lstm_hybrid` (2-layer) | 1.3049 | 0.0714 | stronger but only 3 seeds so far |
| 3 | `cnn_lstm_hybrid` (1-layer, `bs32, lr=6e-4`) | 1.3140 | 0.0654 | fastest near-lossless configuration |
| 4 | `temporal_linear_hybrid v2` | 1.3602 | 0.0917 | strongest round-2 hybrid |
| 5 | `cnn_tcn_hybrid` | 1.3782 | 0.0685 | strongest repeated TCN route |
| 6 | `lasso` | 1.3836 | 0.0603 | strongest classical baseline |
| 7 | `patch_unet_residual` | 1.4165 | 0.0825 | strongest non-hybrid round-2 model |

## 10. Current Recommended Configurations

### 10.1 Best-quality deep configuration

Recommended when the highest repeated accuracy matters most:

- model: `cnn_lstm_hybrid`
- ConvLSTM layers: `1`
- epochs: `60`
- patience: `12`
- learning rate: `3e-4`
- weight decay: `1e-5`
- patch size: `16`
- patch stride: `8`
- patch min valid pixels: `24`
- patch batch size: `16`
- hidden channels: `64`
- split: `spatial_tile`
- interpolation: `linear`

### 10.2 Fast deep configuration

Recommended when many reruns are needed:

- model: `cnn_lstm_hybrid`
- ConvLSTM layers: `1`
- epochs: `60`
- patience: `12`
- learning rate: `6e-4`
- weight decay: `1e-5`
- patch size: `16`
- patch stride: `8`
- patch min valid pixels: `24`
- patch batch size: `32`
- hidden channels: `64`

### 10.3 Best repeated TCN-style configuration

- model: `cnn_tcn_hybrid`
- epochs: `60`
- patience: `12`
- learning rate: `3e-4`
- weight decay: `1e-5`
- patch size: `16`
- patch stride: `8`
- patch min valid pixels: `24`
- patch batch size: `16`
- hidden channels: `64`
- TCN layers: `4`
- kernel size: `3`
- dropout: `0.1`

## 11. Reproducibility Assets

The project currently includes:

- `README.md`
- `environment.yml`
- `configs/`
- `splits/`
- `scripts/reproduce_all.ps1`
- `scripts/reproduce_all.sh`
- `outputs/manifest.csv`
- `outputs/metric_sanity_audit.csv`
- `outputs/reproducibility_checklist.csv`

Environment packaging:

- conda environment name: `found_training_project`
- Python version: `3.11`
- pip dependencies loaded from:
  - `requirements-revision.txt`
  - `requirements-revision-optional.txt`

## 12. Remaining Gaps

The following items are still incomplete or outside the current standalone bundle scope:

1. `E8` temporal or cross-region generalization is not fully implemented in this standalone project.
2. `E9` uncertainty calibration is not fully implemented as a completed experiment block.
3. The strongest 2-layer `cnn_lstm_hybrid` result has only been run on 3 seeds, not the full 5-seed set.
4. `LICENSE` is now present in the public repository package; a stable DOI release is still pending.

## 13. Final Takeaways

1. The original aligned deep-learning results in the first CAGEO suite were invalid as evidence of deep-model capability because the sample-construction and training formulation were wrong.
2. The patch-residual repair corrected the main deep-learning failure mode.
3. Round-2 showed that warm-started hybrid modeling was a strong direction.
4. Round-3 established that the non-Transformer route is now the strongest family in the project.
5. The strongest repeated current result is `cnn_lstm_hybrid (1-layer)`.
6. A fast near-lossless training configuration now exists for `cnn_lstm_hybrid`, making future reruns much cheaper.
7. `lasso` remains a very strong baseline and should stay in every main comparison.
8. `idw` is currently the most promising interpolation option for the next round of top-model reruns.

## 14. Key Files to Read First

If someone needs the most important current files only, read:

1. `CAGEO_COMPLETE_EXPERIMENT_REPORT.md`
2. `CURRENT_EXPERIMENT_REPORT.md`
3. `revision_outputs/nontransformer_round3_cnnlstm_l1_5seed/combined_summary.csv`
4. `revision_outputs/nontransformer_round3_cnnlstm_l1_bs32_lr6e4_5seed/round3_summary.csv`
5. `revision_outputs/nontransformer_round3_cnntcn_multiseed/round3_summary.csv`
6. `revision_outputs/deep_model_round2_hybrid_v2_multiseed/round2_summary.csv`
7. `revision_outputs/cg_suite/E4_interpolation_sensitivity/spatial_tile/grid_256/forecast_metric_summary.csv`
