# CAGEO Reproduction Instructions

This file describes the exact command flow for reproducing the main paper
results and for running the bundled synthetic test case.

## 1. Prepare the environment

Recommended:

```powershell
conda env create -f environment.yml
conda activate found_training_project
pip install -r requirements-revision.txt
```

Optional SHAP dependency:

```powershell
pip install -r requirements-revision-optional.txt
```

If you want CUDA acceleration for the PyTorch-based models, install the
appropriate PyTorch build for your system before long runs. The repository code
also works on CPU, but CPU runtimes are longer.

## 2. Run an installation smoke test

Dependency-only check:

```powershell
python .\smoke_test_revision.py --skip-csv-check
```

If you already have a real or synthetic CSV available:

```powershell
python .\smoke_test_revision.py --csv-path C:\path\to\your.csv
```

## 3. Run the bundled synthetic test case

The repository includes a versioned synthetic CSV and a small smoke runner. This
is the recommended first execution when the real EGMS CSV is not available.

```powershell
python .\run_synthetic_smoke_case.py
```

This command writes outputs under:

- `synthetic_smoke_outputs/`

The synthetic test case is meant to validate installation and workflow behavior.
It is not a substitute for the real manuscript benchmark.

## 4. Place the real EGMS CSV

Expected manuscript CSV filename:

- `EGMS_L3_E32N34_100km_U_2018_2022_1.csv`

Supported placement options:

- `experiments/EGMS_L3_E32N34_100km_U_2018_2022_1.csv`
- `experiments/datasets/EGMS_L3_E32N34_100km_U_2018_2022_1.csv`
- any external path passed through `--csv-path`

See `datasets/README.md` for the data note.

## 5. Run preflight checks on the real data

```powershell
python .\preflight_revision.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
```

This confirms:

- input history length
- target map size
- valid-pixel counts
- train/validation/test split sizes

## 6. Reproduce the main paper experiment stages

### 6.1 Additional CAGEO suite

```powershell
python .\run_cg_additional_suite.py --phase all --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
```

This regenerates the additional-suite outputs described in the report, including:

- primary baseline benchmark
- mask ablation
- interpolation sensitivity
- split comparison
- resolution scaling
- interpretability diagnostics
- audit files
- reproducibility-pack summaries

### 6.2 Deep-model repair

```powershell
python .\run_deep_model_repair.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
```

### 6.3 Round-2 architecture search

```powershell
python .\run_deep_model_round2.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv --models temporal_channel_cnn patch_unet_residual conv_lstm_residual temporal_linear_hybrid --output-root revision_outputs/deep_model_round2
```

### 6.4 Round-3 non-Transformer exploration

```powershell
python .\run_nontransformer_round3.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv --models cnn_lstm_hybrid cnn_tcn_hybrid --output-root revision_outputs/nontransformer_round3
```

### 6.5 Optional fast rerun for the 1-layer Hybrid CNN-LSTM

```powershell
python .\run_nontransformer_round3.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv --models cnn_lstm_hybrid --patch-batch-size 32 --learning-rate 6e-4 --output-root revision_outputs/nontransformer_round3_fast
```

## 7. Validate the regenerated outputs

Check these generated summaries after the runs complete:

- `revision_outputs/cg_suite/E2_primary_multiseed/spatial_tile/grid_256/primary_multiseed_summary.csv`
- `revision_outputs/cg_suite/E3_mask_ablation/spatial_tile/grid_256/mask_ablation_summary.csv`
- `revision_outputs/cg_suite/E4_interpolation_sensitivity/spatial_tile/grid_256/forecast_metric_summary.csv`
- `revision_outputs/cg_suite/E5_split_comparison/grid_256/split_comparison_summary.csv`
- `revision_outputs/cg_suite/E7_resolution_scaling/resolution_scaling_summary.csv`
- `revision_outputs/deep_model_repair/primary_multiseed/deep_repair_summary.csv`
- `revision_outputs/deep_model_round2/round2_summary.csv`
- `revision_outputs/nontransformer_round3/round3_summary.csv`
- `revision_outputs/nontransformer_round3_fast/round3_summary.csv`

The expected experiment line and reference metrics are documented in:

- `CAGEO_COMPLETE_EXPERIMENT_REPORT.md`

## 8. Rebuild the manuscript analysis figures

After the experiment outputs exist locally:

```powershell
python .\build_cageo_analysis_figures.py
```

This creates:

- `cageo_submission_assets/figures/`

Covered figures:

- `fig03_main_benchmark_rmse`
- `fig04_prediction_residual_maps`
- `fig05_error_diagnostics`
- `fig06_interpolation_sensitivity`
- `fig07_runtime_rmse_tradeoff`
- `fig08_persistence_similarity_or_shap`
- `fig09_study_area_timeseries`
- `figS01_resolution_scaling`
- `figS02_split_leakage`

The script intentionally does not recreate `fig01` or `fig02`.
