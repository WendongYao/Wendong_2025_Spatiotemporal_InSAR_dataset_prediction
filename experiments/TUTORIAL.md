# Tutorial

This tutorial covers the two most common use cases:

- a quick functional test with the bundled synthetic dataset
- a full rerun with the real EGMS CSV

## Use case 1: quick functional test

Goal:
- confirm that the package installs correctly
- confirm that the task builder, baselines, and final Hybrid CNN-LSTM path run end to end

Steps:

```powershell
cd experiments
conda env create -f environment.yml
conda activate found_training_project
pip install -r requirements-revision.txt
python .\smoke_test_revision.py --skip-csv-check
python .\run_synthetic_smoke_case.py
```

Expected outputs:

- `synthetic_smoke_outputs/smoke_summary.json`
- model subdirectories under `synthetic_smoke_outputs/`

What this verifies:

- imports
- task construction
- one-seed execution of persistence, LASSO, LightGBM, and Hybrid CNN-LSTM

## Use case 2: full paper rerun

Goal:
- regenerate the manuscript-scale outputs on the real EGMS CSV

Steps:

```powershell
cd experiments
conda activate found_training_project
python .\preflight_revision.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
python .\run_cg_additional_suite.py --phase all --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
python .\run_deep_model_repair.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
python .\run_deep_model_round2.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv --models temporal_channel_cnn patch_unet_residual conv_lstm_residual temporal_linear_hybrid --output-root revision_outputs/deep_model_round2
python .\run_nontransformer_round3.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv --models cnn_lstm_hybrid cnn_tcn_hybrid --output-root revision_outputs/nontransformer_round3
python .\build_cageo_analysis_figures.py
```

Expected outputs:

- `revision_outputs/`
- `outputs/`
- `cageo_submission_assets/figures/`

## Use case 3: rebuild only the analysis figures

Goal:
- recreate manuscript analysis figures after the experiment outputs already exist

Steps:

```powershell
cd experiments
conda activate found_training_project
python .\build_cageo_analysis_figures.py
```

Expected outputs:

- figure files under `cageo_submission_assets/figures/`

## Troubleshooting

If a command fails immediately:

- run `python .\smoke_test_revision.py --skip-csv-check`
- confirm that `torch`, `lightgbm`, and the numerical stack are installed

If the real EGMS CSV is not found:

- pass `--csv-path` explicitly
- or place the file in one of the locations described in `datasets/README.md`
