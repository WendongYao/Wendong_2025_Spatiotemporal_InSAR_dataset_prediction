# CAGEO Reproduction Instructions

This repository tracks only the files required to rerun the CAGEO training and validation pipeline. Generated result folders are created locally and are not committed.

## 1. Prepare the dataset

Expected CSV filename:

- `EGMS_L3_E32N34_100km_U_2018_2022_1.csv`

Recommended placement:

- `experiments/datasets/EGMS_L3_E32N34_100km_U_2018_2022_1.csv`

You can also keep the CSV elsewhere and pass `--csv-path`.

## 2. Create the environment

Recommended:

```powershell
conda env create -f environment.yml
conda activate py311
pip install -r requirements-revision.txt
pip install -r requirements-revision-optional.txt
```

`requirements-revision-optional.txt` is needed for optional SHAP-based interpretability outputs.

## 3. Run sanity checks

```powershell
python .\smoke_test_revision.py --strict-all
python .\preflight_revision.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
```

## 4. Run the experiment stages

### 4.1 Additional CAGEO suite

```powershell
python .\run_cg_additional_suite.py --phase all --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
```

This regenerates the additional-suite outputs described in the report, including the primary baseline benchmark, mask ablation, interpolation sensitivity, split comparison, resolution scaling, interpretability diagnostics, audit files, and the reproduction pack.

### 4.2 Deep-model repair

```powershell
python .\run_deep_model_repair.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
```

### 4.3 Round-2 deep architecture search

```powershell
python .\run_deep_model_round2.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv --models temporal_channel_cnn patch_unet_residual conv_lstm_residual temporal_linear_hybrid --output-root revision_outputs/deep_model_round2
```

### 4.4 Round-3 non-Transformer hybrid exploration

```powershell
python .\run_nontransformer_round3.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv --models cnn_lstm_hybrid cnn_tcn_hybrid --output-root revision_outputs/nontransformer_round3
```

### 4.5 Optional fast rerun for the 1-layer hybrid CNN-LSTM

```powershell
python .\run_nontransformer_round3.py --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv --models cnn_lstm_hybrid --patch-batch-size 32 --learning-rate 6e-4 --output-root revision_outputs/nontransformer_round3_fast
```

## 5. Validate the regenerated outputs

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

The report `CAGEO_COMPLETE_EXPERIMENT_REPORT.md` records the expected experiment line and the main reference metrics for these summaries.
