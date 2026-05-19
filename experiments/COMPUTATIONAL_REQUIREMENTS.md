# Computational Requirements

This file summarizes the software dependencies, hardware expectations, and
runtime tiers for the CAGEO reproduction package.

## Software dependencies

Core dependencies are listed in:

- `requirements-revision.txt`
- `environment.yml`

Core packages:

- Python 3.11
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- lightgbm
- torch

Optional package:

- shap

## Installation notes

Baseline installation:

```powershell
conda env create -f environment.yml
conda activate found_training_project
pip install -r requirements-revision.txt
```

Optional interpretability dependency:

```powershell
pip install -r requirements-revision-optional.txt
```

For CUDA acceleration, install a CUDA-capable PyTorch build that matches your
system. CPU execution is supported, but long experiment stages will be slower.

## Data requirements

Real-data reproduction requires the EGMS CSV referenced in the manuscript:

- `EGMS_L3_E32N34_100km_U_2018_2022_1.csv`

When this file cannot be redistributed, the repository still provides:

- a synthetic CSV for test cases
- a synthetic-data generator script

## Hardware tiers

### Tier 1: dependency smoke test

Suitable for:
- `smoke_test_revision.py --skip-csv-check`

Recommended minimum:
- CPU only
- 4 GB RAM

### Tier 2: bundled synthetic test case

Suitable for:
- `run_synthetic_smoke_case.py`

Recommended minimum:
- CPU only is acceptable
- 8 GB RAM
- no GPU required

### Tier 3: manuscript-scale reproduction

Suitable for:
- `run_cg_additional_suite.py`
- `run_deep_model_repair.py`
- `run_deep_model_round2.py`
- `run_nontransformer_round3.py`

Recommended:
- 16 GB or more system RAM
- NVIDIA GPU with at least 8 GB VRAM for comfortable deep-model runs
- more VRAM is useful for larger patch batches and repeated architecture screens

CPU-only execution is still supported, but the deep-learning stages will take
substantially longer.

## Storage expectations

The clean repository is small because generated outputs are not versioned.
Local reruns can create additional directories such as:

- `revision_outputs/`
- `outputs/`
- `splits/`
- `cageo_submission_assets/`
- `synthetic_smoke_outputs/`

Reserve several gigabytes of free disk space for manuscript-scale reruns.

## Runtime expectations

Runtime depends on hardware and whether GPU acceleration is available. Typical
relative behavior in this project is:

- smoke test: seconds
- synthetic smoke case: minutes or less on most machines
- classical baselines on the real task: short to moderate
- deep-model architecture screens on the real task: the longest stages

For the paper-scale deep runs, the final Hybrid CNN-LSTM and related models are
the stages most likely to benefit from CUDA acceleration.
