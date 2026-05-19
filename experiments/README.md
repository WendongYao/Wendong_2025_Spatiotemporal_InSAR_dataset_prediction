# CAGEO Reproduction Package

This directory contains the executable code and documentation required to rerun
the CAGEO experiments, regenerate the analysis figures, and test the software on
a bundled synthetic dataset.

## What is included

- Core experiment code
  - `revision_config.py`
  - `revision_utils.py`
  - `revision_experiments.py`
  - `deep_patch_models.py`
  - `cg_additional_experiments.py`
- Main run entry points
  - `run_cg_additional_suite.py`
  - `run_deep_model_repair.py`
  - `run_deep_model_round2.py`
  - `run_nontransformer_round3.py`
  - `run_synthetic_smoke_case.py`
  - `preflight_revision.py`
  - `smoke_test_revision.py`
  - `build_cageo_analysis_figures.py`
- Environment and dependency files
  - `environment.yml`
  - `requirements-revision.txt`
  - `requirements-revision-optional.txt`
  - `bootstrap_revision_env.cmd`
  - `bootstrap_revision_env.ps1`
- User-facing documentation
  - `REPRODUCTION_INSTRUCTIONS.md`
  - `USER_GUIDE.md`
  - `TUTORIAL.md`
  - `COMPUTATIONAL_REQUIREMENTS.md`
  - `CAGEO_COMPLETE_EXPERIMENT_REPORT.md`
  - `datasets/README.md`
  - `examples/README.md`
- Helper files
  - `configs/base_revision_config.json`
  - `scripts/reproduce_all.ps1`
  - `scripts/reproduce_all.sh`

## Generated artifacts

Generated artifacts are intentionally not versioned. After running the workflow,
these directories are created locally:

- `revision_outputs/`
- `outputs/`
- `splits/`
- `configs/config_*.json`
- `cageo_submission_assets/`
- `synthetic_smoke_outputs/`

## Minimum install

```powershell
conda env create -f environment.yml
conda activate found_training_project
pip install -r requirements-revision.txt
```

For optional SHAP-based interpretability:

```powershell
pip install -r requirements-revision-optional.txt
```

## First commands to run

Dependency-only smoke test:

```powershell
python .\smoke_test_revision.py --skip-csv-check
```

Bundled synthetic example:

```powershell
python .\run_synthetic_smoke_case.py
```

Full paper rerun with the external EGMS CSV:

```powershell
python .\run_cg_additional_suite.py --phase all --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
```

## Documentation index

- `REPRODUCTION_INSTRUCTIONS.md`
  Step-by-step commands for reproducing the paper outputs
- `USER_GUIDE.md`
  Inputs, outputs, options, and expected behavior of the main scripts
- `TUTORIAL.md`
  Typical use cases
- `COMPUTATIONAL_REQUIREMENTS.md`
  Dependencies, hardware tiers, and runtime notes
- `examples/README.md`
  Synthetic test-case materials

## Data note

The real EGMS CSV used in the manuscript is not redistributed here. The
repository instead provides:

- placement instructions for the real CSV
- a synthetic CSV for functional testing
- a generator script for rebuilding or extending the synthetic example
