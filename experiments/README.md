# CAGEO Minimal Reproduction Bundle

This directory contains only the source files needed to rerun the CAGEO training, validation, and experiment-analysis figure pipeline described in `CAGEO_COMPLETE_EXPERIMENT_REPORT.md`.

Versioned contents:

- core code:
  - `revision_config.py`
  - `revision_utils.py`
  - `revision_experiments.py`
  - `deep_patch_models.py`
  - `cg_additional_experiments.py`
- run entry points:
  - `run_cg_additional_suite.py`
  - `run_deep_model_repair.py`
  - `run_deep_model_round2.py`
  - `run_nontransformer_round3.py`
  - `preflight_revision.py`
  - `smoke_test_revision.py`
  - `build_cageo_analysis_figures.py`
- environment and setup:
  - `environment.yml`
  - `requirements-revision.txt`
  - `requirements-revision-optional.txt`
  - `bootstrap_revision_env.cmd`
  - `bootstrap_revision_env.ps1`
- documentation:
  - `REPRODUCTION_INSTRUCTIONS.md`
  - `CAGEO_COMPLETE_EXPERIMENT_REPORT.md`
  - `datasets/README.md`
- helper files:
  - `configs/base_revision_config.json`
  - `scripts/reproduce_all.ps1`
  - `scripts/reproduce_all.sh`

Generated artifacts are intentionally not versioned. After you run the pipeline, the following directories will be created locally:

- `revision_outputs/`
- `outputs/`
- `splits/`
- `configs/config_*.json`
- `cageo_submission_assets/`

Use `REPRODUCTION_INSTRUCTIONS.md` for the exact command sequence.
