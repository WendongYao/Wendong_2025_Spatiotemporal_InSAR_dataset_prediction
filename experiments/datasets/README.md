# Data Placement

The manuscript-scale workflow expects the external EGMS CSV:

- `EGMS_L3_E32N34_100km_U_2018_2022_1.csv`

The file is not committed in this repository. To rerun the current paper
workflow, either:

1. place the CSV directly in `experiments/`
2. place it in `experiments/datasets/`
3. pass an explicit `--csv-path` to the runner scripts

The manuscript describes the EGMS source as an external product obtained from
the official EGMS portal rather than as a repository-bundled dataset.

## Test-case data that is bundled here

For functional testing, the repository does include a synthetic dataset:

- `examples/synthetic_egms_small.csv`

That synthetic CSV is not intended to reproduce the paper numbers. It exists so
that installation and command-line behavior can be verified without redistributing
the real EGMS file.
