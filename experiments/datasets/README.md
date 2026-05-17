# Data Placement

The current manuscript bundle expects the EGMS CSV file:

`EGMS_L3_E32N34_100km_U_2018_2022_1.csv`

The file is not committed in this repository. To rerun the current paper workflow, either:

1. place the CSV directly in `experiments/`, or
2. place it in `experiments/datasets/`, or
3. pass an explicit `--csv-path` to the runner scripts

The manuscript describes the EGMS source as an external product obtained from the official EGMS portal rather than as a repository-bundled dataset.
