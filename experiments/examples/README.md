# Synthetic Example Materials

This directory contains a versioned synthetic dataset and a generator script for
functional testing.

Included files:

- `synthetic_egms_small.csv`
  A compact synthetic CSV that matches the column layout expected by the code
- `generate_synthetic_egms_dataset.py`
  A script that rebuilds the synthetic CSV or creates a larger/smaller variant

## Why this directory exists

The real EGMS manuscript CSV is not redistributed in this repository. The
synthetic example is included so that users can still:

- test installation
- verify command-line behavior
- exercise the pipeline on a shareable input

## Basic use

Run the bundled smoke case:

```powershell
python .\run_synthetic_smoke_case.py
```

Rebuild the synthetic CSV:

```powershell
python .\examples\generate_synthetic_egms_dataset.py
```

Or create a custom file:

```powershell
python .\examples\generate_synthetic_egms_dataset.py --output C:\path\to\synthetic_custom.csv --nx 24 --ny 24
```
