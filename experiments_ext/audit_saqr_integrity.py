"""Deterministic integrity checks for the frozen SPAR result package.

The ``saqr`` path names are historical machine identifiers retained to verify
the original artifacts without renaming them in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUTPUT = RESULTS / "R075_saqr_integrity"


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, tolerance: float = 2e-6) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"Metric mismatch: {actual} != {expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_direct_artifact(model_dir: Path) -> dict[str, object]:
    metrics = read_json(model_dir / "metrics.json")
    artifact_path = model_dir / "direct_raw_test_predictions.npz"
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    artifact = np.load(artifact_path, allow_pickle=False)
    indices = artifact["indices"].astype(np.int64)
    truth = artifact["truth"].astype(np.float64)
    prediction = artifact["prediction"].astype(np.float64)
    residual = prediction - truth
    if not np.array_equal(artifact["residual"].astype(np.float64), residual):
        if not np.allclose(artifact["residual"], residual, rtol=0.0, atol=2e-6):
            raise AssertionError(f"Residual artifact mismatch in {model_dir}")
    if len(np.unique(indices)) != len(indices):
        raise AssertionError(f"Duplicate direct prediction indices in {model_dir}")
    close(float(np.sqrt(np.mean(residual**2))), float(metrics["direct_raw_rmse"]))
    close(float(np.mean(np.abs(residual))), float(metrics["direct_raw_mae"]))
    close(float(np.mean(residual)), float(metrics["direct_raw_bias"]))
    if int(metrics["direct_raw_point_count"]) != len(indices):
        raise AssertionError(f"Point count mismatch in {model_dir}")
    required_flags = {
        "target_supervision": "raw_observations_only",
        "interpolated_future_target_used_for_loss": False,
        "support_history_source": "direct_raw_point",
        "spatial_context_enabled": False,
        "global_coordinate_conditioning": False,
        "local_coordinate_conditioning": False,
        "anchor_enabled": True,
    }
    for key, expected in required_flags.items():
        if metrics.get(key) != expected:
            raise AssertionError(f"Unexpected {key} in {model_dir}: {metrics.get(key)!r}")
    if metrics.get("test_target_used_in_any_target_grid") is not False:
        raise AssertionError(f"Test-target isolation flag failed in {model_dir}")
    return {
        "model_dir": str(model_dir.resolve()),
        "point_count": len(indices),
        "direct_raw_rmse": float(metrics["direct_raw_rmse"]),
        "prediction_sha256": sha256(artifact_path),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    primary_dirs = [RESULTS / "R069_saqr_frozen_seed42" / "saqr_point_query"]
    primary_dirs.extend(
        RESULTS / f"R068_saqr_seed_{seed}" / "saqr_no_global_coord"
        for seed in range(43, 47)
    )
    external_dirs = [
        RESULTS / "R070_saqr_multiregion_seed42" / tile / "saqr_point_query"
        for tile in ["E29N33", "E36N31", "E37N41"]
    ]
    synthetic_dirs = [
        case / "saqr_point_query"
        for case in sorted((RESULTS / "R072_saqr_synthetic_truth").iterdir())
        if case.is_dir()
    ]
    for model_dir in [*primary_dirs, *external_dirs, *synthetic_dirs]:
        checks.append(verify_direct_artifact(model_dir))

    summary = read_json(RESULTS / "R073_saqr_evidence" / "summary.json")
    primary_rmse = np.asarray([float(item["direct_raw_rmse"]) for item in checks[:5]])
    close(float(primary_rmse.mean()), float(summary["multiseed"]["saqr_direct_rmse_mean"]))
    if int(summary["multiseed"]["wins"]) != 5:
        raise AssertionError("Primary win count is not five.")
    if int(summary["confirmatory_seeds_43_46"]["wins"]) != 4:
        raise AssertionError("Confirmatory win count is not four.")

    composite_metrics = []
    for operator in ["idw", "linear", "nearest"]:
        metrics = read_json(
            RESULTS
            / "R072_saqr_synthetic_truth"
            / f"composite_{operator}"
            / "saqr_point_query"
            / "metrics.json"
        )
        composite_metrics.append(metrics)
    direct_values = np.asarray([float(metrics["direct_raw_rmse"]) for metrics in composite_metrics])
    if float(np.ptp(direct_values)) > 1e-8:
        raise AssertionError("Composite direct metrics changed across interpolation operators.")
    dense_values = np.asarray([float(metrics["dense_analytic_test_rmse"]) for metrics in composite_metrics])
    if float(np.ptp(dense_values)) < 0.30:
        raise AssertionError("Synthetic dense operator sensitivity is unexpectedly small.")

    expected_files = [
        PROJECT_ROOT / "SAQR_EXPERIMENT_RESULTS.md",
        PROJECT_ROOT / "refine-logs" / "RESULT_TO_CLAIM.md",
        RESULTS / "R073_saqr_evidence" / "multiseed.csv",
        RESULTS / "R073_saqr_evidence" / "external_regions.csv",
        RESULTS / "R073_saqr_evidence" / "ablations_seed42.csv",
        RESULTS / "R073_saqr_evidence" / "synthetic_truth.csv",
        RESULTS / "R074_saqr_buffer1_seed42" / "saqr_point_query" / "metrics.json",
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    code_files = [
        PROJECT_ROOT / "experiments_ext" / "support_aware_model.py",
        PROJECT_ROOT / "experiments_ext" / "raw_point_supervision.py",
        PROJECT_ROOT / "experiments_ext" / "run_raw_holdout_pilot.py",
        PROJECT_ROOT / "experiments_ext" / "synthetic_truth_data.py",
        PROJECT_ROOT / "experiments_ext" / "run_saqr_synthetic_truth.py",
        PROJECT_ROOT / "experiments_ext" / "analyze_saqr_results.py",
    ]
    code_hashes = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in code_files}
    payload = {
        "status": "pass_with_provenance_warning",
        "verified_direct_prediction_artifacts": len(checks),
        "primary_artifacts": len(primary_dirs),
        "external_artifacts": len(external_dirs),
        "synthetic_artifacts": len(synthetic_dirs),
        "composite_direct_operator_range": float(np.ptp(direct_values)),
        "composite_dense_operator_range": float(np.ptp(dense_values)),
        "expected_files_missing": missing,
        "current_code_hashes": code_hashes,
        "warning": (
            "Run manifests record the clean public source commit but do not contain hashes of the evolving "
            "experiments_ext code at launch time. Metrics flags and artifacts identify the frozen variants, "
            "but exact historical extension-code provenance is incomplete."
        ),
        "checks": checks,
    }
    (OUTPUT / "audit_checks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
