"""Recompute and verify the R085 interpolated-target confound artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS = PROJECT_ROOT / "source" / "experiments"
if not SOURCE_EXPERIMENTS.is_dir():
    SOURCE_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(SOURCE_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENTS))

from synthetic_truth_data import SyntheticTruthSpec, build_synthetic_truth_task  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "R085_interpolated_target_confound",
    )
    args = parser.parse_args()
    root = args.result_root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != 6:
        raise AssertionError(f"Expected six rows, found {len(rows)}")
    keys = {(row["model"], row["target_interpolation"]) for row in rows}
    if len(keys) != 6:
        raise AssertionError("Duplicate or missing model/operator rows")

    spec_payload = manifest["spec"]
    spec = SyntheticTruthSpec(**spec_payload)
    task = build_synthetic_truth_task(spec)
    test = task.dense_task.test_mask
    checks: list[dict[str, object]] = []
    for row in rows:
        condition = root / f"target_{row['target_interpolation']}"
        model_dir = condition / str(row["model"])
        prediction = np.load(model_dir / "prediction_grid.npy").astype(np.float64)
        analytic = np.load(condition / "analytic_target.npy").astype(np.float64)
        pseudo = np.load(condition / "pseudo_target.npy").astype(np.float64)
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        analytic_rmse = float(np.sqrt(np.mean((prediction[test] - analytic[test]) ** 2)))
        pseudo_rmse = float(np.sqrt(np.mean((prediction[test] - pseudo[test]) ** 2)))
        distortion = float(np.sqrt(np.mean((pseudo[test] - analytic[test]) ** 2)))
        optimism = analytic_rmse - pseudo_rmse
        expected = {
            "analytic_test_rmse": analytic_rmse,
            "pseudo_target_test_rmse": pseudo_rmse,
            "pseudo_target_distortion_rmse": distortion,
            "optimism_gap_mm": optimism,
        }
        for name, value in expected.items():
            if not np.isclose(float(metrics[name]), value, atol=1e-7):
                raise AssertionError(f"Metric mismatch {model_dir} {name}")
            if not np.isclose(float(row[name]), value, atol=1e-7):
                raise AssertionError(f"Summary mismatch {model_dir} {name}")
        if metrics.get("test_target_used_in_any_target_grid") is not True:
            raise AssertionError(f"Missing leakage flag: {model_dir}")
        if metrics.get("deployable_forecast") is not False:
            raise AssertionError(f"Missing diagnostic flag: {model_dir}")
        checks.append(
            {
                "model": row["model"],
                "target_interpolation": row["target_interpolation"],
                **expected,
                "prediction_sha256": sha256(model_dir / "prediction_grid.npy"),
                "metrics_sha256": sha256(model_dir / "metrics.json"),
                "analytic_target_sha256": sha256(condition / "analytic_target.npy"),
                "pseudo_target_sha256": sha256(condition / "pseudo_target.npy"),
            }
        )

    hybrid = {row["target_interpolation"]: row for row in rows if row["model"] == "cnn_lstm_hybrid"}
    if not (
        float(hybrid["idw"]["optimism_gap_mm"])
        > float(hybrid["nearest"]["optimism_gap_mm"])
        > float(hybrid["linear"]["optimism_gap_mm"])
    ):
        raise AssertionError("Expected IDW-matched optimism ordering was not reproduced")

    payload = {
        "verdict": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "analytic-interpolated-target-confound-v1",
        "rows_verified": len(checks),
        "checks": checks,
        "summary_sha256": sha256(root / "summary.json"),
        "manifest_sha256": sha256(root / "run_manifest.json"),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "scope_warning": (
            "This verifies artifact consistency only. The experiment is deliberately "
            "non-deployable because held-out future support values enter the pseudo-target."
        ),
    }
    (root / "integrity_checks.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"verdict": "PASS", "rows_verified": len(checks)}))


if __name__ == "__main__":
    main()
