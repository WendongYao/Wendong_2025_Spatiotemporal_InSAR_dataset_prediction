"""Post-hoc error stratification by the EGMS CSV rmse product attribute."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

from raw_holdout_data import RawHoldoutSpec, load_quality_rmse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    residual = prediction - truth
    return {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(
        dict.fromkeys(field for row in rows for field in row)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--baseline-seed42-root", type=Path, required=True)
    parser.add_argument("--baseline-confirmation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    quality_all = load_quality_rmse(
        RawHoldoutSpec(csv_path=args.csv_path, tile="E32N34")
    )
    rows: list[dict[str, object]] = []
    consumed: list[Path] = [args.csv_path, Path(__file__)]
    for seed in args.seeds:
        confirmation = args.confirmation_root / "E32N34" / f"seed_{seed}"
        baseline = (
            args.baseline_seed42_root
            if seed == 42
            else args.baseline_confirmation_root / "E32N34" / f"seed_{seed}"
        )
        paths = {
            "persistence": baseline
            / "persistence"
            / "direct_native_test_predictions.npz",
            "linear_trend": baseline
            / "linear_trend"
            / "direct_native_test_predictions.npz",
            "dlinear": baseline
            / "dlinear"
            / "direct_native_test_predictions.npz",
            "lasso": confirmation
            / "lasso_raw_supervised"
            / "direct_raw_test_predictions.npz",
            "spar": confirmation
            / "saqr_point_query"
            / "direct_raw_test_predictions.npz",
        }
        artifacts = {
            name: np.load(path, allow_pickle=False) for name, path in paths.items()
        }
        consumed.extend(paths.values())
        reference = artifacts["spar"]
        indices = reference["indices"].astype(np.int64)
        truth = reference["truth"].astype(np.float32)
        quality = quality_all[indices].astype(np.float64)
        edges = np.quantile(quality, [0.0, 0.25, 0.5, 0.75, 1.0])
        for artifact in artifacts.values():
            if not np.array_equal(artifact["indices"].astype(np.int64), indices):
                raise AssertionError(f"Prediction indices differ for seed {seed}.")
            if not np.allclose(artifact["truth"].astype(np.float32), truth):
                raise AssertionError(f"Prediction truths differ for seed {seed}.")
        for bin_index in range(4):
            lower = float(edges[bin_index])
            upper = float(edges[bin_index + 1])
            mask = (
                (quality >= lower) & (quality <= upper)
                if bin_index == 3
                else (quality >= lower) & (quality < upper)
            )
            for model_name, artifact in artifacts.items():
                row = {
                    "seed": seed,
                    "model": model_name,
                    "quality_bin": f"Q{bin_index + 1}",
                    "quality_lower": lower,
                    "quality_upper": upper,
                    **metrics(
                        truth[mask],
                        artifact["prediction"].astype(np.float32)[mask],
                    ),
                }
                rows.append(row)

    summary_rows: list[dict[str, object]] = []
    for quality_bin in ("Q1", "Q2", "Q3", "Q4"):
        for model_name in ("persistence", "linear_trend", "dlinear", "lasso", "spar"):
            selected = [
                row
                for row in rows
                if row["quality_bin"] == quality_bin and row["model"] == model_name
            ]
            rmses = np.asarray([float(row["rmse"]) for row in selected])
            maes = np.asarray([float(row["mae"]) for row in selected])
            summary_rows.append(
                {
                    "quality_bin": quality_bin,
                    "model": model_name,
                    "partitions": len(selected),
                    "mean_rmse": float(rmses.mean()),
                    "std_rmse": float(rmses.std(ddof=1)),
                    "mean_mae": float(maes.mean()),
                    "std_mae": float(maes.std(ddof=1)),
                }
            )
        lasso = np.asarray(
            [
                float(row["rmse"])
                for row in rows
                if row["quality_bin"] == quality_bin and row["model"] == "lasso"
            ]
        )
        spar = np.asarray(
            [
                float(row["rmse"])
                for row in rows
                if row["quality_bin"] == quality_bin and row["model"] == "spar"
            ]
        )
        paired = lasso - spar
        summary_rows.append(
            {
                "quality_bin": quality_bin,
                "model": "spar_vs_lasso",
                "partitions": len(paired),
                "mean_rmse": None,
                "std_rmse": None,
                "mean_mae": None,
                "std_mae": None,
                "mean_paired_rmse_reduction_mm": float(paired.mean()),
                "paired_t_two_sided_p": float(stats.ttest_rel(lasso, spar).pvalue),
                "spar_wins": int(np.sum(spar < lasso)),
            }
        )
    write_csv(args.output_root / "quality_stratified_metrics.csv", rows)
    write_csv(args.output_root / "quality_stratified_summary.csv", summary_rows)
    (args.output_root / "quality_stratified_metrics.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "quality_field": "CSV column rmse",
        "quality_semantics": (
            "EGMS product attribute used for post-hoc stratification only; "
            "not an independent truth error and not the official 8-mm STD."
        ),
        "binning": "within-partition test-set quartiles",
        "consumed_sha256": {
            str(path.resolve()): sha256(path) for path in consumed
        },
        "output_sha256": {
            path.name: sha256(path)
            for path in args.output_root.iterdir()
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
