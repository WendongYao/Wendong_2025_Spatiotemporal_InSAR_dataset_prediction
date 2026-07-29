"""Aggregate v2.2 native-support, provenance, support, and quality evidence."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, consumed: list[Path]) -> dict[str, object]:
    consumed.append(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_summary(
    baseline_name: str,
    baseline: np.ndarray,
    spar: np.ndarray,
    test_train_ratio: np.ndarray,
) -> dict[str, object]:
    difference = baseline - spar
    n = len(difference)
    mean = float(difference.mean())
    sem = float(stats.sem(difference))
    critical = float(stats.t.ppf(0.975, n - 1))
    variance = float(np.var(difference, ddof=1))
    ratio = float(np.mean(test_train_ratio))
    corrected_sem = float(np.sqrt((1.0 / n + ratio) * variance))
    corrected_t = mean / corrected_sem if corrected_sem > 0 else float("nan")
    try:
        wilcoxon_p = float(
            stats.wilcoxon(
                difference,
                alternative="greater",
                method="exact",
                zero_method="wilcox",
            ).pvalue
        )
    except ValueError:
        wilcoxon_p = float("nan")
    return {
        "baseline": baseline_name,
        "n": n,
        "baseline_mean_rmse": float(baseline.mean()),
        "baseline_std_rmse": float(baseline.std(ddof=1)),
        "spar_mean_rmse": float(spar.mean()),
        "spar_std_rmse": float(spar.std(ddof=1)),
        "mean_paired_reduction_mm": mean,
        "paired_reduction_ci95_low_mm": mean - critical * sem,
        "paired_reduction_ci95_high_mm": mean + critical * sem,
        "corrected_resampled_ci95_low_mm": mean - critical * corrected_sem,
        "corrected_resampled_ci95_high_mm": mean + critical * corrected_sem,
        "ratio_of_means_reduction_percent": float(
            100.0 * (baseline.mean() - spar.mean()) / baseline.mean()
        ),
        "spar_wins": int(np.sum(spar < baseline)),
        "paired_t_two_sided_p": float(stats.ttest_rel(baseline, spar).pvalue),
        "corrected_resampled_t_two_sided_p": float(
            2.0 * stats.t.sf(abs(corrected_t), n - 1)
        ),
        "wilcoxon_one_sided_p": wilcoxon_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--baseline-seed42-root", type=Path, required=True)
    parser.add_argument("--baseline-confirmation-root", type=Path, required=True)
    parser.add_argument("--multires-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    consumed: list[Path] = [Path(__file__)]
    rows: list[dict[str, object]] = []
    arrays: dict[str, list[float]] = {
        name: []
        for name in (
            "persistence",
            "linear_trend",
            "dlinear",
            "lasso",
            "lightgbm",
            "gru",
            "spar",
        )
    }
    test_train_ratios = []
    for seed in range(42, 47):
        confirmation = args.confirmation_root / "E32N34" / f"seed_{seed}"
        baseline = (
            args.baseline_seed42_root
            if seed == 42
            else args.baseline_confirmation_root / "E32N34" / f"seed_{seed}"
        )
        paths = {
            "persistence": baseline / "persistence" / "metrics.json",
            "linear_trend": baseline / "linear_trend" / "metrics.json",
            "dlinear": baseline / "dlinear" / "metrics.json",
            "lasso": confirmation / "lasso_raw_supervised" / "metrics.json",
            "spar": confirmation / "saqr_point_query" / "metrics.json",
            "lightgbm": (
                PROJECT_ROOT
                / "results"
                / "R076_lightgbm_target_selection_E32N34_seed42"
                / "direct_raw_lightgbm"
                / "metrics.json"
                if seed == 42
                else PROJECT_ROOT
                / "results"
                / "R079_lightgbm_E32N34_seeds43_46"
                / "E32N34"
                / f"seed_{seed}"
                / "direct_raw_lightgbm"
                / "metrics.json"
            ),
            "gru": (
                PROJECT_ROOT
                / "results"
                / "R078_pointwise_gru_E32N34_seed42"
                / "pointwise_gru"
                / "metrics.json"
                if seed == 42
                else PROJECT_ROOT
                / "results"
                / "R080_E32N34_seeds43_46_gru_no_anchor"
                / "E32N34"
                / f"seed_{seed}"
                / "pointwise_gru"
                / "metrics.json"
            ),
        }
        payloads = {name: read_json(path, consumed) for name, path in paths.items()}
        train_count = int(payloads["spar"]["raw_train_points"])
        test_count = int(payloads["spar"]["raw_test_points"])
        test_train_ratios.append(test_count / train_count)
        row: dict[str, object] = {
            "tile": "E32N34",
            "seed": seed,
            "raw_train_product_cells": train_count,
            "raw_test_product_cells": test_count,
            "test_train_ratio": test_count / train_count,
        }
        for name, payload in payloads.items():
            rmse = float(payload.get("native_cell_rmse", payload["direct_raw_rmse"]))
            arrays[name].append(rmse)
            row[f"{name}_native_cell_rmse"] = rmse
            row[f"{name}_native_cell_mae"] = float(
                payload.get("native_cell_mae", payload["direct_raw_mae"])
            )
            row[f"{name}_training_seconds"] = float(
                payload.get("training_seconds") or 0.0
            ) + float(payload.get("warm_start_seconds") or 0.0)
            row[f"{name}_inference_seconds"] = float(
                payload.get("inference_seconds") or 0.0
            )
            row[f"{name}_core_seconds"] = (
                row[f"{name}_training_seconds"] + row[f"{name}_inference_seconds"]
            )
            row[f"{name}_parameter_count"] = payload.get("parameter_count")
        rows.append(row)

    spar = np.asarray(arrays["spar"], dtype=np.float64)
    ratios = np.asarray(test_train_ratios, dtype=np.float64)
    statistics_rows = [
        paired_summary(
            name,
            np.asarray(arrays[name], dtype=np.float64),
            spar,
            ratios,
        )
        for name in (
            "persistence",
            "linear_trend",
            "dlinear",
            "lasso",
            "lightgbm",
            "gru",
        )
    ]
    confirmation_rows = [
        paired_summary(
            name,
            np.asarray(arrays[name][1:], dtype=np.float64),
            spar[1:],
            ratios[1:],
        )
        for name in (
            "persistence",
            "linear_trend",
            "dlinear",
            "lasso",
            "lightgbm",
            "gru",
        )
    ]

    frozen_path = (
        PROJECT_ROOT
        / "source"
        / "results"
        / "spar_v2"
        / "aggregates"
        / "multiseed.csv"
    )
    consumed.append(frozen_path)
    with frozen_path.open("r", encoding="utf-8-sig", newline="") as handle:
        frozen = {int(row["seed"]): row for row in csv.DictReader(handle)}
    reproduction_rows = []
    for row in rows:
        seed = int(row["seed"])
        reproduction_rows.append(
            {
                "seed": seed,
                "v21_lasso_rmse": float(frozen[seed]["lasso_direct_rmse"]),
                "v22_lasso_rmse": float(row["lasso_native_cell_rmse"]),
                "lasso_absolute_difference": abs(
                    float(frozen[seed]["lasso_direct_rmse"])
                    - float(row["lasso_native_cell_rmse"])
                ),
                "v21_spar_rmse": float(frozen[seed]["spar_direct_rmse"]),
                "v22_spar_rmse": float(row["spar_native_cell_rmse"]),
                "spar_absolute_difference": abs(
                    float(frozen[seed]["spar_direct_rmse"])
                    - float(row["spar_native_cell_rmse"])
                ),
            }
        )

    multires_csv = args.multires_root / "multires_metrics.csv"
    quality_csv = args.quality_root / "quality_stratified_summary.csv"
    consumed.extend([multires_csv, quality_csv])
    with multires_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        multires_rows = list(csv.DictReader(handle))
    grouped_multires: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in multires_rows:
        grouped_multires[(row["support"], row["model"])].append(row)
    multires_summary_rows: list[dict[str, object]] = []
    for (support, model), selected in grouped_multires.items():
        rmses = np.asarray([float(row["rmse"]) for row in selected])
        maes = np.asarray([float(row["mae"]) for row in selected])
        multires_summary_rows.append(
            {
                "support": support,
                "model": model,
                "partitions": len(selected),
                "mean_rmse": float(rmses.mean()),
                "std_rmse": float(rmses.std(ddof=1)),
                "mean_mae": float(maes.mean()),
                "std_mae": float(maes.std(ddof=1)),
            }
        )
    multires_paired_rows: list[dict[str, object]] = []
    for support in dict.fromkeys(row["support"] for row in multires_rows):
        lasso = np.asarray(
            [
                float(row["rmse"])
                for row in multires_rows
                if row["support"] == support and row["model"] == "lasso"
            ]
        )
        support_spar = np.asarray(
            [
                float(row["rmse"])
                for row in multires_rows
                if row["support"] == support and row["model"] == "spar"
            ]
        )
        difference = lasso - support_spar
        multires_paired_rows.append(
            {
                "support": support,
                "partitions": len(difference),
                "lasso_mean_rmse": float(lasso.mean()),
                "spar_mean_rmse": float(support_spar.mean()),
                "mean_paired_reduction_mm": float(difference.mean()),
                "ratio_of_means_reduction_percent": float(
                    100.0 * (lasso.mean() - support_spar.mean()) / lasso.mean()
                ),
                "spar_wins": int(np.sum(support_spar < lasso)),
                "paired_t_two_sided_p": float(
                    stats.ttest_rel(lasso, support_spar).pvalue
                ),
            }
        )
    with quality_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        quality_rows = list(csv.DictReader(handle))
    native_model_summary_rows: list[dict[str, object]] = []
    for name in (
        "persistence",
        "linear_trend",
        "dlinear",
        "lasso",
        "lightgbm",
        "gru",
        "spar",
    ):
        rmses = np.asarray([float(row[f"{name}_native_cell_rmse"]) for row in rows])
        maes = np.asarray([float(row[f"{name}_native_cell_mae"]) for row in rows])
        core_seconds = np.asarray(
            [float(row[f"{name}_core_seconds"]) for row in rows]
        )
        parameter_counts = [
            row[f"{name}_parameter_count"]
            for row in rows
            if row[f"{name}_parameter_count"] is not None
        ]
        native_model_summary_rows.append(
            {
                "model": name,
                "partitions": len(rows),
                "mean_rmse": float(rmses.mean()),
                "std_rmse": float(rmses.std(ddof=1)),
                "mean_mae": float(maes.mean()),
                "std_mae": float(maes.std(ddof=1)),
                "mean_core_seconds": float(core_seconds.mean()),
                "std_core_seconds": float(core_seconds.std(ddof=1)),
                "parameter_count": (
                    int(parameter_counts[0]) if parameter_counts else None
                ),
            }
        )
    write_csv(args.output_root / "native_primary_models_multiseed.csv", rows)
    write_csv(
        args.output_root / "native_primary_model_summary.csv",
        native_model_summary_rows,
    )
    write_csv(args.output_root / "native_primary_paired_statistics.csv", statistics_rows)
    write_csv(
        args.output_root / "native_confirmation_paired_statistics.csv",
        confirmation_rows,
    )
    write_csv(args.output_root / "v21_v22_reproduction.csv", reproduction_rows)
    write_csv(args.output_root / "multires_support_summary.csv", multires_summary_rows)
    write_csv(args.output_root / "multires_support_paired.csv", multires_paired_rows)
    write_csv(args.output_root / "quality_stratified_summary.csv", quality_rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "native_model_summary": native_model_summary_rows,
        "primary_statistics": statistics_rows,
        "confirmation_statistics": confirmation_rows,
        "reproduction": reproduction_rows,
        "multires_summary": multires_summary_rows,
        "multires_paired": multires_paired_rows,
        "quality_summary": quality_rows,
        "consumed_sha256": {
            str(path.resolve()): sha256(path) for path in consumed
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    output_files = sorted(
        path
        for path in args.output_root.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "consumed_sha256": {
            str(path.resolve()): sha256(path) for path in consumed
        },
        "output_sha256": {
            path.name: sha256(path) for path in output_files
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
