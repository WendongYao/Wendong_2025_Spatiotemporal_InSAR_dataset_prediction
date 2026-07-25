"""Aggregate the reviewer-priority CAGEO experiments with paired statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def _load(path: Path, consumed: list[Path]) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    consumed.append(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(path: Path, consumed: list[Path]) -> dict[str, object]:
    payload = _load(path, consumed)
    if "direct_raw_rmse" not in payload:
        raise KeyError(f"Missing direct_raw_rmse: {path}")
    return payload


def _core_seconds(payload: dict[str, object]) -> float:
    return float(payload.get("warm_start_seconds") or 0.0) + float(
        payload.get("training_seconds") or 0.0
    ) + float(payload.get("inference_seconds") or 0.0)


def _paired_summary(
    *,
    baseline_name: str,
    baseline: np.ndarray,
    spar: np.ndarray,
    test_train_ratios: np.ndarray,
) -> dict[str, object]:
    difference = baseline - spar
    n = len(difference)
    mean_difference = float(np.mean(difference))
    sem = float(stats.sem(difference)) if n > 1 else float("nan")
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    ci_low = mean_difference - critical * sem
    ci_high = mean_difference + critical * sem
    t_result = stats.ttest_rel(baseline, spar)
    mean_test_train_ratio = float(np.mean(test_train_ratios))
    sample_variance = float(np.var(difference, ddof=1)) if n > 1 else float("nan")
    corrected_sem = float(
        np.sqrt((1.0 / n + mean_test_train_ratio) * sample_variance)
    ) if n > 1 else float("nan")
    corrected_t = mean_difference / corrected_sem if corrected_sem > 0 else float("nan")
    corrected_p = float(2.0 * stats.t.sf(abs(corrected_t), n - 1)) if n > 1 else float("nan")
    corrected_ci_low = mean_difference - critical * corrected_sem
    corrected_ci_high = mean_difference + critical * corrected_sem
    try:
        wilcoxon = stats.wilcoxon(
            difference,
            alternative="greater",
            method="exact",
            zero_method="wilcox",
        )
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    return {
        "baseline": baseline_name,
        "n": int(n),
        "baseline_mean_rmse": float(np.mean(baseline)),
        "baseline_std_rmse": float(np.std(baseline, ddof=1)),
        "spar_mean_rmse": float(np.mean(spar)),
        "spar_std_rmse": float(np.std(spar, ddof=1)),
        "mean_paired_reduction_mm": mean_difference,
        "paired_reduction_ci95_low_mm": float(ci_low),
        "paired_reduction_ci95_high_mm": float(ci_high),
        "mean_test_train_ratio": mean_test_train_ratio,
        "corrected_resampled_ci95_low_mm": float(corrected_ci_low),
        "corrected_resampled_ci95_high_mm": float(corrected_ci_high),
        "ratio_of_means_reduction_percent": float(
            100.0 * (np.mean(baseline) - np.mean(spar)) / np.mean(baseline)
        ),
        "mean_seedwise_reduction_percent": float(
            np.mean(100.0 * (baseline - spar) / baseline)
        ),
        "spar_wins": int(np.sum(spar < baseline)),
        "ties": int(np.sum(np.isclose(spar, baseline))),
        "paired_t_two_sided_p": float(t_result.pvalue),
        "corrected_resampled_t_two_sided_p": corrected_p,
        "wilcoxon_one_sided_p": wilcoxon_p,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT / "R083_priority_aggregates",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    consumed: list[Path] = []
    primary_rows: list[dict[str, object]] = []
    arrays: dict[str, list[float]] = {
        "lasso": [],
        "spar": [],
        "lightgbm": [],
        "gru": [],
        "no_anchor": [],
    }
    primary_test_train_ratios: list[float] = []
    frozen_table_path = (
        PROJECT_ROOT / "source" / "results" / "spar_v2" / "aggregates" / "multiseed.csv"
    )
    if not frozen_table_path.exists():
        raise FileNotFoundError(frozen_table_path)
    consumed.append(frozen_table_path)
    with frozen_table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        frozen_by_seed = {
            int(row["seed"]): row
            for row in csv.DictReader(handle)
        }

    for seed in range(42, 47):
        frozen = frozen_by_seed[seed]
        frozen_manifest = _metric(
            PROJECT_ROOT
            / "source"
            / "results"
            / "spar_v2"
            / "manifests"
            / f"E32N34_seed{seed}_spar_metrics.json",
            consumed,
        )
        train_points = int(frozen_manifest["raw_train_points"])
        test_points = int(frozen_manifest["raw_test_points"])
        primary_test_train_ratios.append(test_points / train_points)
        lasso_metrics_path = (
            RESULTS_ROOT / "R069_saqr_frozen_seed42" / "lasso_raw_supervised" / "metrics.json"
            if seed == 42
            else RESULTS_ROOT
            / f"R068_saqr_seed_{seed}"
            / "lasso_raw_supervised"
            / "metrics.json"
        )
        lasso_metrics = _metric(lasso_metrics_path, consumed)
        if not np.isclose(
            float(lasso_metrics["direct_raw_rmse"]),
            float(frozen["lasso_direct_rmse"]),
            atol=2e-6,
        ):
            raise AssertionError(f"Frozen LASSO mismatch for seed {seed}")
        if seed == 42:
            lightgbm_path = (
                RESULTS_ROOT
                / "R076_lightgbm_target_selection_E32N34_seed42"
                / "direct_raw_lightgbm"
                / "metrics.json"
            )
            gru_path = (
                RESULTS_ROOT
                / "R078_pointwise_gru_E32N34_seed42"
                / "pointwise_gru"
                / "metrics.json"
            )
            no_anchor_path = (
                RESULTS_ROOT
                / "R069_saqr_frozen_seed42"
                / "saqr_no_anchor"
                / "metrics.json"
            )
        else:
            lightgbm_path = (
                RESULTS_ROOT
                / "R079_lightgbm_E32N34_seeds43_46"
                / "E32N34"
                / f"seed_{seed}"
                / "direct_raw_lightgbm"
                / "metrics.json"
            )
            gru_path = (
                RESULTS_ROOT
                / "R080_E32N34_seeds43_46_gru_no_anchor"
                / "E32N34"
                / f"seed_{seed}"
                / "pointwise_gru"
                / "metrics.json"
            )
            no_anchor_path = (
                RESULTS_ROOT
                / "R080_E32N34_seeds43_46_gru_no_anchor"
                / "E32N34"
                / f"seed_{seed}"
                / "saqr_no_anchor"
                / "metrics.json"
            )
        payloads = {
            "lightgbm": _metric(lightgbm_path, consumed),
            "gru": _metric(gru_path, consumed),
            "no_anchor": _metric(no_anchor_path, consumed),
        }
        lasso_value = float(frozen["lasso_direct_rmse"])
        spar_value = float(frozen["spar_direct_rmse"])
        arrays["lasso"].append(lasso_value)
        arrays["spar"].append(spar_value)
        row: dict[str, object] = {
            "tile": "E32N34",
            "seed": seed,
            "lasso_direct_raw_rmse": lasso_value,
            "lasso_direct_equal_cell_rmse": float(frozen["lasso_equal_cell_rmse"]),
            "lasso_core_seconds": float(frozen["lasso_core_seconds"]),
            "lasso_training_seconds": float(lasso_metrics["training_seconds"]),
            "lasso_inference_seconds": float(lasso_metrics["inference_seconds"]),
            "spar_direct_raw_rmse": spar_value,
            "spar_direct_equal_cell_rmse": float(frozen["spar_equal_cell_rmse"]),
            "spar_core_seconds": float(frozen["spar_core_seconds"]),
            "spar_anchor_seconds": float(frozen_manifest.get("warm_start_seconds") or 0.0),
            "spar_training_seconds": float(frozen_manifest["training_seconds"]),
            "spar_inference_seconds": float(frozen_manifest["inference_seconds"]),
            "spar_parameter_count": int(frozen["spar_parameter_count"]),
            "raw_train_points": train_points,
            "raw_test_points": test_points,
            "test_train_ratio": test_points / train_points,
        }
        for name, payload in payloads.items():
            value = float(payload["direct_raw_rmse"])
            arrays[name].append(value)
            row[f"{name}_direct_raw_rmse"] = value
            row[f"{name}_direct_equal_cell_rmse"] = float(
                payload["direct_cell_mean_rmse_equal_cell"]
            )
            row[f"{name}_core_seconds"] = _core_seconds(payload)
            row[f"{name}_training_seconds"] = float(payload.get("training_seconds") or 0.0)
            row[f"{name}_inference_seconds"] = float(payload.get("inference_seconds") or 0.0)
            if name == "no_anchor":
                row[f"{name}_anchor_seconds"] = float(payload.get("warm_start_seconds") or 0.0)
            row[f"{name}_parameter_count"] = payload.get("parameter_count")
        primary_rows.append(row)

    spar_array = np.asarray(arrays["spar"], dtype=np.float64)
    comparison_rows = [
        _paired_summary(
            baseline_name=name,
            baseline=np.asarray(arrays[name], dtype=np.float64),
            spar=spar_array,
            test_train_ratios=np.asarray(primary_test_train_ratios, dtype=np.float64),
        )
        for name in ("lasso", "lightgbm", "gru", "no_anchor")
    ]
    confirmation_rows = [
        _paired_summary(
            baseline_name=name,
            baseline=np.asarray(arrays[name][1:], dtype=np.float64),
            spar=np.asarray(arrays["spar"][1:], dtype=np.float64),
            test_train_ratios=np.asarray(primary_test_train_ratios[1:], dtype=np.float64),
        )
        for name in ("lasso", "lightgbm", "gru", "no_anchor")
    ]

    external_rows: list[dict[str, object]] = []
    external_statistics: list[dict[str, object]] = []
    for tile in ("E29N33", "E36N31", "E37N41"):
        tile_lasso: list[float] = []
        tile_spar: list[float] = []
        tile_test_train_ratios: list[float] = []
        for seed in range(42, 47):
            if seed == 42:
                root = RESULTS_ROOT / "R070_saqr_multiregion_seed42" / tile
            else:
                root = (
                    RESULTS_ROOT
                    / "R081_external_seeds43_46"
                    / tile
                    / f"seed_{seed}"
                )
            lasso = _metric(root / "lasso_raw_supervised" / "metrics.json", consumed)
            spar = _metric(root / "saqr_point_query" / "metrics.json", consumed)
            lasso_rmse = float(lasso["direct_raw_rmse"])
            spar_rmse = float(spar["direct_raw_rmse"])
            tile_lasso.append(lasso_rmse)
            tile_spar.append(spar_rmse)
            train_points = int(spar["raw_train_points"])
            test_points = int(spar["raw_test_points"])
            tile_test_train_ratios.append(test_points / train_points)
            external_rows.append(
                {
                    "tile": tile,
                    "seed": seed,
                    "lasso_direct_raw_rmse": lasso_rmse,
                    "spar_direct_raw_rmse": spar_rmse,
                    "lasso_direct_equal_cell_rmse": float(
                        lasso["direct_cell_mean_rmse_equal_cell"]
                    ),
                    "spar_direct_equal_cell_rmse": float(
                        spar["direct_cell_mean_rmse_equal_cell"]
                    ),
                    "reduction_percent": 100.0 * (lasso_rmse - spar_rmse) / lasso_rmse,
                    "lasso_core_seconds": _core_seconds(lasso),
                    "spar_core_seconds": _core_seconds(spar),
                    "raw_train_points": train_points,
                    "raw_test_points": test_points,
                    "test_train_ratio": test_points / train_points,
                }
            )
        external_statistics.append(
            {
                "tile": tile,
                **_paired_summary(
                    baseline_name="lasso",
                    baseline=np.asarray(tile_lasso),
                    spar=np.asarray(tile_spar),
                    test_train_ratios=np.asarray(tile_test_train_ratios),
                ),
            }
        )

    buffer_summary = _load(
        RESULTS_ROOT / "R082_controlled_buffer_E32N34_seed42" / "summary.json",
        consumed,
    )
    _write_csv(args.output_root / "primary_models_multiseed.csv", primary_rows)
    _write_csv(args.output_root / "primary_paired_statistics.csv", comparison_rows)
    _write_csv(args.output_root / "primary_confirmation_statistics.csv", confirmation_rows)
    _write_csv(args.output_root / "external_regions_multiseed.csv", external_rows)
    _write_csv(args.output_root / "external_regions_statistics.csv", external_statistics)
    summary = {
        "primary_paired_statistics": comparison_rows,
        "primary_confirmation_statistics": confirmation_rows,
        "external_region_statistics": external_statistics,
        "controlled_buffer": buffer_summary,
        "input_metrics": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for path in consumed
        ],
        "command": [sys.executable, *sys.argv],
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(args.output_root.resolve()), "inputs": len(consumed)}))


if __name__ == "__main__":
    main()
