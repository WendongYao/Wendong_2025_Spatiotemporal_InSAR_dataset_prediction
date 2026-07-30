"""Aggregate R094--R097 into paper-facing CAGEO v2.3 evidence tables."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUTPUT = RESULTS / "R098_v23_aggregates"
MODEL_PATHS = {
    "Persistence": "persistence",
    "DLinear": "dlinear",
    "LASSO": "lasso",
    "Causal TCN": "tcn",
    "SPAR": "spar_all_cells_uniform",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(payload: dict[str, object]) -> float:
    return float(payload.get("native_cell_rmse", payload["direct_raw_rmse"]))


def mae(payload: dict[str, object]) -> float:
    return float(payload.get("native_cell_mae", payload["direct_raw_mae"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(
    rows: list[dict[str, object]],
    *,
    baseline: str,
    candidate: str = "SPAR",
) -> dict[str, object]:
    baseline_values = np.asarray(
        [float(row["rmse"]) for row in rows if row["model"] == baseline],
        dtype=np.float64,
    )
    candidate_values = np.asarray(
        [float(row["rmse"]) for row in rows if row["model"] == candidate],
        dtype=np.float64,
    )
    if len(baseline_values) != len(candidate_values) or len(baseline_values) < 2:
        raise ValueError(f"Pairing mismatch: {baseline} vs {candidate}")
    difference = baseline_values - candidate_values
    ratios = np.asarray(
        [
            float(row["test_train_ratio"])
            for row in rows
            if row["model"] == baseline
        ],
        dtype=np.float64,
    )
    n = len(difference)
    variance = float(np.var(difference, ddof=1))
    corrected_se = float(np.sqrt((1.0 / n + float(ratios.mean())) * variance))
    critical = float(stats.t.ppf(0.975, df=n - 1))
    corrected_t = float(difference.mean() / corrected_se)
    wilcoxon = stats.wilcoxon(
        difference,
        alternative="greater",
        method="exact",
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "pair_count": n,
        "baseline_rmse_mean": float(baseline_values.mean()),
        "candidate_rmse_mean": float(candidate_values.mean()),
        "ratio_of_means_reduction_percent": float(
            100.0
            * (baseline_values.mean() - candidate_values.mean())
            / baseline_values.mean()
        ),
        "paired_difference_mean_mm": float(difference.mean()),
        "paired_difference_sd_mm": float(difference.std(ddof=1)),
        "wins": int(np.sum(difference > 0)),
        "ordinary_paired_t_two_sided_p": float(
            stats.ttest_rel(
                baseline_values,
                candidate_values,
            ).pvalue
        ),
        "mean_test_train_ratio": float(ratios.mean()),
        "corrected_resampled_se_mm": corrected_se,
        "corrected_resampled_ci95_low_mm": float(
            difference.mean() - critical * corrected_se
        ),
        "corrected_resampled_ci95_high_mm": float(
            difference.mean() + critical * corrected_se
        ),
        "corrected_resampled_t_two_sided_p": float(
            2.0 * stats.t.sf(abs(corrected_t), df=n - 1)
        ),
        "exact_wilcoxon_greater_statistic": float(wilcoxon.statistic),
        "exact_wilcoxon_greater_p": float(wilcoxon.pvalue),
    }


def confirmation_tables() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    root = RESULTS / "R095_v23_locked_confirmation" / "E32N34"
    rows: list[dict[str, object]] = []
    for seed in (47, 48, 49, 50):
        seed_root = root / f"seed_{seed}"
        task = metric(seed_root / "task_metadata.json")
        test_train_ratio = float(task["raw_test_points"]) / float(task["raw_train_points"])
        for model_name, relative in MODEL_PATHS.items():
            payload = metric(seed_root / relative / "metrics.json")
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "rmse": rmse(payload),
                    "mae": mae(payload),
                    "bias": float(
                        payload.get("native_cell_bias", payload["direct_raw_bias"])
                    ),
                    "training_seconds": float(payload.get("training_seconds", 0.0)),
                    "inference_seconds": float(payload.get("inference_seconds", 0.0)),
                    "core_seconds": float(
                        payload.get(
                            "core_seconds",
                            float(payload.get("training_seconds", 0.0))
                            + float(payload.get("inference_seconds", 0.0)),
                        )
                    ),
                    "parameter_count": int(payload.get("parameter_count", 0)),
                    "train_cells": int(task["raw_train_points"]),
                    "val_cells": int(task["raw_val_points"]),
                    "test_cells": int(task["raw_test_points"]),
                    "test_train_ratio": test_train_ratio,
                }
            )
    summaries: list[dict[str, object]] = []
    for model_name in MODEL_PATHS:
        selected = [row for row in rows if row["model"] == model_name]
        summaries.append(
            {
                "model": model_name,
                "seed_count": len(selected),
                "rmse_mean": float(np.mean([row["rmse"] for row in selected])),
                "rmse_sd": float(np.std([row["rmse"] for row in selected], ddof=1)),
                "mae_mean": float(np.mean([row["mae"] for row in selected])),
                "mae_sd": float(np.std([row["mae"] for row in selected], ddof=1)),
                "core_seconds_mean": float(
                    np.mean([row["core_seconds"] for row in selected])
                ),
                "core_seconds_sd": float(
                    np.std([row["core_seconds"] for row in selected], ddof=1)
                ),
                "parameter_count": int(selected[0]["parameter_count"]),
            }
        )
    paired = [
        paired_statistics(rows, baseline=baseline)
        for baseline in ("Persistence", "DLinear", "LASSO", "Causal TCN")
    ]
    return rows, summaries, paired


def temporal_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root = RESULTS / "R096_v23_temporal_origins" / "E32N34"
    rows: list[dict[str, object]] = []
    for target_col in (252, 272, 292, 312):
        origin_root = root / f"target_col_{target_col}"
        task = metric(origin_root / "task_metadata.json")
        for model_name, relative in MODEL_PATHS.items():
            payload = metric(origin_root / relative / "metrics.json")
            rows.append(
                {
                    "target_col": target_col,
                    "target_date": task["target_date"],
                    "history_date_start": task["history_date_start"],
                    "history_date_end": task["history_date_end"],
                    "forecast_horizon_days": task[
                        "forecast_horizon_days_from_last_history"
                    ],
                    "model": model_name,
                    "rmse": rmse(payload),
                    "mae": mae(payload),
                    "bias": float(
                        payload.get("native_cell_bias", payload["direct_raw_bias"])
                    ),
                    "core_seconds": float(
                        payload.get(
                            "core_seconds",
                            float(payload.get("training_seconds", 0.0))
                            + float(payload.get("inference_seconds", 0.0)),
                        )
                    ),
                }
            )
    summaries: list[dict[str, object]] = []
    spar_values = {
        int(row["target_col"]): float(row["rmse"])
        for row in rows
        if row["model"] == "SPAR"
    }
    for model_name in MODEL_PATHS:
        selected = [row for row in rows if row["model"] == model_name]
        reductions = [
            100.0
            * (float(row["rmse"]) - spar_values[int(row["target_col"])])
            / float(row["rmse"])
            for row in selected
            if model_name != "SPAR"
        ]
        summaries.append(
            {
                "model": model_name,
                "origin_count": len(selected),
                "rmse_mean": float(np.mean([row["rmse"] for row in selected])),
                "rmse_sd": float(np.std([row["rmse"] for row in selected], ddof=1)),
                "rmse_min": float(np.min([row["rmse"] for row in selected])),
                "rmse_max": float(np.max([row["rmse"] for row in selected])),
                "spar_wins": (
                    ""
                    if model_name == "SPAR"
                    else int(
                        np.sum(
                            [
                                spar_values[int(row["target_col"])]
                                < float(row["rmse"])
                                for row in selected
                            ]
                        )
                    )
                ),
                "spar_reduction_percent_mean": (
                    "" if model_name == "SPAR" else float(np.mean(reductions))
                ),
            }
        )
    return rows, summaries


def analytic_tables() -> tuple[list[dict[str, object]], dict[str, object]]:
    root = RESULTS / "R097_v23_analytic_multiseed"
    rows: list[dict[str, object]] = []
    for seed in range(42, 52):
        payload = json.loads(
            (root / f"seed_{seed}" / "summary.json").read_text(encoding="utf-8")
        )
        if len(payload) != 1:
            raise AssertionError(f"Expected one matched-IDW row for seed {seed}")
        row = payload[0]
        rows.append(
            {
                "seed": seed,
                "pseudo_target_test_rmse": float(row["pseudo_target_test_rmse"]),
                "analytic_test_rmse": float(row["analytic_test_rmse"]),
                "optimism_gap_mm": float(row["optimism_gap_mm"]),
                "pseudo_target_distortion_rmse": float(
                    row["pseudo_target_distortion_rmse"]
                ),
                "training_seconds": float(row["training_seconds"]),
                "best_epoch": int(row["best_epoch"]),
            }
        )
    gap = np.asarray([row["optimism_gap_mm"] for row in rows], dtype=np.float64)
    pseudo = np.asarray(
        [row["pseudo_target_test_rmse"] for row in rows],
        dtype=np.float64,
    )
    analytic = np.asarray(
        [row["analytic_test_rmse"] for row in rows],
        dtype=np.float64,
    )
    summary = {
        "seed_count": len(rows),
        "positive_optimism_gap_count": int(np.sum(gap > 0)),
        "pseudo_rmse_mean": float(pseudo.mean()),
        "pseudo_rmse_sd": float(pseudo.std(ddof=1)),
        "analytic_rmse_mean": float(analytic.mean()),
        "analytic_rmse_sd": float(analytic.std(ddof=1)),
        "optimism_gap_mean_mm": float(gap.mean()),
        "optimism_gap_sd_mm": float(gap.std(ddof=1)),
        "optimism_gap_min_mm": float(gap.min()),
        "optimism_gap_max_mm": float(gap.max()),
    }
    return rows, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sampler_rows = json.loads(
        (
            RESULTS
            / "R094_v23_sampler_ablation"
            / "E32N34"
            / "seed_42"
            / "sampler_summary.json"
        ).read_text(encoding="utf-8")
    )
    confirmation_rows, confirmation_summary, confirmation_paired = confirmation_tables()
    temporal_rows, temporal_summary = temporal_tables()
    analytic_rows, analytic_summary = analytic_tables()
    write_csv(OUTPUT / "sampler_ablation.csv", sampler_rows)
    write_csv(OUTPUT / "locked_confirmation_rows.csv", confirmation_rows)
    write_csv(OUTPUT / "locked_confirmation_summary.csv", confirmation_summary)
    write_csv(OUTPUT / "locked_confirmation_paired_statistics.csv", confirmation_paired)
    write_csv(OUTPUT / "temporal_origin_rows.csv", temporal_rows)
    write_csv(OUTPUT / "temporal_origin_summary.csv", temporal_summary)
    write_csv(OUTPUT / "analytic_multiseed_rows.csv", analytic_rows)
    write_csv(OUTPUT / "analytic_multiseed_summary.csv", [analytic_summary])
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selected_sampler": "all_cells_uniform",
        "sampler_ablation": sampler_rows,
        "locked_confirmation_summary": confirmation_summary,
        "locked_confirmation_paired_statistics": confirmation_paired,
        "temporal_origin_summary": temporal_summary,
        "analytic_multiseed_summary": analytic_summary,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    result_files = [
        path
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name not in {"manifest.json", "RESULTS_SUMMARY.md"}
    ]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "aggregator_sha256": sha256(Path(__file__).resolve()),
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in result_files
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selected_sampler": "all_cells_uniform",
                "confirmation_rows": len(confirmation_rows),
                "temporal_rows": len(temporal_rows),
                "analytic_rows": len(analytic_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
