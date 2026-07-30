"""Aggregate final-sampler regional replications with the frozen v2.3 core tables."""

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
INPUT_ROOT = RESULTS / "R099_v23_external_locked"
CORE_ROOT = RESULTS / "R098_v23_aggregates"
CONFIRMATION_ROOT = RESULTS / "R095_v23_locked_confirmation" / "E32N34"
ANCHOR_ROOT = RESULTS / "R101_v23_anchor_ablation" / "E32N34"
OUTPUT = RESULTS / "R100_v23_final_aggregates"
TILES = ("E29N33", "E36N31", "E37N41")
SEEDS = (47, 48, 49, 50)
MODEL_PATHS = {
    "Persistence": "persistence",
    "DLinear": "dlinear",
    "LASSO": "lasso",
    "SPAR": "spar_all_cells_uniform",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(payload: dict[str, object]) -> float:
    if "native_cell_rmse" in payload:
        return float(payload["native_cell_rmse"])
    return float(payload["direct_raw_rmse"])


def mae(payload: dict[str, object]) -> float:
    if "native_cell_mae" in payload:
        return float(payload["native_cell_mae"])
    return float(payload["direct_raw_mae"])


def bias(payload: dict[str, object]) -> float:
    if "native_cell_bias" in payload:
        return float(payload["native_cell_bias"])
    return float(payload["direct_raw_bias"])


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
    tile: str,
    baseline: str,
) -> dict[str, object]:
    tile_rows = [row for row in rows if row["tile"] == tile]
    baseline_by_seed = {
        int(row["seed"]): float(row["rmse"])
        for row in tile_rows
        if row["model"] == baseline
    }
    spar_by_seed = {
        int(row["seed"]): float(row["rmse"])
        for row in tile_rows
        if row["model"] == "SPAR"
    }
    ratio_by_seed = {
        int(row["seed"]): float(row["test_train_ratio"])
        for row in tile_rows
        if row["model"] == baseline
    }
    if set(baseline_by_seed) != set(SEEDS) or set(spar_by_seed) != set(SEEDS):
        raise ValueError(f"Pairing mismatch for {tile}: {baseline} vs SPAR")
    baseline_values = np.asarray([baseline_by_seed[seed] for seed in SEEDS])
    spar_values = np.asarray([spar_by_seed[seed] for seed in SEEDS])
    ratios = np.asarray([ratio_by_seed[seed] for seed in SEEDS])
    difference = baseline_values - spar_values
    n = len(difference)
    variance = float(np.var(difference, ddof=1))
    corrected_se = float(np.sqrt((1.0 / n + ratios.mean()) * variance))
    critical = float(stats.t.ppf(0.975, df=n - 1))
    corrected_t = (
        float(difference.mean() / corrected_se)
        if corrected_se > 0
        else float("inf")
    )
    wilcoxon = stats.wilcoxon(
        difference,
        alternative="greater",
        method="exact",
    )
    return {
        "tile": tile,
        "baseline": baseline,
        "candidate": "SPAR",
        "pair_count": n,
        "baseline_rmse_mean": float(baseline_values.mean()),
        "candidate_rmse_mean": float(spar_values.mean()),
        "ratio_of_means_reduction_percent": float(
            100.0
            * (baseline_values.mean() - spar_values.mean())
            / baseline_values.mean()
        ),
        "paired_difference_mean_mm": float(difference.mean()),
        "paired_difference_sd_mm": float(difference.std(ddof=1)),
        "wins": int(np.sum(difference > 0)),
        "ordinary_paired_t_two_sided_p": float(
            stats.ttest_rel(baseline_values, spar_values).pvalue
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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    consumed: dict[str, dict[str, object]] = {}
    for tile in TILES:
        for seed in SEEDS:
            seed_root = INPUT_ROOT / tile / f"seed_{seed}"
            task_path = seed_root / "task_metadata.json"
            run_path = seed_root / "run_manifest.json"
            task = read_json(task_path)
            assert isinstance(task, dict)
            train_count = int(task["raw_train_points"])
            test_count = int(task["raw_test_points"])
            for path in (task_path, run_path):
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                consumed[relative] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            for model_name, relative_dir in MODEL_PATHS.items():
                metrics_path = seed_root / relative_dir / "metrics.json"
                payload = read_json(metrics_path)
                assert isinstance(payload, dict)
                relative = metrics_path.relative_to(PROJECT_ROOT).as_posix()
                consumed[relative] = {
                    "size_bytes": metrics_path.stat().st_size,
                    "sha256": sha256(metrics_path),
                }
                rows.append(
                    {
                        "tile": tile,
                        "seed": seed,
                        "target_date": task["target_date"],
                        "model": model_name,
                        "rmse": rmse(payload),
                        "mae": mae(payload),
                        "bias": bias(payload),
                        "training_seconds": float(
                            payload.get("training_seconds", 0.0)
                        ),
                        "inference_seconds": float(
                            payload.get("inference_seconds", 0.0)
                        ),
                        "core_seconds": float(
                            payload.get(
                                "core_seconds",
                                float(payload.get("training_seconds", 0.0))
                                + float(payload.get("inference_seconds", 0.0)),
                            )
                        ),
                        "parameter_count": int(payload.get("parameter_count", 0)),
                        "train_cells": train_count,
                        "val_cells": int(task["raw_val_points"]),
                        "test_cells": test_count,
                        "test_train_ratio": float(test_count / train_count),
                    }
                )

    summaries: list[dict[str, object]] = []
    for tile in TILES:
        for model_name in MODEL_PATHS:
            selected = [
                row
                for row in rows
                if row["tile"] == tile and row["model"] == model_name
            ]
            summaries.append(
                {
                    "tile": tile,
                    "model": model_name,
                    "seed_count": len(selected),
                    "rmse_mean": float(np.mean([row["rmse"] for row in selected])),
                    "rmse_sd": float(
                        np.std([row["rmse"] for row in selected], ddof=1)
                    ),
                    "mae_mean": float(np.mean([row["mae"] for row in selected])),
                    "mae_sd": float(
                        np.std([row["mae"] for row in selected], ddof=1)
                    ),
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
        paired_statistics(rows, tile=tile, baseline=baseline)
        for tile in TILES
        for baseline in ("Persistence", "DLinear", "LASSO")
    ]

    anchor_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        confirmation_seed = CONFIRMATION_ROOT / f"seed_{seed}"
        ablation_seed = ANCHOR_ROOT / f"seed_{seed}"
        task_path = confirmation_seed / "task_metadata.json"
        anchored_path = (
            confirmation_seed
            / "spar_all_cells_uniform"
            / "metrics.json"
        )
        no_anchor_path = (
            ablation_seed
            / "spar_all_cells_uniform_no_anchor"
            / "metrics.json"
        )
        no_anchor_run_path = ablation_seed / "run_manifest.json"
        task = read_json(task_path)
        anchored = read_json(anchored_path)
        no_anchor = read_json(no_anchor_path)
        assert isinstance(task, dict)
        assert isinstance(anchored, dict)
        assert isinstance(no_anchor, dict)
        for path in (
            task_path,
            anchored_path,
            no_anchor_path,
            no_anchor_run_path,
        ):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            consumed[relative] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        ratio = float(task["raw_test_points"]) / float(task["raw_train_points"])
        for variant, payload in (
            ("SPAR", anchored),
            ("SPAR without anchor", no_anchor),
        ):
            anchor_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "rmse": rmse(payload),
                    "mae": mae(payload),
                    "bias": bias(payload),
                    "core_seconds": float(
                        payload.get(
                            "core_seconds",
                            float(payload.get("training_seconds", 0.0))
                            + float(payload.get("inference_seconds", 0.0)),
                        )
                    ),
                    "parameter_count": int(payload.get("parameter_count", 0)),
                    "test_train_ratio": ratio,
                }
            )
    anchor_summary: list[dict[str, object]] = []
    for variant in ("SPAR", "SPAR without anchor"):
        selected = [row for row in anchor_rows if row["variant"] == variant]
        anchor_summary.append(
            {
                "variant": variant,
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
    anchored_by_seed = {
        int(row["seed"]): float(row["rmse"])
        for row in anchor_rows
        if row["variant"] == "SPAR"
    }
    no_anchor_by_seed = {
        int(row["seed"]): float(row["rmse"])
        for row in anchor_rows
        if row["variant"] == "SPAR without anchor"
    }
    ratios = np.asarray(
        [
            float(
                next(
                    row["test_train_ratio"]
                    for row in anchor_rows
                    if int(row["seed"]) == seed
                )
            )
            for seed in SEEDS
        ]
    )
    anchored_values = np.asarray([anchored_by_seed[seed] for seed in SEEDS])
    no_anchor_values = np.asarray([no_anchor_by_seed[seed] for seed in SEEDS])
    anchor_difference = no_anchor_values - anchored_values
    anchor_variance = float(np.var(anchor_difference, ddof=1))
    anchor_corrected_se = float(
        np.sqrt((1.0 / len(SEEDS) + ratios.mean()) * anchor_variance)
    )
    anchor_critical = float(stats.t.ppf(0.975, df=len(SEEDS) - 1))
    anchor_corrected_t = float(anchor_difference.mean() / anchor_corrected_se)
    anchor_wilcoxon = stats.wilcoxon(
        anchor_difference,
        alternative="greater",
        method="exact",
    )
    anchor_statistics = {
        "baseline": "SPAR without anchor",
        "candidate": "SPAR",
        "pair_count": len(SEEDS),
        "no_anchor_rmse_mean": float(no_anchor_values.mean()),
        "anchored_rmse_mean": float(anchored_values.mean()),
        "ratio_of_means_reduction_percent": float(
            100.0
            * (no_anchor_values.mean() - anchored_values.mean())
            / no_anchor_values.mean()
        ),
        "paired_difference_mean_mm": float(anchor_difference.mean()),
        "paired_difference_sd_mm": float(anchor_difference.std(ddof=1)),
        "wins": int(np.sum(anchor_difference > 0)),
        "ordinary_paired_t_two_sided_p": float(
            stats.ttest_rel(no_anchor_values, anchored_values).pvalue
        ),
        "mean_test_train_ratio": float(ratios.mean()),
        "corrected_resampled_se_mm": anchor_corrected_se,
        "corrected_resampled_ci95_low_mm": float(
            anchor_difference.mean() - anchor_critical * anchor_corrected_se
        ),
        "corrected_resampled_ci95_high_mm": float(
            anchor_difference.mean() + anchor_critical * anchor_corrected_se
        ),
        "corrected_resampled_t_two_sided_p": float(
            2.0
            * stats.t.sf(abs(anchor_corrected_t), df=len(SEEDS) - 1)
        ),
        "exact_wilcoxon_greater_statistic": float(anchor_wilcoxon.statistic),
        "exact_wilcoxon_greater_p": float(anchor_wilcoxon.pvalue),
    }

    core_summary_path = CORE_ROOT / "summary.json"
    core_manifest_path = CORE_ROOT / "manifest.json"
    core_summary = read_json(core_summary_path)
    consumed[core_summary_path.relative_to(PROJECT_ROOT).as_posix()] = {
        "size_bytes": core_summary_path.stat().st_size,
        "sha256": sha256(core_summary_path),
    }
    consumed[core_manifest_path.relative_to(PROJECT_ROOT).as_posix()] = {
        "size_bytes": core_manifest_path.stat().st_size,
        "sha256": sha256(core_manifest_path),
    }
    final_summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "cageo-v23-final-evidence",
        "core": core_summary,
        "external_region_summary": summaries,
        "external_region_paired_statistics": paired,
        "anchor_ablation_summary": anchor_summary,
        "anchor_ablation_paired_statistics": anchor_statistics,
        "regional_scope": (
            "same-origin independently trained tile replications; "
            "not cross-region transfer or Europe-wide inference"
        ),
    }

    write_csv(OUTPUT / "external_region_rows.csv", rows)
    write_csv(OUTPUT / "external_region_summary.csv", summaries)
    write_csv(OUTPUT / "external_region_paired_statistics.csv", paired)
    write_csv(OUTPUT / "anchor_ablation_rows.csv", anchor_rows)
    write_csv(OUTPUT / "anchor_ablation_summary.csv", anchor_summary)
    write_csv(OUTPUT / "anchor_ablation_paired_statistics.csv", [anchor_statistics])
    (OUTPUT / "summary.json").write_text(
        json.dumps(final_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    output_files = {}
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            output_files[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "aggregator_sha256": sha256(Path(__file__)),
        "prelaunch_addendum_sha256": sha256(
            PROJECT_ROOT
            / "refine-logs"
            / "EXPERIMENT_PLAN_V23_REGIONAL_ADDENDUM_20260730_011500.md"
        ),
        "consumed_files": consumed,
        "output_files": output_files,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "row_count": len(rows),
                "summary_count": len(summaries),
                "paired_count": len(paired),
            }
        )
    )


if __name__ == "__main__":
    main()
