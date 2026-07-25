"""Aggregate CAGEO result artifacts and compute paired, auditable statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


def load_rows(roots: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_files: set[Path] = set()
    for root in roots:
        paths = [root] if root.name == "summary.json" else sorted(root.rglob("summary.json"))
        for path in paths:
            resolved = path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = [payload]
            for item in payload:
                row = dict(item)
                row["artifact"] = str(resolved)
                rows.append(row)
    return rows


def paired_bootstrap_ci(differences: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    boot = differences[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_statistics(
    rows: list[dict[str, object]],
    *,
    baseline: str,
    candidate: str,
    metric: str,
    bootstrap_samples: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    lookup: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in rows:
        model = str(row.get("model", ""))
        if model not in {baseline, candidate} or metric not in row:
            continue
        tile = str(row.get("tile", "unknown"))
        seed = int(row.get("split_seed", -1))
        key = (tile, seed, model)
        if key in lookup and not np.isclose(float(lookup[key][metric]), float(row[metric])):
            raise ValueError(f"Conflicting duplicate for {key}: {lookup[key]['artifact']} vs {row['artifact']}")
        lookup[key] = row

    pair_rows: list[dict[str, object]] = []
    pair_keys = sorted({(tile, seed) for tile, seed, model in lookup if model == baseline})
    for tile, seed in pair_keys:
        base = lookup.get((tile, seed, baseline))
        cand = lookup.get((tile, seed, candidate))
        if base is None or cand is None:
            continue
        baseline_value = float(base[metric])
        candidate_value = float(cand[metric])
        pair_rows.append(
            {
                "tile": tile,
                "seed": seed,
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "candidate_minus_baseline": candidate_value - baseline_value,
                "candidate_relative_improvement_pct": 100.0 * (baseline_value - candidate_value) / baseline_value,
                "baseline_artifact": base["artifact"],
                "candidate_artifact": cand["artifact"],
            }
        )
    if not pair_rows:
        return [], {"n": 0, "reason": "no matched pairs"}

    differences = np.asarray([row["candidate_minus_baseline"] for row in pair_rows], dtype=np.float64)
    stats_payload: dict[str, object] = {
        "n": len(differences),
        "mean_candidate_minus_baseline": float(differences.mean()),
        "std_candidate_minus_baseline": float(differences.std(ddof=1)) if len(differences) > 1 else float("nan"),
        "median_candidate_minus_baseline": float(np.median(differences)),
        "candidate_better_count": int(np.sum(differences < 0)),
        "candidate_worse_count": int(np.sum(differences > 0)),
        "candidate_tie_count": int(np.sum(differences == 0)),
        "mean_relative_improvement_pct": float(
            np.mean([row["candidate_relative_improvement_pct"] for row in pair_rows])
        ),
    }
    stats_payload["bootstrap_95pct_ci_mean_difference"] = paired_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        seed=20260724,
    )
    if len(differences) > 1:
        t_result = stats.ttest_1samp(differences, popmean=0.0)
        stats_payload["paired_t_statistic"] = float(t_result.statistic)
        stats_payload["paired_t_pvalue_two_sided"] = float(t_result.pvalue)
        std = float(differences.std(ddof=1))
        stats_payload["cohens_dz"] = float(differences.mean() / std) if std > 0 else float("inf")
        if not np.any(differences == 0):
            wilcoxon = stats.wilcoxon(differences, alternative="two-sided", method="exact")
            stats_payload["wilcoxon_statistic"] = float(wilcoxon.statistic)
            stats_payload["wilcoxon_pvalue_two_sided"] = float(wilcoxon.pvalue)
    return pair_rows, stats_payload


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline", default="lasso")
    parser.add_argument("--candidate", default="cnn_lstm_hybrid")
    parser.add_argument("--metric", default="rmse")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.result_roots)
    for row in rows:
        if row.get("within_cell_target_rmse") is not None and row.get("rmse") is not None:
            row["within_cell_mse_fraction_of_point_mse"] = (
                float(row["within_cell_target_rmse"]) ** 2 / float(row["rmse"]) ** 2
            )
    pairs, stat_payload = paired_statistics(
        rows,
        baseline=args.baseline,
        candidate=args.candidate,
        metric=args.metric,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_csv(args.output_root / "all_results.csv", rows)
    write_csv(args.output_root / "paired_results.csv", pairs)
    (args.output_root / "paired_statistics.json").write_text(
        json.dumps(stat_payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(stat_payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
