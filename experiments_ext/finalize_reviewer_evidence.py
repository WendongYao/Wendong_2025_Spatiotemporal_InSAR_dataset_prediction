"""Build deterministic reviewer-evidence tables from completed CAGEO artifacts.

The script is intentionally read-only with respect to experiment directories.  It
writes derived CSV/JSON tables to a separate output directory and tolerates an
unfinished queue by reporting missing inputs explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def as_rows(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        return [dict(payload)]
    raise TypeError(f"Unsupported JSON payload: {type(payload)!r}")


def rows_from_summary(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = as_rows(read_json(path))
    for row in rows:
        row["artifact"] = str(path.resolve())
    return rows


def rows_from_root(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("summary.json")) if root.exists() else []:
        rows.extend(rows_from_summary(path))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite(values: Iterable[object]) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def summarize_group(rows: list[dict[str, object]], metrics: list[str]) -> dict[str, object]:
    output: dict[str, object] = {"n": len(rows)}
    for metric in metrics:
        values = finite(row[metric] for row in rows if row.get(metric) is not None)
        if values.size:
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
            output[f"{metric}_min"] = float(values.min())
            output[f"{metric}_max"] = float(values.max())
    return output


def paired_summary(
    rows: list[dict[str, object]],
    *,
    baseline: str,
    candidate: str,
    metric: str,
) -> dict[str, object]:
    lookup: dict[tuple[str, int], float] = {}
    for row in rows:
        model = str(row.get("model", ""))
        if model not in {baseline, candidate} or row.get(metric) is None:
            continue
        seed = int(row.get("split_seed", row.get("seed", -1)))
        lookup[(model, seed)] = float(row[metric])
    seeds = sorted(
        seed
        for model, seed in lookup
        if model == baseline and (candidate, seed) in lookup
    )
    differences = np.asarray(
        [lookup[(candidate, seed)] - lookup[(baseline, seed)] for seed in seeds],
        dtype=np.float64,
    )
    result: dict[str, object] = {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "seeds": seeds,
        "n": len(seeds),
    }
    if not len(seeds):
        return result
    result.update(
        {
            "mean_candidate_minus_baseline": float(differences.mean()),
            "median_candidate_minus_baseline": float(np.median(differences)),
            "candidate_better_count": int(np.sum(differences < 0)),
            "candidate_worse_count": int(np.sum(differences > 0)),
        }
    )
    if len(seeds) > 1:
        t_result = stats.ttest_1samp(differences, 0.0)
        result["paired_t_pvalue_two_sided"] = float(t_result.pvalue)
        if not np.any(differences == 0):
            result["wilcoxon_pvalue_two_sided"] = float(
                stats.wilcoxon(differences, alternative="two-sided", method="exact").pvalue
            )
    return result


def paired_variant_summary(
    rows: list[dict[str, object]],
    *,
    baseline: str,
    candidate: str,
    model: str,
    metric: str,
) -> dict[str, object]:
    """Summarize a paired component deletion against the full model."""
    relabelled: list[dict[str, object]] = []
    for row in rows:
        if row.get("model") != model or row.get("variant") not in {baseline, candidate}:
            continue
        copied = dict(row)
        copied["model"] = copied["variant"]
        relabelled.append(copied)
    result = paired_summary(
        relabelled,
        baseline=baseline,
        candidate=candidate,
        metric=metric,
    )
    result["evaluated_model"] = model
    return result


def add_variant(rows: list[dict[str, object]], variant: str) -> list[dict[str, object]]:
    output = []
    for row in rows:
        copied = dict(row)
        copied["variant"] = variant
        output.append(copied)
    return output


def apply_seed42_support_metrics(rows: list[dict[str, object]], results_root: Path) -> None:
    """Overlay posthoc equal-cell metrics onto early seed-42 artifacts in memory."""
    path = results_root / "R011_measurement_support_analysis" / "metrics.json"
    if not path.exists():
        return
    aliases = {
        "hybrid_dense_target": "cnn_lstm_hybrid",
        "lasso_dense_target": "lasso",
        "hybrid_raw_supervised": "cnn_lstm_raw_supervised",
        "lasso_raw_supervised": "lasso_raw_supervised",
        "persistence": "persistence",
    }
    posthoc = {
        aliases[str(row["model"])]: row
        for row in as_rows(read_json(path))
        if str(row.get("model")) in aliases
    }
    metric_names = [
        "cell_count",
        "cell_mean_rmse_equal_cell",
        "cell_mean_mae_equal_cell",
        "cell_mean_rmse_point_weighted",
        "within_cell_target_rmse",
        "nearest_cell_point_rmse",
        "mse_decomposition_error",
        "points_per_cell_mean",
        "points_per_cell_p95",
    ]
    for row in rows:
        if int(row.get("split_seed", -1)) != 42:
            continue
        source = posthoc.get(str(row.get("model", "")))
        if source is None:
            continue
        for metric in metric_names:
            if row.get(metric) is None and source.get(metric) is not None:
                row[metric] = source[metric]


def build_depth_table(results_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    one_layer: list[dict[str, object]] = []
    one_layer.extend(rows_from_summary(results_root / "R011_raw_holdout_E32N34_seed42" / "summary.json"))
    one_layer.extend(rows_from_summary(results_root / "R011c_raw_supervised_E32N34_seed42" / "summary.json"))
    one_layer.extend(rows_from_root(results_root / "R032_E32N34_seeds43_46_main"))
    wanted = {"cnn_lstm_hybrid", "cnn_lstm_raw_supervised"}
    table = add_variant([row for row in one_layer if row.get("model") in wanted], "one_layer")
    two_layer = rows_from_root(results_root / "R042_two_layer_E32N34_seeds42_44")
    table.extend(add_variant([row for row in two_layer if row.get("model") in wanted], "two_layer"))
    apply_seed42_support_metrics(table, results_root)

    comparisons: list[dict[str, object]] = []
    for model in sorted(wanted):
        lookup: dict[tuple[str, int], dict[str, object]] = {}
        for row in table:
            if row.get("model") == model:
                lookup[(str(row["variant"]), int(row.get("split_seed", -1)))] = row
        for seed in sorted(seed for variant, seed in lookup if variant == "one_layer"):
            one = lookup.get(("one_layer", seed))
            two = lookup.get(("two_layer", seed))
            if not one or not two:
                continue
            comparisons.append(
                {
                    "model": model,
                    "seed": seed,
                    "one_layer_rmse": float(one["rmse"]),
                    "two_layer_rmse": float(two["rmse"]),
                    "two_minus_one_rmse": float(two["rmse"]) - float(one["rmse"]),
                    "one_layer_cell_rmse": one.get("cell_mean_rmse_equal_cell"),
                    "two_layer_cell_rmse": two.get("cell_mean_rmse_equal_cell"),
                    "one_layer_training_seconds": one.get("training_seconds"),
                    "two_layer_training_seconds": two.get("training_seconds"),
                    "one_layer_parameters": one.get("parameter_count"),
                    "two_layer_parameters": two.get("parameter_count"),
                    "one_layer_peak_gpu_mb": one.get("peak_gpu_memory_mb"),
                    "two_layer_peak_gpu_mb": two.get("peak_gpu_memory_mb"),
                }
            )
    return table, comparisons


def build_formulation_table(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(rows_from_summary(results_root / "R043_convlstm_raw_supervised_E32N34_seed42" / "summary.json"))
    rows.extend(rows_from_summary(results_root / "R040_formulation_ablation_E32N34_seed42" / "summary.json"))
    rows.extend(rows_from_root(results_root / "R040_formulation_E32N34_seeds43_44"))
    wanted = {
        "conv_lstm_raw_supervised",
        "conv_lstm_raw_residual",
        "conv_lstm_absolute_supervised",
    }
    filtered = [row for row in rows if row.get("model") in wanted]
    for row in filtered:
        model = str(row["model"])
        seed = int(row.get("split_seed", -1))
        if seed == 42 and model == "conv_lstm_raw_supervised":
            history = results_root / "R043_convlstm_raw_supervised_E32N34_seed42" / model / "training_history.csv"
        elif seed == 42:
            history = results_root / "R040_formulation_ablation_E32N34_seed42" / model / "training_history.csv"
        else:
            history = (
                results_root
                / "R040_formulation_E32N34_seeds43_44"
                / "E32N34"
                / f"seed_{seed}"
                / model
                / "training_history.csv"
            )
        if not history.exists():
            continue
        with history.open("r", encoding="utf-8", newline="") as handle:
            history_rows = list(csv.DictReader(handle))
        gradients = finite(
            item["gradient_l2_mean"]
            for item in history_rows
            if item.get("gradient_l2_mean") not in {None, ""}
        )
        row["epochs_run"] = len(history_rows)
        row["gradient_l2_mean_across_epochs"] = float(gradients.mean()) if gradients.size else float("nan")
        row["gradient_l2_median_across_epochs"] = float(np.median(gradients)) if gradients.size else float("nan")
        row["training_history_artifact"] = str(history.resolve())
    return filtered


def build_component_table(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(
        add_variant(
            rows_from_summary(results_root / "R011_raw_holdout_E32N34_seed42" / "summary.json"),
            "full_hybrid",
        )
    )
    rows.extend(
        add_variant(
            rows_from_summary(results_root / "R011c_raw_supervised_E32N34_seed42" / "summary.json"),
            "full_hybrid",
        )
    )
    multiseed_baseline = [
        row
        for row in rows_from_root(results_root / "R032_E32N34_seeds43_46_main")
        if int(row.get("split_seed", -1)) in {43, 44}
    ]
    rows.extend(add_variant(multiseed_baseline, "full_hybrid"))
    for root_name, variant in [
        ("R041_no_warm_seed42", "no_warm_start"),
        ("R041_no_recent_gate_seed42", "no_recent_gate"),
        ("R041_no_spatial_correction_seed42", "no_spatial_correction"),
    ]:
        rows.extend(add_variant(rows_from_summary(results_root / root_name / "summary.json"), variant))
    for root_name, variant in [
        ("R041_no_recent_gate_seeds43_44", "no_recent_gate"),
        ("R041_no_spatial_correction_seeds43_44", "no_spatial_correction"),
    ]:
        rows.extend(add_variant(rows_from_root(results_root / root_name), variant))
    wanted = {"cnn_lstm_hybrid", "cnn_lstm_raw_supervised"}
    filtered = [row for row in rows if row.get("model") in wanted]
    apply_seed42_support_metrics(filtered, results_root)
    return filtered


def build_synthetic_tables(
    results_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for root_name in [
        "R021_synthetic_scenarios_grid128",
        "R021_synthetic_stress_grid64",
        "R022_synthetic_operator_shift_grid64",
    ]:
        path = results_root / root_name / "combined_results.json"
        if path.exists():
            suite = as_rows(read_json(path))
            for row in suite:
                row["suite"] = root_name
                row["artifact"] = str(path.resolve())
            rows.extend(suite)

    grouping = [
        "suite",
        "scenario",
        "grid_size",
        "support_points",
        "observation_noise_std",
        "input_interpolation",
        "model",
    ]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in grouping)
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        output = dict(zip(grouping, key, strict=True))
        output.update(
            summarize_group(
                group_rows,
                [
                    "rmse",
                    "gradient_vector_rmse",
                    "extreme_change_rmse",
                    "peak_amplitude_absolute_error",
                    "near_support_rmse",
                    "middle_support_rmse",
                    "far_support_rmse",
                    "training_seconds",
                ],
            )
        )
        summary.append(output)
    config_fields = [
        "suite",
        "scenario",
        "grid_size",
        "support_points",
        "observation_noise_std",
        "input_interpolation",
        "data_seed",
        "split_seed",
    ]
    win_metrics = [
        "rmse",
        "gradient_vector_rmse",
        "extreme_change_rmse",
        "peak_amplitude_absolute_error",
        "near_support_rmse",
        "middle_support_rmse",
        "far_support_rmse",
    ]
    config_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        config_groups.setdefault(tuple(row.get(field) for field in config_fields), []).append(row)
    wins: list[dict[str, object]] = []
    for key, config_rows in sorted(config_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        for metric in win_metrics:
            eligible = [row for row in config_rows if row.get(metric) is not None and np.isfinite(float(row[metric]))]
            if not eligible:
                continue
            winner = min(eligible, key=lambda row: float(row[metric]))
            output = dict(zip(config_fields, key, strict=True))
            output.update(
                {
                    "metric": metric,
                    "winner_model": winner["model"],
                    "winner_value": float(winner[metric]),
                    "candidate_count": len(eligible),
                }
            )
            wins.append(output)
    return rows, summary, wins


def build_cost_table(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root_name in [
        "R011_raw_holdout_E32N34_seed42",
        "R011d_raw_lasso_E32N34_seed42",
        "R043_dense_baselines_E32N34_seed42",
        "R032_E32N34_seeds43_46_main",
    ]:
        rows.extend(rows_from_root(results_root / root_name))
    wanted = {"lasso_raw_supervised", "cnn_lstm_hybrid", "conv_lstm_residual", "simvp_style_residual"}
    filtered = [row for row in rows if row.get("model") in wanted]
    apply_seed42_support_metrics(filtered, results_root)
    output: list[dict[str, object]] = []
    for model in sorted(wanted):
        group = [row for row in filtered if row.get("model") == model]
        if not group:
            continue
        row = {"model": model}
        row.update(
            summarize_group(
                group,
                ["rmse", "cell_mean_rmse_equal_cell", "training_seconds", "inference_seconds"],
            )
        )
        row["parameter_count_max"] = max(int(item.get("parameter_count", 0) or 0) for item in group)
        peak = finite(item["peak_gpu_memory_mb"] for item in group if item.get("peak_gpu_memory_mb") is not None)
        row["peak_gpu_memory_mb_max"] = float(peak.max()) if peak.size else 0.0
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "R052_reviewer_evidence")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    depth_rows, depth_pairs = build_depth_table(args.results_root)
    formulation_rows = build_formulation_table(args.results_root)
    component_rows = build_component_table(args.results_root)
    synthetic_rows, synthetic_summary, synthetic_wins = build_synthetic_tables(args.results_root)
    cost_rows = build_cost_table(args.results_root)

    write_csv(args.output_root / "depth_ablation_all.csv", depth_rows)
    write_csv(args.output_root / "depth_ablation_pairs.csv", depth_pairs)
    write_csv(args.output_root / "formulation_ablation.csv", formulation_rows)
    write_csv(args.output_root / "component_ablation.csv", component_rows)
    write_csv(args.output_root / "synthetic_all.csv", synthetic_rows)
    write_csv(args.output_root / "synthetic_grouped.csv", synthetic_summary)
    write_csv(args.output_root / "synthetic_winners.csv", synthetic_wins)
    write_csv(args.output_root / "accuracy_cost_summary.csv", cost_rows)

    diagnostics = {
        "depth_pair_count": len(depth_pairs),
        "formulation_row_count": len(formulation_rows),
        "component_row_count": len(component_rows),
        "synthetic_row_count": len(synthetic_rows),
        "synthetic_winner_row_count": len(synthetic_wins),
        "missing_expected_artifacts": [
            str(path.resolve())
            for path in [
                args.results_root / "R042_two_layer_E32N34_seeds42_44" / "combined_results.json",
                args.results_root / "R040_formulation_E32N34_seeds43_44" / "combined_results.json",
                args.results_root / "R041_no_warm_seed42" / "summary.json",
                args.results_root / "R041_no_recent_gate_seed42" / "summary.json",
                args.results_root / "R041_no_recent_gate_seeds43_44" / "combined_results.json",
                args.results_root / "R041_no_spatial_correction_seed42" / "summary.json",
                args.results_root / "R041_no_spatial_correction_seeds43_44" / "combined_results.json",
                args.results_root / "R021_synthetic_scenarios_grid128" / "combined_results.json",
                args.results_root / "R021_synthetic_stress_grid64" / "combined_results.json",
                args.results_root / "R022_synthetic_operator_shift_grid64" / "combined_results.json",
                args.results_root / "R051_lightgbm_variance" / "lightgbm_variance.json",
            ]
            if not path.exists()
        ],
        "formulation_pairwise": [
            paired_summary(
                formulation_rows,
                baseline="conv_lstm_absolute_supervised",
                candidate=candidate,
                metric="rmse",
            )
            for candidate in ["conv_lstm_raw_supervised", "conv_lstm_raw_residual"]
        ],
        "component_pairwise": [
            paired_variant_summary(
                component_rows,
                baseline="full_hybrid",
                candidate=candidate,
                model="cnn_lstm_raw_supervised",
                metric=metric,
            )
            for candidate in ["no_recent_gate", "no_spatial_correction"]
            for metric in ["rmse", "cell_mean_rmse_equal_cell"]
        ],
    }
    lightgbm_path = args.results_root / "R051_lightgbm_variance" / "lightgbm_variance.json"
    if lightgbm_path.exists():
        diagnostics["lightgbm"] = read_json(lightgbm_path)
    (args.output_root / "evidence_summary.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
