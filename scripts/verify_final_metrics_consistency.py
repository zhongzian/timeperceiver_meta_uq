#!/usr/bin/env python3
"""Reproduce all formal counts and deltas from persisted metric artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from final_synthesis_lib import (
    PROTECTED_FORMAL_SOURCES,
    ROOT,
    as_bool,
    as_float,
    exact_close,
    generated_at,
    read_csv_rows,
    read_json,
    sample_mean,
    sample_std,
    sha256_file,
    write_json,
)


def by_stage(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["stage"] == stage]


def pair_rows(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    output: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        output[tuple(row[field] for field in group_fields)][row["variant"]] = row
    return output


def close_delta(student: dict[str, Any], gaussian: dict[str, Any], metric: str, expected: Any) -> bool:
    left, right = as_float(student.get(metric)), as_float(gaussian.get(metric))
    return left is not None and right is not None and exact_close(left - right, expected)


def main() -> None:
    registry = read_json("results/final_evidence_registry.json")
    rows = registry["rows"]
    gate_results: dict[str, dict[str, Any]] = {}

    # Gate 2: canonical rows must agree with the independent paper-comparison table.
    gate2 = by_stage(rows, "gate2_clean_baseline")
    paper_comparison = {row["dataset"]: row for row in read_csv_rows("reports/gate2_paper_comparison.csv")}
    gate2_checks = {
        "clean_rows_3_of_3": len(gate2) == 3 and {row["dataset"] for row in gate2} == {"etth1", "weather", "electricity"},
        "mse_matches_formal": all(exact_close(row["mse"], paper_comparison[row["dataset"]]["current_mse"]) for row in gate2),
        "mae_matches_formal": all(exact_close(row["mae"], paper_comparison[row["dataset"]]["current_mae"]) for row in gate2),
        "rmse_identity": all(exact_close(float(row["rmse"]) ** 2, row["mse"], tolerance=1e-7) for row in gate2),
    }
    gate_results["gate2_baseline"] = {"status": "PASS" if all(gate2_checks.values()) else "FAILED", "checks": gate2_checks}

    # Gate 3b: full factorial completeness, fixed-mask trace, and curve monotonicity.
    gate3b = by_stage(rows, "gate3b_fixed_mask")
    expected_cells = {
        (dataset, missing_type, rate)
        for dataset in ("etth1", "weather", "electricity")
        for missing_type in ("random", "block", "channel", "mixed")
        for rate in (0.1, 0.3, 0.5, 0.7)
    }
    actual_cells = {(row["dataset"], row["missing_type"], as_float(row["missing_rate"])) for row in gate3b}
    curves: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gate3b:
        curves[(row["dataset"], row["missing_type"])].append(row)
    monotonic_curves = {
        f"{key[0]}:{key[1]}": all(
            float(current["mse"]) <= float(following["mse"]) + 1e-12
            for current, following in zip(
                sorted(values, key=lambda item: float(item["missing_rate"])),
                sorted(values, key=lambda item: float(item["missing_rate"]))[1:],
            )
        )
        for key, values in curves.items()
    }
    gate3b_status = read_json("results/gate3b_full_status.json")
    gate3b_parity = read_json("results/gate3b_full_evaluator_parity.json")
    gate3b_checks = {
        "rows_48_of_48": len(gate3b) == 48,
        "factorial_cells_complete": actual_cells == expected_cells,
        "dataset_type_curves_12_complete": len(curves) == 12 and all(len(values) == 4 for values in curves.values()),
        "fixed_test_masks_verified": all(row["test_mask_sha256"] and row["mask_application_verified"] is True for row in gate3b),
        "all_status_success": all(row["status"] == "SUCCESS" for row in gate3b),
        "formal_status_pass": gate3b_status.get("GATE3B_FULL_PASS") is True and gate3b_status.get("succeeded") == 48,
        "evaluator_parity": gate3b_parity.get("status") == "PASS" and all(gate3b_parity.get(key) == 0 for key in ("mse_diff", "mae_diff", "rmse_diff")),
        "all_12_mse_curves_monotonic": all(monotonic_curves.values()),
    }
    gate_results["gate3b_48_mask_matrix"] = {
        "status": "PASS" if all(gate3b_checks.values()) else "FAILED",
        "checks": gate3b_checks,
        "monotonic_curves": monotonic_curves,
    }

    # Gate 4a: reproduce all decision deltas directly from its three rows.
    gate4a = {row["variant"]: row for row in by_stage(rows, "gate4a_adapter_ablation")}
    gate4a_decision = read_json("results/gate4a_formal_decision.json")
    official, static, aware = gate4a.get("official_baseline"), gate4a.get("static_adapter"), gate4a.get("missingness_aware_adapter")
    gate4a_checks = {
        "three_variants_found": set(gate4a) == {"official_baseline", "static_adapter", "missingness_aware_adapter"},
        "static_mse_delta": official is not None and static is not None and exact_close(float(static["mse"]) - float(official["mse"]), gate4a_decision["deltas"]["static_minus_official"]["mse"]),
        "missing_aware_official_delta": official is not None and aware is not None and exact_close(float(aware["mse"]) - float(official["mse"]), gate4a_decision["deltas"]["missing_aware_minus_official"]["mse"]),
        "missing_aware_static_delta": static is not None and aware is not None and exact_close(float(aware["mse"]) - float(static["mse"]), gate4a_decision["deltas"]["missing_aware_minus_static"]["mse"]),
        "static_beats_official": static is not None and official is not None and float(static["mse"]) < float(official["mse"]) and gate4a_decision["GATE4A_STATIC_ADAPTER_BEATS_BASELINE"] is True,
        "conditioning_does_not_beat_static": aware is not None and static is not None and float(aware["mse"]) > float(static["mse"]) and gate4a_decision["GATE4A_MISSINGNESS_CONDITIONING_BEATS_STATIC"] is False,
        "formal_decision_negative_conditioning": gate4a_decision["status"] == "ADAPTER_CAPACITY_SIGNAL_ONLY" and gate4a_decision["GATE4A_SIGNAL_POSITIVE"] is False,
    }
    gate_results["gate4a_decision"] = {"status": "PASS" if all(gate4a_checks.values()) else "FAILED", "checks": gate4a_checks}

    # Gate 4b expanded: six paired fixed-mask conditions.
    expanded = by_stage(rows, "gate4b_expanded_conditions")
    expanded_pairs = pair_rows(expanded, ("missing_type", "missing_rate"))
    expanded_deltas = read_csv_rows("reports/gate4b_expanded_condition_deltas.csv")
    expanded_decision = read_json("results/gate4b_expanded_formal_decision.json")
    expanded_delta_match = []
    for delta in expanded_deltas:
        missing_type, rate_text = delta["condition"].split("_rate")
        pair = expanded_pairs[(missing_type, float(rate_text))]
        expanded_delta_match.append(
            close_delta(pair["student_t_head"], pair["gaussian_head"], "nll", delta["delta_test_mean_nll"])
            and close_delta(pair["student_t_head"], pair["gaussian_head"], "crps", delta["delta_test_crps"])
            and close_delta(pair["student_t_head"], pair["gaussian_head"], "mse", delta["delta_test_mse"])
        )
    corrected = read_json("results/gate4b_corrected_minimal_decision.json")
    corrected_ids = {corrected["gaussian_run"].split("/")[-1], corrected["student_t_run"].split("/")[-1]}
    random03_ids = {row["experiment_id"] for row in expanded if row["missing_type"] == "random" and as_float(row["missing_rate"]) == 0.3}
    expanded_counts = {
        "nll": sum(as_bool(row["student_t_nll_win"]) is True for row in expanded_deltas),
        "crps": sum(as_bool(row["student_t_crps_noninferior"]) is True for row in expanded_deltas),
        "point": sum(as_bool(row["student_t_point_noninferior"]) is True for row in expanded_deltas),
    }
    expanded_checks = {
        "conditions_6_of_6": len(expanded_pairs) == 6 and all(len(pair) == 2 for pair in expanded_pairs.values()),
        "probability_rows_12": len(expanded) == 12,
        "deltas_reproduced": len(expanded_delta_match) == 6 and all(expanded_delta_match),
        "random03_corrected_fixed_mask": random03_ids == corrected_ids and corrected["corrected_fixed_mask_protocol"] is True,
        "old_unmasked_excluded": all(row["legacy_unmasked_invalid"] is False for row in expanded),
        "formal_counts_reproduced": expanded_counts == {
            "nll": expanded_decision["GATE4B_EXPANDED_NLL_WIN_COUNT"],
            "crps": expanded_decision["GATE4B_EXPANDED_CRPS_NONINFERIOR_COUNT"],
            "point": expanded_decision["GATE4B_EXPANDED_POINT_NONINFERIOR_COUNT"],
        },
        "formal_signal_positive": expanded_decision["GATE4B_EXPANDED_SIGNAL_POSITIVE"] is True,
    }
    gate_results["gate4b_expanded"] = {"status": "PASS" if all(expanded_checks.values()) else "FAILED", "checks": expanded_checks, "counts": expanded_counts}

    # Gate 4b cross-dataset: two paired datasets.
    cross = by_stage(rows, "gate4b_cross_dataset")
    cross_pairs = pair_rows(cross, ("dataset",))
    cross_deltas = read_csv_rows("reports/gate4b_cross_dataset_deltas.csv")
    cross_decision = read_json("results/gate4b_cross_dataset_formal_decision.json")
    cross_delta_match = []
    for delta in cross_deltas:
        pair = cross_pairs[(delta["dataset"],)]
        cross_delta_match.append(
            close_delta(pair["student_t_head"], pair["gaussian_head"], "nll", delta["delta_test_mean_nll"])
            and close_delta(pair["student_t_head"], pair["gaussian_head"], "crps", delta["delta_test_crps"])
            and close_delta(pair["student_t_head"], pair["gaussian_head"], "mse", delta["delta_test_mse"])
        )
    cross_counts = {
        "nll": sum(as_bool(row["student_t_nll_win"]) is True for row in cross_deltas),
        "crps": sum(as_bool(row["student_t_crps_noninferior"]) is True for row in cross_deltas),
        "point": sum(as_bool(row["student_t_point_noninferior"]) is True for row in cross_deltas),
    }
    cross_checks = {
        "weather_electricity_pairs": set(key[0] for key in cross_pairs) == {"weather", "electricity"} and all(len(pair) == 2 for pair in cross_pairs.values()),
        "probability_rows_4": len(cross) == 4,
        "deltas_reproduced": len(cross_delta_match) == 2 and all(cross_delta_match),
        "formal_counts_2_of_2": cross_counts == {"nll": 2, "crps": 2, "point": 2},
        "formal_decision_reproduced": cross_counts == {
            "nll": cross_decision["GATE4B_CROSS_DATASET_NLL_WIN_COUNT"],
            "crps": cross_decision["GATE4B_CROSS_DATASET_CRPS_NONINFERIOR_COUNT"],
            "point": cross_decision["GATE4B_CROSS_DATASET_POINT_NONINFERIOR_COUNT"],
        },
    }
    gate_results["gate4b_cross_dataset"] = {"status": "PASS" if all(cross_checks.values()) else "FAILED", "checks": cross_checks, "counts": cross_counts}

    # Gate 4b multi-seed: 24 rows, 12 pairs, exact paired deltas and ddof=1 summaries.
    multiseed = by_stage(rows, "gate4b_multiseed")
    multiseed_pairs = pair_rows(multiseed, ("dataset", "experiment_seed"))
    paired = read_csv_rows("reports/gate4b_multiseed_paired_deltas.csv")
    summary = read_csv_rows("reports/gate4b_multiseed_dataset_summary.csv")
    multiseed_decision = read_json("results/gate4b_multiseed_formal_decision.json")
    paired_match = []
    for delta in paired:
        pair = multiseed_pairs[(delta["dataset"], int(delta["experiment_seed"]))]
        paired_match.append(
            close_delta(pair["student_t_head"], pair["gaussian_head"], "nll", delta["delta_nll"])
            and close_delta(pair["student_t_head"], pair["gaussian_head"], "crps", delta["delta_crps"])
            and close_delta(pair["student_t_head"], pair["gaussian_head"], "mse", delta["delta_mse"])
        )
    summary_match = []
    for item in summary:
        if item["summary_type"] != "variant":
            continue
        subset = [row for row in multiseed if row["dataset"] == item["dataset"] and row["variant"] == item["variant"]]
        for metric in ("nll", "crps", "mse"):
            values = [float(row[metric]) for row in subset]
            summary_match.append(exact_close(sample_mean(values), item[f"{metric}_mean"]))
            summary_match.append(exact_close(sample_std(values), item[f"{metric}_std_ddof1"]))
    multiseed_counts = {
        "nll": sum(as_bool(row["nll_win"]) is True for row in paired),
        "crps": sum(as_bool(row["crps_noninferior"]) is True for row in paired),
        "point": sum(as_bool(row["point_noninferior"]) is True for row in paired),
        "full": sum(as_bool(row["full_pair_pass"]) is True for row in paired),
        "dataset_mean": sum(item.get("pass") is True for item in multiseed_decision["dataset_gates"]),
    }
    multiseed_checks = {
        "datasets_3": {row["dataset"] for row in multiseed} == {"etth1", "weather", "electricity"},
        "seeds_4": {int(row["experiment_seed"]) for row in multiseed} == {2021, 2022, 2023, 2025},
        "variants_2": {row["variant"] for row in multiseed} == {"gaussian_head", "student_t_head"},
        "probability_rows_24": len(multiseed) == 24,
        "paired_cells_12": len(multiseed_pairs) == 12 and all(len(pair) == 2 for pair in multiseed_pairs.values()),
        "paired_deltas_reproduced": len(paired_match) == 12 and all(paired_match),
        "sample_mean_std_ddof1_reproduced": len(summary_match) == 36 and all(summary_match),
        "formal_counts_12_12_12_12_3": multiseed_counts == {"nll": 12, "crps": 12, "point": 12, "full": 12, "dataset_mean": 3},
        "decision_counts_match": multiseed_counts == {
            "nll": multiseed_decision["GATE4B_MULTI_SEED_NLL_WIN_COUNT"],
            "crps": multiseed_decision["GATE4B_MULTI_SEED_CRPS_NONINFERIOR_COUNT"],
            "point": multiseed_decision["GATE4B_MULTI_SEED_POINT_NONINFERIOR_COUNT"],
            "full": multiseed_decision["GATE4B_MULTI_SEED_FULL_PAIR_PASS_COUNT"],
            "dataset_mean": multiseed_decision["GATE4B_MULTI_SEED_DATASET_MEAN_PASS_COUNT"],
        },
        "probability_seed_scope_only": multiseed_decision["FULL_END_TO_END_MULTI_SEED"] is False,
        "formal_signal_positive": multiseed_decision["GATE4B_MULTI_SEED_SIGNAL_POSITIVE"] is True,
    }
    gate_results["gate4b_multiseed"] = {"status": "PASS" if all(multiseed_checks.values()) else "FAILED", "checks": multiseed_checks, "counts": multiseed_counts}

    # Gate 4c: exact interval factorial, negative formal gate, and unchanged mechanism audit.
    gate4c = by_stage(rows, "gate4c_interval_calibration")
    gate4c_decision = read_json("results/gate4c_formal_decision.json")
    mechanism = read_json("results/gate4c_mechanism_audit.json")
    gate4c_cells = {(row["variant"], row["interval_method"], float(row["coverage_level"])) for row in gate4c}
    expected_gate4c_cells = {
        (head, method, coverage)
        for head in ("gaussian_head", "student_t_head")
        for method in ("native", "marginal", "conditioned")
        for coverage in (0.9, 0.95)
    }
    ace_worsened = []
    for head in ("gaussian_head", "student_t_head"):
        for coverage in (0.9, 0.95):
            native = next(row for row in gate4c if row["variant"] == head and row["interval_method"] == "native" and float(row["coverage_level"]) == coverage)
            for method in ("marginal", "conditioned"):
                candidate = next(row for row in gate4c if row["variant"] == head and row["interval_method"] == method and float(row["coverage_level"]) == coverage)
                ace_worsened.append(float(candidate["absolute_coverage_error"]) > float(native["absolute_coverage_error"]))
    before = read_json("results/final_formal_source_hashes_before.json")["hashes"]
    after = {path: sha256_file(path) for path in PROTECTED_FORMAL_SOURCES}
    gate4c_checks = {
        "interval_rows_12": len(gate4c) == 12,
        "head_method_level_factorial": gate4c_cells == expected_gate4c_cells,
        "formal_implementation_pass": gate4c_decision["GATE4C_IMPLEMENTATION_PASS"] is True,
        "formal_negative_signal": gate4c_decision["GATE4C_SIGNAL_POSITIVE"] is False and gate4c_decision["ENTER_GATE4C_EXPANDED_MATRIX"] is False,
        "conformal_ace_worse_in_all_8_comparisons": len(ace_worsened) == 8 and all(ace_worsened),
        "mechanism_audit_complete": mechanism["status"] == "COMPLETE",
        "mechanism_formal_gate_unchanged": mechanism["formal_gate_unchanged"] is True and mechanism["formal_artifacts"]["all_hashes_unchanged"] is True,
        "protected_formal_hashes_unchanged": before == after,
    }
    gate_results["gate4c_formal_and_mechanism"] = {"status": "PASS" if all(gate4c_checks.values()) else "FAILED", "checks": gate4c_checks}

    # Gate 5a: 3 x 4 distribution ablation, exact pairwise deltas, and fixed decision rules.
    gate5a = by_stage(rows, "gate5a_distribution_ablation")
    gate5a_pairs = pair_rows(gate5a, ("dataset",))
    gate5a_deltas = read_csv_rows("reports/gate5a_pairwise_deltas.csv")
    gate5a_decision = read_json("results/gate5a_formal_decision.json")
    expected_variants = {"gaussian_head", "laplace_head", "student_t_fixed_df8_head", "student_t_head"}
    delta_match = []
    comparison_map = {
        "Laplace vs Gaussian": ("laplace_head", "gaussian_head"),
        "Fixed-t vs Gaussian": ("student_t_fixed_df8_head", "gaussian_head"),
        "Learnable-t vs Fixed-t": ("student_t_head", "student_t_fixed_df8_head"),
        "Learnable-t vs Laplace": ("student_t_head", "laplace_head"),
    }
    for item in gate5a_deltas:
        candidate, reference = comparison_map[item["comparison"]]
        pair = gate5a_pairs[(item["dataset"],)]
        delta_match.append(
            close_delta(pair[candidate], pair[reference], "nll", item["delta_nll"])
            and close_delta(pair[candidate], pair[reference], "crps", item["delta_crps"])
            and close_delta(pair[candidate], pair[reference], "mse", item["delta_mse"])
        )
    truth = lambda value: str(value).lower() == "true"
    count_map = {}
    for comparison in comparison_map:
        subset = [row for row in gate5a_deltas if row["comparison"] == comparison]
        count_map[comparison] = {
            "nll": sum(truth(row["nll_win"]) for row in subset),
            "crps": sum(truth(row["crps_noninferior"]) for row in subset),
            "point": sum(truth(row["point_noninferior"]) for row in subset),
        }
    gate5a_checks = {
        "rows_12": len(gate5a) == 12,
        "datasets_3": set(key[0] for key in gate5a_pairs) == {"etth1", "weather", "electricity"},
        "four_heads_per_dataset": all(set(pair) == expected_variants for pair in gate5a_pairs.values()),
        "pairwise_rows_12": len(gate5a_deltas) == 12,
        "pairwise_deltas_reproduced": len(delta_match) == 12 and all(delta_match),
        "laplace_counts_match": count_map.get("Laplace vs Gaussian") == {
            "nll": gate5a_decision["GATE5A_LAPLACE_NLL_WIN_COUNT"],
            "crps": gate5a_decision["GATE5A_LAPLACE_CRPS_NONINFERIOR_COUNT"],
            "point": gate5a_decision["GATE5A_LAPLACE_POINT_NONINFERIOR_COUNT"],
        },
        "fixed_t_counts_match": count_map.get("Fixed-t vs Gaussian") == {
            "nll": gate5a_decision["GATE5A_FIXED_T_NLL_WIN_COUNT"],
            "crps": gate5a_decision["GATE5A_FIXED_T_CRPS_NONINFERIOR_COUNT"],
            "point": gate5a_decision["GATE5A_FIXED_T_POINT_NONINFERIOR_COUNT"],
        },
        "learned_df_counts_match": count_map.get("Learnable-t vs Fixed-t") == {
            "nll": gate5a_decision["GATE5A_LEARNED_DF_NLL_WIN_COUNT"],
            "crps": gate5a_decision["GATE5A_LEARNED_DF_CRPS_NONINFERIOR_COUNT"],
            "point": gate5a_decision["GATE5A_LEARNED_DF_POINT_NONINFERIOR_COUNT"],
        },
        "implementation_pass": gate5a_decision["GATE5A_IMPLEMENTATION_PASS"] is True,
        "mechanism_secondary_descriptive": read_json("results/gate5a_mechanism_audit.json").get("descriptive_only") is True,
        "efficiency_complete": read_json("results/gate5a_efficiency.json").get("status") == "PASS",
    }
    gate_results["gate5a_distribution_ablation"] = {
        "status": "PASS" if all(gate5a_checks.values()) else "FAILED",
        "checks": gate5a_checks,
        "counts": count_map,
    }

    # Gate 5b: exact PatchTST Gaussian/fixed-t8 pairs and pre-registered point tradeoff.
    gate5b = by_stage(rows, "gate5b_second_backbone")
    gate5b_pairs = pair_rows(gate5b, ("dataset",))
    gate5b_deltas = read_csv_rows("reports/gate5b_patchtst_paired_deltas.csv")
    gate5b_decision = read_json("results/gate5b_formal_decision.json")
    gate5b_delta_match = []
    for item in gate5b_deltas:
        pair = gate5b_pairs[(item["dataset"],)]
        gate5b_delta_match.append(
            close_delta(pair["student_t_fixed_df8_head"], pair["gaussian_head"], "nll", item["delta_nll"])
            and close_delta(pair["student_t_fixed_df8_head"], pair["gaussian_head"], "crps", item["delta_crps"])
            and close_delta(pair["student_t_fixed_df8_head"], pair["gaussian_head"], "mse", item["delta_mse"])
        )
    gate5b_counts = {
        "nll": sum(as_bool(row["nll_win"]) is True for row in gate5b_deltas),
        "crps": sum(as_bool(row["crps_noninferior"]) is True for row in gate5b_deltas),
        "point": sum(as_bool(row["point_noninferior"]) is True for row in gate5b_deltas),
        "full": sum(as_bool(row["full_pair_pass"]) is True for row in gate5b_deltas),
    }
    gate5b_checks = {
        "probability_rows_6": len(gate5b) == 6,
        "datasets_3": set(key[0] for key in gate5b_pairs) == {"etth1", "weather", "electricity"},
        "two_capacity_matched_heads_per_dataset": all(
            set(pair) == {"gaussian_head", "student_t_fixed_df8_head"}
            and pair["gaussian_head"]["trainable_parameter_count"]
            == pair["student_t_fixed_df8_head"]["trainable_parameter_count"]
            for pair in gate5b_pairs.values()
        ),
        "paired_rows_3": len(gate5b_deltas) == 3,
        "paired_deltas_reproduced": len(gate5b_delta_match) == 3 and all(gate5b_delta_match),
        "formal_counts_match": gate5b_counts == {
            "nll": gate5b_decision["GATE5B_NLL_WIN_COUNT"],
            "crps": gate5b_decision["GATE5B_CRPS_NONINFERIOR_COUNT"],
            "point": gate5b_decision["GATE5B_POINT_NONINFERIOR_COUNT"],
            "full": gate5b_decision["GATE5B_FULL_PAIR_PASS_COUNT"],
        },
        "implementation_pass": gate5b_decision["GATE5B_IMPLEMENTATION_PASS"] is True,
        "point_tradeoff_preserved": (
            gate5b_decision["decision"] == "SECOND_BACKBONE_POINT_TRADEOFF"
            and gate5b_decision["GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE"] is False
            and gate5b_counts == {"nll": 3, "crps": 3, "point": 2, "full": 2}
        ),
        "tail_diagnostic_secondary": (
            read_json("results/gate5b_tail_diagnostic.json").get("status") == "PASS"
            and read_json("results/gate5b_tail_diagnostic.json").get("formal_gate_evidence") is False
        ),
    }
    gate_results["gate5b_second_backbone"] = {
        "status": "PASS" if all(gate5b_checks.values()) else "FAILED",
        "checks": gate5b_checks,
        "counts": gate5b_counts,
    }

    passed = all(result["status"] == "PASS" for result in gate_results.values())
    payload = {
        "status": "PASS" if passed else "BLOCKED_FINAL_SYNTHESIS_METRIC_INCONSISTENCY",
        "FINAL_METRICS_CONSISTENCY_PASS": passed,
        "generated_at": generated_at(),
        "gate_results": gate_results,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
    }
    write_json("results/final_metrics_consistency.json", payload)
    lines = ["# Final Metrics Consistency", "", f"**Status: {payload['status']}**", ""]
    for name, result in gate_results.items():
        lines.extend([f"## {name}", "", f"Status: {result['status']}", ""])
        lines.extend(f"- {check}: {'PASS' if value else 'FAIL'}" for check, value in result["checks"].items())
        lines.append("")
    (ROOT / "reports/final_metrics_consistency.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gates": {key: value["status"] for key, value in gate_results.items()}}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
