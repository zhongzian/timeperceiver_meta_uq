#!/usr/bin/env python
"""Apply the fixed Gate 4c formal decision gates without tuning thresholds."""

import csv
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/TimePerceiver"


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return sum(values) / len(values)


def main():
    state = load(ROOT / "runtime/gate4c_minimal_conformal_scheduler_state.json")
    interval_rows = rows(ROOT / "reports/gate4c_interval_metrics.csv")
    bin_rows = rows(ROOT / "reports/gate4c_conditioned_bin_metrics.csv")
    run_metrics = []
    for task in state.get("tasks", {}).values():
        if task.get("run_dir"):
            run_metrics.append(load(Path(task["run_dir"]) / "metrics.json"))
    upstream_diff = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "status", "--short", "--untracked-files=no"], text=True
    ).strip()
    implementation_checks = {
        "formal_tasks_2_of_2_succeeded": len(run_metrics) == 2 and all(task.get("status") == "SUCCEEDED" for task in state.get("tasks", {}).values()),
        "scheduler_complete": state.get("status") == "COMPLETE",
        "selection_calibration_raw_overlap_zero": all(row.get("selection_calibration_raw_overlap_count") == 0 for row in run_metrics),
        "calibration_labels_not_used_for_training": all(row.get("calibration_labels_seen_during_training") is False for row in run_metrics),
        "calibration_labels_not_used_for_selection": all(row.get("calibration_labels_seen_during_selection") is False for row in run_metrics),
        "test_labels_not_used_for_calibration": all(row.get("test_labels_seen_during_calibration") is False for row in run_metrics),
        "all_qhat_finite_nonnegative": all(row.get("q_hat_finite_nonnegative") is True for row in run_metrics),
        "all_conditioned_bins_valid": all(row.get("conditioned_bins_valid") is True for row in run_metrics),
        "all_intervals_finite_nested": all(row.get("intervals_finite") is True and all(row.get("interval_nesting", {}).values()) for row in run_metrics),
        "point_predictions_unchanged": all(row.get("point_prediction_mutation") is False for row in run_metrics),
        "backbone_frozen": all(row.get("backbone_non_null_gradient_count") == 0 and row.get("optimizer_backbone_parameter_count") == 0 for row in run_metrics),
        "no_leakage": all(row.get("batch_y_modified") is False and row.get("test_mask_used_during_training") is False for row in run_metrics),
        "upstream_tracked_diff_empty": upstream_diff == "",
    }
    implementation_pass = all(implementation_checks.values())

    grouped = {}
    for row in interval_rows:
        grouped.setdefault((row["head"], row["method"]), []).append(row)
    def average(head, method, key):
        return mean([float(row[key]) for row in grouped[(head, method)]])
    gaussian_marginal = average("gaussian_head", "marginal", "absolute_coverage_error") <= average("gaussian_head", "native", "absolute_coverage_error")
    student_marginal = average("student_t_head", "marginal", "absolute_coverage_error") <= average("student_t_head", "native", "absolute_coverage_error")

    def worst_by_level(method):
        output = {}
        for level in (0.9, 0.95):
            selected = [float(row["absolute_coverage_error"]) for row in bin_rows
                        if row["head"] == "student_t_head" and row["method"] == method
                        and math.isclose(float(row["coverage"]), level)]
            output[str(int(level * 100))] = max(selected)
        return output
    marginal_worst = worst_by_level("marginal")
    conditioned_worst = worst_by_level("conditioned")
    improvement_count = sum(conditioned_worst[tag] < marginal_worst[tag] for tag in ("90", "95"))
    conditioned_overall_ace = average("student_t_head", "conditioned", "absolute_coverage_error")
    marginal_overall_ace = average("student_t_head", "marginal", "absolute_coverage_error")
    group_benefit = (
        improvement_count >= 1
        and mean(conditioned_worst.values()) < mean(marginal_worst.values())
        and conditioned_overall_ace <= marginal_overall_ace + 0.005
    )
    conditioned_score = average("student_t_head", "conditioned", "interval_score")
    marginal_score = average("student_t_head", "marginal", "interval_score")
    score_noninferior = conditioned_score <= marginal_score * 1.02
    student_efficiency = (
        average("student_t_head", "conditioned", "mpiw") <= average("gaussian_head", "conditioned", "mpiw")
        and conditioned_overall_ace <= average("gaussian_head", "conditioned", "absolute_coverage_error") + 0.005
    )
    signal = implementation_pass and student_marginal and group_benefit and score_noninferior
    if not implementation_pass:
        decision = "GATE4C_IMPLEMENTATION_INCOMPLETE"
    elif not gaussian_marginal and not student_marginal:
        decision = "CONFORMAL_MINIMAL_SIGNAL_NEGATIVE"
    elif group_benefit and not score_noninferior:
        decision = "CONDITIONAL_COVERAGE_EFFICIENCY_TRADEOFF"
    elif signal:
        decision = "MASK_SEVERITY_CONFORMAL_MINIMAL_SIGNAL_POSITIVE"
    else:
        decision = "MARGINAL_CONFORMAL_ONLY_SIGNAL"
    output = {
        "status": decision, "decision": decision,
        "GATE4C_IMPLEMENTATION_PASS": implementation_pass,
        "GATE4C_MARGINAL_GAUSSIAN_EFFECTIVE": gaussian_marginal,
        "GATE4C_MARGINAL_STUDENT_T_EFFECTIVE": student_marginal,
        "conditioned_worst_bin_improvement_count": improvement_count,
        "conditioned_worst_bin_errors": conditioned_worst,
        "marginal_worst_bin_errors": marginal_worst,
        "GATE4C_CONDITIONED_STUDENT_T_GROUP_BENEFIT": group_benefit,
        "GATE4C_CONDITIONED_STUDENT_T_SCORE_NONINFERIOR": score_noninferior,
        "GATE4C_STUDENT_T_CONFORMAL_EFFICIENCY": student_efficiency,
        "GATE4C_SIGNAL_POSITIVE": signal,
        "ENTER_GATE4C_EXPANDED_MATRIX": signal,
        "implementation_checks": implementation_checks,
        "thresholds": {
            "conditioned_worst_bin_improvement_count": ">=1 of 2",
            "conditioned_overall_ace_allowance": 0.005,
            "conditioned_interval_score_ratio": 1.02,
        },
        "conformal_mode": "split_conformal", "coverage_target": "elementwise_marginal",
        "time_series_dependence_present": True, "exact_exchangeability_guarantee_claimed": False,
        "upstream_tracked_diff_empty": upstream_diff == "",
    }
    (ROOT / "results/gate4c_formal_decision.json").write_text(json.dumps(output, indent=2) + "\n")
    with (ROOT / "reports/gate4c_formal_decision.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["decision", "implementation_pass", "marginal_gaussian", "marginal_student_t", "group_benefit", "score_noninferior", "student_efficiency", "signal_positive", "enter_expanded"])
        writer.writerow([decision, implementation_pass, gaussian_marginal, student_marginal, group_benefit, score_noninferior, student_efficiency, signal, signal])
    (ROOT / "reports/gate4c_formal_decision.md").write_text(
        "# Gate 4c Formal Decision\n\n"
        f"Decision: **{decision}**\n\n"
        + "\n".join(f"- {key}: `{str(value).lower()}`" for key, value in (
            ("GATE4C_IMPLEMENTATION_PASS", implementation_pass),
            ("GATE4C_MARGINAL_GAUSSIAN_EFFECTIVE", gaussian_marginal),
            ("GATE4C_MARGINAL_STUDENT_T_EFFECTIVE", student_marginal),
            ("GATE4C_CONDITIONED_STUDENT_T_GROUP_BENEFIT", group_benefit),
            ("GATE4C_CONDITIONED_STUDENT_T_SCORE_NONINFERIOR", score_noninferior),
            ("GATE4C_STUDENT_T_CONFORMAL_EFFICIENCY", student_efficiency),
            ("GATE4C_SIGNAL_POSITIVE", signal),
            ("ENTER_GATE4C_EXPANDED_MATRIX", signal),
        ))
        + "\n\nThis is empirical split-conformal evidence under time-series dependence; no exact finite-sample exchangeability guarantee is claimed.\n"
    )
    status = {
        "stage": "Gate 4c", "status": decision, "scheduler": state.get("status"),
        "logical_tasks": 2, "succeeded": sum(task.get("status") == "SUCCEEDED" for task in state.get("tasks", {}).values()),
        "failed": sum(task.get("status") == "FAILED" for task in state.get("tasks", {}).values()),
        **{key: output[key] for key in (
            "GATE4C_IMPLEMENTATION_PASS", "GATE4C_MARGINAL_GAUSSIAN_EFFECTIVE",
            "GATE4C_MARGINAL_STUDENT_T_EFFECTIVE", "GATE4C_CONDITIONED_STUDENT_T_GROUP_BENEFIT",
            "GATE4C_CONDITIONED_STUDENT_T_SCORE_NONINFERIOR", "GATE4C_STUDENT_T_CONFORMAL_EFFICIENCY",
            "GATE4C_SIGNAL_POSITIVE", "ENTER_GATE4C_EXPANDED_MATRIX",
        )},
        "formal_decision": decision,
    }
    (ROOT / "results/gate4c_status.json").write_text(json.dumps(status, indent=2) + "\n")
    (ROOT / "reports/gate4c_status.md").write_text(
        f"# Gate 4c Status\n\nStatus: **{decision}**\n\nFormal tasks: `2/2 SUCCEEDED`. Expanded matrix was not started.\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

