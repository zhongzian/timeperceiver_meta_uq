#!/usr/bin/env python
"""Apply the preregistered Gate 4b multi-seed decision gates."""
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate4b_multiseed_common import DATASETS, VARIANTS, dump, load, write_csv


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def truth(value):
    return str(value).lower() == "true"


def main():
    status = load(ROOT / "results/gate4b_multiseed_status.json")
    raw = read_csv(ROOT / "reports/gate4b_multiseed_results.csv")
    pairs = read_csv(ROOT / "reports/gate4b_multiseed_paired_deltas.csv")
    summaries = read_csv(ROOT / "reports/gate4b_multiseed_dataset_summary.csv")
    probability = [row for row in raw if row["variant"] in VARIANTS]
    upstream = ROOT / "upstream/TimePerceiver"
    tracked_diff = subprocess.check_output(["git", "-C", str(upstream), "diff", "--stat"], text=True).strip()
    implementation_checks = {
        "datasets_complete_3_of_3": {row["dataset"] for row in probability} == set(DATASETS),
        "probability_tasks_complete_24_of_24": len(probability) == 24,
        "raw_rows_complete_27": len(raw) == 27,
        "paired_rows_complete_12": len(pairs) == 12,
        "all_probability_status_success": all(row["status"] == "SUCCESS" for row in probability),
        "corrected_fixed_mask_only": all(truth(row["corrected_fixed_mask_evidence"]) for row in probability),
        "old_unmasked_excluded": all(not truth(row["legacy_unmasked_invalid"]) for row in probability),
        "validation_nll_selection": all(not truth(row["test_used_for_model_selection"]) for row in probability),
        "test_mask_not_used_for_training": all(not truth(row["test_mask_used_during_training"]) for row in probability),
        "batch_y_unchanged": all(not truth(row["batch_y_modified"]) for row in probability),
        "input_mask_applied": all(truth(row["input_mask_applied"]) for row in probability),
        "backbone_frozen": all(int(float(row["backbone_non_null_gradient_count"])) == 0 for row in probability),
        "optimizer_excludes_backbone": all(int(float(row["optimizer_backbone_parameter_count"])) == 0 for row in probability),
        "all_core_metrics_finite": all(
            all(math.isfinite(float(row[key])) for key in (
                "test_mean_nll", "test_crps", "test_mse", "test_mae", "test_rmse",
                "mean_absolute_coverage_error", "mean_interval_width", "mean_interval_score",
            )) for row in probability
        ),
        "all_scales_positive": all(float(row["scale_min"]) > 0 for row in probability),
        "all_student_df_above_2_01": all(
            float(row["min_df"]) > 2.01 for row in probability if row["variant"] == "student_t_head"
        ),
        "scheduler_completion_gates": all(status.get("completion_gates", {}).values()),
        "upstream_tracked_diff_empty": tracked_diff == "",
    }
    implementation_pass = all(implementation_checks.values())
    nll_wins = sum(truth(row["nll_win"]) for row in pairs)
    crps_noninferior = sum(truth(row["crps_noninferior"]) for row in pairs)
    point_noninferior = sum(truth(row["point_noninferior"]) for row in pairs)
    full_pair_pass = sum(truth(row["full_pair_pass"]) for row in pairs)
    dataset_gates = []
    for dataset in DATASETS:
        gaussian = next(row for row in summaries if row["summary_type"] == "variant"
                        and row["dataset"] == dataset and row["variant"] == "gaussian_head")
        student = next(row for row in summaries if row["summary_type"] == "variant"
                       and row["dataset"] == dataset and row["variant"] == "student_t_head")
        paired = next(row for row in summaries if row["summary_type"] == "paired_delta"
                      and row["dataset"] == dataset)
        checks = {
            "mean_nll_win": float(student["nll_mean"]) < float(gaussian["nll_mean"]),
            "mean_crps_noninferior": float(student["crps_mean"]) <= float(gaussian["crps_mean"]),
            "mean_point_noninferior": float(student["mse_mean"]) <= float(gaussian["mse_mean"]) * 1.01,
            "nll_win_count_at_least_3_of_4": int(paired["nll_win_count"]) >= 3,
        }
        dataset_gates.append({
            "dataset": dataset,
            "gaussian_mean_nll": float(gaussian["nll_mean"]),
            "student_t_mean_nll": float(student["nll_mean"]),
            "gaussian_mean_crps": float(gaussian["crps_mean"]),
            "student_t_mean_crps": float(student["crps_mean"]),
            "gaussian_mean_mse": float(gaussian["mse_mean"]),
            "student_t_mean_mse": float(student["mse_mean"]),
            "nll_win_count": int(paired["nll_win_count"]),
            "checks": checks,
            "pass": all(checks.values()),
        })
    dataset_pass_count = sum(row["pass"] for row in dataset_gates)
    signal_positive = (
        implementation_pass
        and nll_wins >= 10
        and crps_noninferior >= 10
        and point_noninferior >= 11
        and full_pair_pass >= 9
        and dataset_pass_count == 3
    )
    if not implementation_pass:
        decision = "MULTI_SEED_IMPLEMENTATION_INCOMPLETE"
    elif signal_positive:
        decision = "STUDENT_T_MULTI_SEED_SIGNAL_POSITIVE"
    elif point_noninferior < 11:
        decision = "MULTI_SEED_PROBABILISTIC_POINT_TRADEOFF"
    elif dataset_pass_count == 3:
        decision = "STUDENT_T_MULTI_SEED_PARTIAL_CONSISTENCY"
    elif nll_wins >= 10 and crps_noninferior < 10:
        decision = "MULTI_SEED_TAIL_LIKELIHOOD_ONLY"
    elif dataset_pass_count in (1, 2):
        decision = "STUDENT_T_DATASET_DEPENDENT_MULTI_SEED_SIGNAL"
    else:
        decision = "STUDENT_T_MULTI_SEED_SIGNAL_NEGATIVE"
    payload = {
        "status": decision,
        "decision": decision,
        "scope": "PROBABILISTIC_HEAD_TRAINING_SEEDS",
        "FULL_END_TO_END_MULTI_SEED": False,
        "GATE4B_MULTI_SEED_IMPLEMENTATION_PASS": implementation_pass,
        "GATE4B_MULTI_SEED_NLL_WIN_COUNT": nll_wins,
        "GATE4B_MULTI_SEED_CRPS_NONINFERIOR_COUNT": crps_noninferior,
        "GATE4B_MULTI_SEED_POINT_NONINFERIOR_COUNT": point_noninferior,
        "GATE4B_MULTI_SEED_FULL_PAIR_PASS_COUNT": full_pair_pass,
        "GATE4B_MULTI_SEED_DATASET_MEAN_PASS_COUNT": dataset_pass_count,
        "GATE4B_MULTI_SEED_SIGNAL_POSITIVE": signal_positive,
        "READY_FOR_GATE4C_PLANNING": signal_positive,
        "ENTER_GATE4C": False,
        "implementation_checks": implementation_checks,
        "dataset_gates": dataset_gates,
        "thresholds": {
            "nll_wins": "at least 10 of 12",
            "crps_noninferior": "at least 10 of 12",
            "point_noninferior": "at least 11 of 12",
            "full_pair_pass": "at least 9 of 12",
            "dataset_mean_pass": "3 of 3",
        },
        "limitations": [
            "Probability-head training seeds only; backbone checkpoints are fixed.",
            "Validation and test masks are fixed across seeds; train masks vary with experiment seed.",
            "Gaussian and Student-t are paired within each dataset and seed.",
            "Only four head-training seeds are included; no statistical-significance claim is made.",
            "Historical unmasked Gate 4b results are excluded.",
            "No conformal calibration was run; native intervals are not calibrated intervals.",
        ],
        "statistical_significance_claim": False,
        "upstream_tracked_diff_empty": tracked_diff == "",
    }
    dump(ROOT / "results/gate4b_multiseed_formal_decision.json", payload)
    flat = {key: value for key, value in payload.items()
            if not isinstance(value, (dict, list))}
    write_csv(ROOT / "reports/gate4b_multiseed_formal_decision.csv", [flat])
    lines = [
        "# Gate 4b Multi-Seed Formal Decision", "", f"Decision: **{decision}**", "",
        "This is probability-head seed robustness with fixed official backbones, not full end-to-end multi-seed training.", "",
        "| Gate | Result |", "| --- | --- |",
        f"| Implementation | {implementation_pass} |",
        f"| NLL wins | {nll_wins}/12 |",
        f"| CRPS noninferior | {crps_noninferior}/12 |",
        f"| Point noninferior | {point_noninferior}/12 |",
        f"| Full pair pass | {full_pair_pass}/12 |",
        f"| Dataset mean pass | {dataset_pass_count}/3 |",
        f"| Strong signal | {signal_positive} |",
        f"| Ready for Gate 4c planning | {signal_positive} |",
        "| ENTER_GATE4C | False |", "", "No statistical-significance claim is made.", "",
        "```json", json.dumps(payload, indent=2), "```", "",
    ]
    (ROOT / "reports/gate4b_multiseed_formal_decision.md").write_text("\n".join(lines))
    status.update({
        "status": decision,
        "formal_decision": decision,
        "GATE4B_MULTI_SEED_IMPLEMENTATION_PASS": implementation_pass,
        "GATE4B_MULTI_SEED_SIGNAL_POSITIVE": signal_positive,
        "READY_FOR_GATE4C_PLANNING": signal_positive,
        "ENTER_GATE4C": False,
    })
    dump(ROOT / "results/gate4b_multiseed_status.json", status)
    (ROOT / "reports/gate4b_multiseed_status.md").write_text(
        f"# Gate 4b Multi-Seed Status\n\nStatus: **{decision}**\n\n"
        f"Implementation pass: {implementation_pass}; strong signal: {signal_positive}.\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
