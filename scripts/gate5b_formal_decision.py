#!/usr/bin/env python3
"""Apply the pre-registered Gate 5b second-backbone decision rule."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import DATASETS, PATCHTST, TIMEPERCEIVER, dump, load, resolve, sha256_file, write_csv, write_report  # noqa: E402


def read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("nll_win", "crps_noninferior", "point_noninferior", "full_pair_pass"):
            row[key] = row[key].lower() == "true"
    return rows


def tracked_diff_empty(path):
    return subprocess.run(["git", "-C", str(path), "diff", "--quiet"]).returncode == 0


def completed_runs(state):
    if state.get("status") != "COMPLETE":
        return []
    runs = []
    for task in state.get("tasks", {}).values():
        if task.get("status") != "SUCCEEDED" or not task.get("run_dir"):
            return []
        runs.append(resolve(task["run_dir"]))
    return runs


def probability_run_checks(runs):
    required_finite = (
        "test_mse", "test_mae", "test_rmse", "test_mean_nll", "test_crps",
        "mean_absolute_coverage_error", "mean_interval_width", "mean_interval_score",
        "scale_min", "scale_max",
    )
    rows = []
    for run in runs:
        metrics = load(run / "metrics.json")
        status = load(run / "status.json")
        source = resolve(metrics["source_backbone_checkpoint"])
        checks = {
            "status_success": status.get("status") == "SUCCESS",
            "metrics_finite": all(math.isfinite(float(metrics[key])) for key in required_finite),
            "input_mask_applied": metrics.get("input_mask_applied") is True,
            "mask_application_verified": metrics.get("mask_application_verified") is True,
            "batch_y_unchanged": metrics.get("batch_y_modified") is False,
            "test_not_used_for_selection": metrics.get("test_used_for_model_selection") is False,
            "no_leakage": metrics.get("no_leakage") is True,
            "backbone_frozen": metrics.get("backbone_non_null_gradient_count") == 0,
            "optimizer_excludes_backbone": metrics.get("optimizer_backbone_parameter_count") == 0,
            "strict_reload": metrics.get("checkpoint_strict_reload") is True,
            "evaluation_after_reload": metrics.get("evaluation_after_reload") is True,
            "scale_finite_positive": metrics.get("scale_finite") is True and metrics.get("scale_positive") is True,
            "source_checkpoint_hash": source.is_file() and sha256_file(source) == metrics.get("source_backbone_checkpoint_sha256"),
        }
        if metrics.get("distribution") == "fixed_t8":
            checks["fixed_df_exact"] = metrics.get("fixed_df_exact") is True and metrics.get("fixed_df") == 8.0
            checks["fixed_df_parameter_count_zero"] = metrics.get("fixed_df_parameter_count") == 0
        rows.append({
            "experiment_id": run.name,
            "dataset": metrics.get("dataset"),
            "distribution": metrics.get("distribution"),
            "trainable_parameter_count": metrics.get("trainable_parameter_count"),
            "checks": checks,
            "pass": all(checks.values()),
        })
    return rows


def main():
    status = load(ROOT / "results/gate5b_status.json")
    if status.get("GATE5B_PROBABILITY_MATRIX_COMPLETE") is not True:
        raise SystemExit("Gate 5b probability matrix is not complete")
    rows = read_rows(ROOT / "reports/gate5b_patchtst_paired_deltas.csv")
    nll = sum(row["nll_win"] for row in rows)
    crps = sum(row["crps_noninferior"] for row in rows)
    point = sum(row["point_noninferior"] for row in rows)
    full = sum(row["full_pair_pass"] for row in rows)
    backbone_state = load(ROOT / "runtime/gate5b_patchtst_backbones_scheduler_state.json")
    probability_state = load(ROOT / "runtime/gate5b_patchtst_probability_scheduler_state.json")
    backbone_runs = completed_runs(backbone_state)
    probability_runs = completed_runs(probability_state)
    run_audit = probability_run_checks(probability_runs) if len(probability_runs) == 6 else []
    capacity = {}
    for dataset in DATASETS:
        local = [row for row in run_audit if row["dataset"] == dataset]
        capacity[dataset] = (
            len(local) == 2
            and {row["distribution"] for row in local} == {"gaussian", "fixed_t8"}
            and len({row["trainable_parameter_count"] for row in local}) == 1
        )
    gate5a = load(ROOT / "results/gate5a_formal_decision.json")
    implementation_checks = {
        "source_audit": load(ROOT / "results/gate5b_patchtst_source_audit.json").get("status") == "PASS",
        "environment_audit": load(ROOT / "results/gate5b_environment_audit.json").get("status") == "ENV_COMPATIBLE",
        "data_protocol_3_of_3": load(ROOT / "results/gate5b_data_protocol_validation.json").get("status") == "PASS",
        "backbone_smoke_3_of_3": load(ROOT / "results/gate5b_backbone_smoke.json").get("backbone_smoke_pass") == "3/3",
        "backbone_formal_3_of_3": len(backbone_runs) == 3,
        "distribution_validation": load(ROOT / "results/gate5b_distribution_validation.json").get("status") == "PASS",
        "pairing_3_of_3": load(ROOT / "results/gate5b_pairing_validation.json").get("pairs_pass") == "3/3",
        "probability_smoke_6_of_6": load(ROOT / "results/gate5b_probability_smoke.json").get("probability_smoke_pass") == "6/6",
        "probability_formal_6_of_6": len(probability_runs) == 6,
        "probability_run_audits": len(run_audit) == 6 and all(row["pass"] for row in run_audit),
        "capacity_fairness_3_of_3": all(capacity.values()),
        "paired_rows_3": len(rows) == 3 and {row["dataset"] for row in rows} == set(DATASETS),
        "timeperceiver_upstream_tracked_diff_empty": tracked_diff_empty(TIMEPERCEIVER),
        "patchtst_upstream_tracked_diff_empty": tracked_diff_empty(PATCHTST),
    }
    implementation = all(implementation_checks.values())
    signal = nll >= 2 and crps >= 2 and point == 3
    if not implementation:
        decision = "GATE5B_IMPLEMENTATION_FAILED"
        ready = False
    elif point < 3:
        decision = "SECOND_BACKBONE_POINT_TRADEOFF"
        ready = True
    elif signal:
        decision = "SECOND_BACKBONE_REPLICATION_POSITIVE"
        ready = True
    else:
        decision = "BACKBONE_DEPENDENT_HEAVY_TAIL_SIGNAL"
        ready = True
    payload = {
        "status": "PASS" if implementation else "FAILED", "decision": decision,
        "GATE5B_IMPLEMENTATION_PASS": implementation, "GATE5B_NLL_WIN_COUNT": nll,
        "GATE5B_CRPS_NONINFERIOR_COUNT": crps, "GATE5B_POINT_NONINFERIOR_COUNT": point,
        "GATE5B_FULL_PAIR_PASS_COUNT": full, "GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE": signal,
        "CROSS_BACKBONE_DIRECTION_CONSISTENT": gate5a.get("GATE5A_FIXED_T_SIGNAL_POSITIVE") is True and signal,
        "READY_FOR_PAPER_FINALIZATION": ready, "statistical_significance_claim": False,
        "scope": "two evaluated backbones; three datasets; random0.3; PatchTST seed2025",
        "implementation_checks": implementation_checks,
        "probability_run_audit": run_audit,
        "capacity_fairness": capacity,
        "paired_results": rows,
        "gate5a_fixed_heavy_tail_signal_positive": gate5a.get("GATE5A_FIXED_T_SIGNAL_POSITIVE") is True,
    }
    dump(ROOT / "results/gate5b_formal_decision.json", payload)
    write_csv(ROOT / "reports/gate5b_formal_decision.csv", [payload])
    write_report(ROOT / "reports/gate5b_formal_decision.md", "Gate 5b Formal Decision", payload)
    print(json.dumps(payload, indent=2))
    return 0 if implementation else 2


if __name__ == "__main__":
    raise SystemExit(main())
