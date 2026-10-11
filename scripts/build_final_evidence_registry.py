#!/usr/bin/env python3
"""Build the canonical, paper-scoped evidence registry without recomputation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from final_synthesis_lib import (
    PROTECTED_FORMAL_SOURCES,
    REGISTRY_FIELDS,
    ROOT,
    as_bool,
    as_float,
    as_int,
    best_checkpoint,
    evidence_id,
    generated_at,
    markdown_table,
    normalize_registry_row,
    protocol_hash,
    read_csv_rows,
    read_json,
    relative_path,
    resolve_path,
    run_metrics,
    run_status,
    sha256_file,
    source_manifest,
    write_csv,
    write_json,
)


FORMAL_DECISIONS = {
    "gate2": "OFFICIAL_BASELINE_COMPLETE",
    "gate3b": "GATE3B_FULL_PASS",
    "gate4a": "ADAPTER_CAPACITY_SIGNAL_ONLY",
    "gate4b_expanded": "STUDENT_T_EXPANDED_SIGNAL_POSITIVE",
    "gate4b_cross_dataset": "STUDENT_T_CROSS_DATASET_SIGNAL_POSITIVE",
    "gate4b_multiseed": "STUDENT_T_MULTI_SEED_SIGNAL_POSITIVE",
    "gate4c": "CONFORMAL_MINIMAL_SIGNAL_NEGATIVE",
    "gate4c_mechanism": "MECHANISM_AUDIT_COMPLETE",
    "gate5a": "GATE5A_DISTRIBUTION_HEAD_ABLATION",
    "gate5b": "GATE5B_SECOND_BACKBONE_REPLICATION",
}

SOURCE_FILES = [
    "reports/latest_results.csv",
    "reports/gate2_paper_comparison.csv",
    "reports/gate3a_mask_inventory.csv",
    "results/gate3b_full_status.json",
    "results/gate3b_full_checkpoint_inventory.json",
    "results/gate3b_full_evaluator_parity.json",
    "reports/gate3b_full_results.csv",
    "reports/gate3b_degradation_summary.csv",
    "results/gate4a_status.json",
    "results/gate4a_formal_decision.json",
    "results/gate4a_mechanism_audit.json",
    "reports/gate4a_minimal_results.csv",
    "reports/gate4a_parameter_comparison.csv",
    "results/gate4b_corrected_minimal_decision.json",
    "results/gate4b_expanded_status.json",
    "results/gate4b_expanded_formal_decision.json",
    "reports/gate4b_expanded_results.csv",
    "reports/gate4b_expanded_condition_deltas.csv",
    "reports/gate4b_expanded_df_diagnostics.csv",
    "results/gate4b_cross_dataset_status.json",
    "results/gate4b_cross_dataset_formal_decision.json",
    "reports/gate4b_cross_dataset_results.csv",
    "reports/gate4b_cross_dataset_deltas.csv",
    "reports/gate4b_cross_dataset_df_diagnostics.csv",
    "results/gate4b_multiseed_status.json",
    "results/gate4b_multiseed_formal_decision.json",
    "reports/gate4b_multiseed_results.csv",
    "reports/gate4b_multiseed_paired_deltas.csv",
    "reports/gate4b_multiseed_dataset_summary.csv",
    "reports/gate4b_multiseed_interval_summary.csv",
    "reports/gate4b_multiseed_df_diagnostics.csv",
    "reports/gate4b_multiseed_mask_inventory.csv",
    "results/gate4c_status.json",
    "results/gate4c_formal_decision.json",
    "results/gate4c_split_audit.json",
    "results/gate4c_mask_inventory.json",
    "results/gate4c_mechanism_audit.json",
    "reports/gate4c_minimal_results.csv",
    "reports/gate4c_interval_metrics.csv",
    "reports/gate4c_conditioned_bin_metrics.csv",
    "reports/gate4c_calibration_quantiles.csv",
    "reports/gate4c_mechanism_native_shift.csv",
    "reports/gate4c_mechanism_expansion_cost.csv",
    "reports/gate4c_mechanism_counterfactual_shrinkage.csv",
    "reports/gate4c_mechanism_horizon_metrics.csv",
    "reports/gate4c_mechanism_channel_metrics.csv",
    "results/gate5a_status.json",
    "results/gate5a_formal_decision.json",
    "results/gate5a_mechanism_audit.json",
    "results/gate5a_efficiency.json",
    "reports/gate5a_distribution_results.csv",
    "reports/gate5a_pairwise_deltas.csv",
    "reports/gate5a_pit_metrics.csv",
    "reports/gate5a_qq_metrics.csv",
    "reports/gate5a_tail_exceedance.csv",
    "reports/gate5a_nll_gain_by_residual_bin.csv",
    "reports/gate5a_channel_tail_analysis.csv",
    "reports/gate5a_channel_tail_correlations.csv",
    "reports/gate5a_efficiency.csv",
    "results/gate5b_status.json",
    "results/gate5b_formal_decision.json",
    "reports/gate5b_patchtst_probability_results.csv",
    "reports/gate5b_patchtst_paired_deltas.csv",
    "reports/gate5b_patchtst_tail_nll_gain.csv",
]


def source_key(row: dict[str, Any], keys: list[str]) -> str:
    pairs = [f"{key}={row.get(key)}" for key in keys if row.get(key) not in (None, "")]
    return f"row={row.get('_row_number')};" + ";".join(pairs)


def checkpoint_fields(run_dir: Any, status: dict[str, Any], allow_head: bool = True) -> dict[str, Any]:
    checkpoint = best_checkpoint(run_dir) if allow_head else ""
    if not checkpoint:
        checkpoint = relative_path(status.get("checkpoint_path") or status.get("source_checkpoint"))
    source_checkpoint = relative_path(status.get("source_checkpoint") or status.get("checkpoint_path"))
    return {
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_checkpoint_path": source_checkpoint,
        "source_checkpoint_sha256": status.get("source_checkpoint_sha256")
        or status.get("checkpoint_sha256")
        or sha256_file(source_checkpoint),
    }


def protocol_fields(run_dir: Any, status: dict[str, Any], metric_file: str) -> dict[str, Any]:
    return {
        "training_protocol_hash": status.get("training_protocol_hash") or protocol_hash(run_dir, metric_file),
        "metric_protocol_hash": status.get("metric_protocol_hash") or sha256_file(metric_file),
    }


def mask_fields(status: dict[str, Any], gate3b: bool = False) -> dict[str, Any]:
    test_path = status.get("test_mask") or status.get("test_mask_path")
    test_sha = status.get("test_mask_sha256")
    if gate3b:
        test_path = status.get("mask_path")
        test_sha = status.get("mask_sha256")
    return {
        "validation_mask_path": relative_path(status.get("validation_mask") or status.get("validation_mask_path") or status.get("selection_mask_path")),
        "validation_mask_sha256": status.get("validation_mask_sha256") or status.get("selection_mask_sha256") or "",
        "calibration_mask_path": relative_path(status.get("calibration_mask") or status.get("calibration_mask_path")),
        "calibration_mask_sha256": status.get("calibration_mask_sha256") or "",
        "test_mask_path": relative_path(test_path),
        "test_mask_sha256": test_sha or "",
    }


def audit_fields(status: dict[str, Any], masked: bool | None) -> dict[str, Any]:
    return {
        "input_mask_applied": status.get("input_mask_applied", masked),
        "mask_application_verified": status.get("mask_application_verified", masked),
        "batch_y_modified": status.get("batch_y_modified", status.get("future_target_masked", False)),
        "freeze_backbone": status.get("freeze_backbone", True),
        "test_used_for_model_selection": status.get("test_used_for_model_selection", False),
        "calibration_used_for_model_selection": status.get("calibration_labels_seen_during_selection", False),
    }


def add(rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    row = normalize_registry_row(payload)
    row["evidence_id"] = evidence_id(row)
    rows.append(row)


def build_gate2(rows: list[dict[str, Any]]) -> None:
    source = "reports/latest_results.csv"
    for item in read_csv_rows(source):
        if item.get("mode") != "official_baseline" or item.get("status") != "SUCCESS":
            continue
        run_dir = relative_path(item["run_dir"])
        status = run_status(run_dir)
        metrics = run_metrics(run_dir)
        payload = {
            "stage": "gate2_clean_baseline",
            "gate": "Gate 2",
            "dataset": item["dataset"],
            "missing_type": "clean",
            "missing_rate": 0.0,
            "experiment_seed": as_int(item["seed"]),
            "seed_scope": "single_official_training_seed",
            "variant": "official_timeperceiver",
            "interval_method": "NA",
            "coverage_level": "NA",
            "is_formal": True,
            "is_smoke": False,
            "is_reused": False,
            "is_valid": True,
            "paper_evidence": True,
            "legacy_unmasked_invalid": False,
            "experiment_id": item["experiment_id"],
            "run_dir": run_dir,
            "source_report": source,
            "source_row_or_key": source_key(item, ["experiment_id"]),
            "mse": as_float(item["mse"]),
            "mae": as_float(item["mae"]),
            "rmse": as_float(item["rmse"]),
            "best_epoch": as_int(item["best_epoch"]),
            "input_mask_applied": False,
            "mask_application_verified": True,
            "batch_y_modified": False,
            "freeze_backbone": False,
            "test_used_for_model_selection": False,
            "calibration_used_for_model_selection": False,
            "status": status.get("status", item["status"]),
            "formal_decision": FORMAL_DECISIONS["gate2"],
            "notes": "Official clean-input TimePerceiver reproduction.",
        }
        payload.update(checkpoint_fields(run_dir, {**status, "checkpoint_path": metrics.get("checkpoint_path")}))
        payload.update(protocol_fields(run_dir, status, "scripts/gate2_official_runner.py"))
        add(rows, payload)


def build_gate3b(rows: list[dict[str, Any]]) -> None:
    source = "reports/gate3b_full_results.csv"
    for item in read_csv_rows(source):
        run_dir = relative_path(item["run_dir"])
        status = run_status(run_dir)
        baseline_run = relative_path(status.get("run_dir"))
        payload = {
            "stage": "gate3b_fixed_mask",
            "gate": "Gate 3b",
            "dataset": item["dataset"],
            "missing_type": item["missing_type"],
            "missing_rate": as_float(item["target_missing_rate"]),
            "experiment_seed": as_int(item["seed"]),
            "seed_scope": "single_fixed_test_mask_seed",
            "variant": "official_timeperceiver",
            "interval_method": "NA",
            "coverage_level": "NA",
            "is_formal": True,
            "is_smoke": False,
            "is_reused": as_bool(item.get("reused")) or False,
            "is_valid": item["status"] == "SUCCESS",
            "paper_evidence": item["status"] == "SUCCESS",
            "legacy_unmasked_invalid": False,
            "experiment_id": item["experiment_id"],
            "run_dir": run_dir,
            "source_report": source,
            "source_row_or_key": source_key(item, ["dataset", "missing_type", "target_missing_rate", "seed"]),
            "mse": as_float(item["mse"]),
            "mae": as_float(item["mae"]),
            "rmse": as_float(item["rmse"]),
            "status": item["status"],
            "formal_decision": FORMAL_DECISIONS["gate3b"],
            "notes": "Fixed Gate 3a test mask; historical input only is masked.",
        }
        payload.update(checkpoint_fields(run_dir, status))
        payload.update(mask_fields(status, gate3b=True))
        payload.update(audit_fields(status, masked=True))
        payload["training_protocol_hash"] = protocol_hash(baseline_run, "scripts/gate2_official_runner.py")
        payload["metric_protocol_hash"] = sha256_file("scripts/gate3b_official_missing_eval.py")
        add(rows, payload)


def build_gate4a(rows: list[dict[str, Any]]) -> None:
    source = "reports/gate4a_minimal_results.csv"
    for item in read_csv_rows(source):
        run_dir = relative_path(item["run_dir"])
        status = run_status(run_dir)
        variant = item["variant"]
        payload = {
            "stage": "gate4a_adapter_ablation",
            "gate": "Gate 4a",
            "dataset": "etth1",
            "missing_type": "random",
            "missing_rate": 0.3,
            "experiment_seed": 2025,
            "seed_scope": "single_training_seed",
            "variant": variant,
            "interval_method": "NA",
            "coverage_level": "NA",
            "is_formal": True,
            "is_smoke": False,
            "is_reused": variant == "official_baseline",
            "is_valid": True,
            "paper_evidence": True,
            "legacy_unmasked_invalid": False,
            "experiment_id": item["experiment_id"],
            "run_dir": run_dir,
            "source_report": source,
            "source_row_or_key": source_key(item, ["variant", "experiment_id"]),
            "mse": as_float(item["test_mse"]),
            "mae": as_float(item["test_mae"]),
            "rmse": as_float(item["test_rmse"]),
            "trainable_parameter_count": as_int(item["trainable_parameter_count"]),
            "total_parameter_count": as_int(item["total_parameter_count"]),
            "parameter_overhead_ratio": as_float(item["parameter_overhead_ratio"]),
            "best_epoch": as_int(item["best_epoch"]),
            "validation_metric": as_float(item["best_validation_mse"]),
            "status": "SUCCESS",
            "formal_decision": FORMAL_DECISIONS["gate4a"],
            "notes": "Frozen-backbone adapter ablation; Gate 4a conditioning signal is negative.",
        }
        payload.update(checkpoint_fields(run_dir, status))
        payload.update(mask_fields(status, gate3b=variant == "official_baseline"))
        payload.update(audit_fields(status, masked=True))
        payload.update(protocol_fields(run_dir, status, "scripts/gate4a_train_adapter.py"))
        add(rows, payload)


def build_legacy_gate4b(rows: list[dict[str, Any]]) -> None:
    source = "reports/gate4b_minimal_results.csv"
    for item in read_csv_rows(source):
        if item.get("variant") not in {"gaussian_head", "student_t_head"}:
            continue
        run_dir = relative_path(item["run_dir"])
        status = run_status(run_dir)
        payload = {
            "stage": "gate4b_legacy_unmasked_audit",
            "gate": "Gate 4b",
            "dataset": "etth1",
            "missing_type": "random",
            "missing_rate": 0.3,
            "experiment_seed": 2025,
            "seed_scope": "legacy_single_seed",
            "variant": item["variant"],
            "interval_method": "NA",
            "coverage_level": "NA",
            "is_formal": False,
            "is_smoke": False,
            "is_reused": False,
            "is_valid": False,
            "paper_evidence": False,
            "invalid_reason": "Historical Gate 4b minimal run did not apply the fixed input mask.",
            "legacy_unmasked_invalid": True,
            "experiment_id": item["experiment_id"],
            "run_dir": run_dir,
            "source_report": source,
            "source_row_or_key": source_key(item, ["variant", "experiment_id"]),
            "input_mask_applied": False,
            "mask_application_verified": False,
            "batch_y_modified": False,
            "freeze_backbone": True,
            "test_used_for_model_selection": False,
            "calibration_used_for_model_selection": False,
            "status": status.get("status", "SUCCESS"),
            "formal_decision": "LEGACY_UNMASKED_INVALID",
            "notes": "AUDIT_RETAINED; reusable=false; formal_evidence=false; paper_evidence=false.",
        }
        payload.update(checkpoint_fields(run_dir, status, allow_head=False))
        payload.update(protocol_fields(run_dir, status, "scripts/gate4b_train_probabilistic.py"))
        add(rows, payload)


def probability_payload(item: dict[str, Any], stage: str, gate: str, source: str, decision: str, seed_scope: str) -> dict[str, Any]:
    run_dir = relative_path(item["run_dir"])
    status = run_status(run_dir)
    dataset = item.get("dataset") or "etth1"
    experiment_seed = as_int(item.get("experiment_seed")) or 2025
    payload = {
        "stage": stage,
        "gate": gate,
        "dataset": dataset,
        "missing_type": item.get("missing_type") or status.get("missing_type") or "random",
        "missing_rate": as_float(item.get("target_missing_rate")) or as_float(status.get("target_missing_rate")) or 0.3,
        "experiment_seed": experiment_seed,
        "seed_scope": seed_scope,
        "variant": item["variant"],
        "interval_method": "native",
        "coverage_level": "NA",
        "is_formal": True,
        "is_smoke": False,
        "is_reused": as_bool(item.get("reused")) or False,
        "is_valid": item.get("status") == "SUCCESS" and not (as_bool(item.get("legacy_unmasked_invalid")) or False),
        "paper_evidence": item.get("status") == "SUCCESS" and not (as_bool(item.get("legacy_unmasked_invalid")) or False),
        "legacy_unmasked_invalid": as_bool(item.get("legacy_unmasked_invalid")) or False,
        "experiment_id": item["experiment_id"],
        "run_dir": run_dir,
        "source_report": source,
        "source_row_or_key": source_key(item, ["dataset", "variant", "condition", "experiment_seed"]),
        "mse": as_float(item.get("test_mse")),
        "mae": as_float(item.get("test_mae")),
        "rmse": as_float(item.get("test_rmse")),
        "nll": as_float(item.get("test_mean_nll")),
        "crps": as_float(item.get("test_crps")),
        "absolute_coverage_error": as_float(item.get("mean_absolute_coverage_error")),
        "mpiw": as_float(item.get("mean_interval_width")),
        "interval_score": as_float(item.get("mean_interval_score")),
        "mean_df": as_float(item.get("mean_df")),
        "median_df": as_float(item.get("median_df")),
        "min_df": as_float(item.get("min_df")),
        "max_df": as_float(item.get("max_df")),
        "trainable_parameter_count": as_int(item.get("trainable_parameter_count")),
        "total_parameter_count": as_int(item.get("total_parameter_count")),
        "parameter_overhead_ratio": as_float(item.get("parameter_overhead_ratio")),
        "best_epoch": as_int(item.get("best_epoch")),
        "validation_metric": as_float(item.get("best_validation_nll")),
        "status": item["status"],
        "formal_decision": decision,
        "notes": "Corrected fixed-mask probability-head evidence with a fixed official backbone.",
    }
    payload.update(checkpoint_fields(run_dir, status))
    payload.update(mask_fields(status))
    payload.update(audit_fields(status, masked=True))
    payload.update(protocol_fields(run_dir, status, "scripts/gate4b_train_probabilistic.py"))
    return payload


def build_probability_stages(rows: list[dict[str, Any]]) -> None:
    specs = [
        (
            "reports/gate4b_expanded_results.csv",
            "gate4b_expanded_conditions",
            FORMAL_DECISIONS["gate4b_expanded"],
            "single_probability_head_seed",
        ),
        (
            "reports/gate4b_cross_dataset_results.csv",
            "gate4b_cross_dataset",
            FORMAL_DECISIONS["gate4b_cross_dataset"],
            "single_probability_head_seed",
        ),
        (
            "reports/gate4b_multiseed_results.csv",
            "gate4b_multiseed",
            FORMAL_DECISIONS["gate4b_multiseed"],
            "probability_head_training_seed",
        ),
    ]
    for source, stage, decision, seed_scope in specs:
        for item in read_csv_rows(source):
            if item.get("variant") not in {"gaussian_head", "student_t_head"}:
                continue
            add(rows, probability_payload(item, stage, "Gate 4b", source, decision, seed_scope))


def build_gate5a(rows: list[dict[str, Any]]) -> None:
    decision_path = ROOT / "results/gate5a_formal_decision.json"
    source = "reports/gate5a_distribution_results.csv"
    if not decision_path.is_file() or not (ROOT / source).is_file():
        return
    decision = read_json(decision_path)
    if decision.get("GATE5A_IMPLEMENTATION_PASS") is not True:
        return
    label = decision.get("decision") or FORMAL_DECISIONS["gate5a"]
    for item in read_csv_rows(source):
        payload = probability_payload(
            item, "gate5a_distribution_ablation", "Gate 5a", source, label,
            "single_probability_head_training_seed",
        )
        payload["notes"] = (
            "Formal corrected fixed-mask distribution-head ablation. Associated PIT, QQ, tail, "
            "and channel diagnostics are secondary descriptive evidence only; no significance claim."
        )
        add(rows, payload)


def build_gate5b(rows: list[dict[str, Any]]) -> None:
    decision_path = ROOT / "results/gate5b_formal_decision.json"
    source = "reports/gate5b_patchtst_probability_results.csv"
    if not decision_path.is_file() or not (ROOT / source).is_file():
        return
    decision = read_json(decision_path)
    if decision.get("GATE5B_IMPLEMENTATION_PASS") is not True:
        return
    for item in read_csv_rows(source):
        payload = probability_payload(
            item,
            "gate5b_second_backbone",
            "Gate 5b",
            source,
            decision["decision"],
            "single_patchtst_backbone_and_probability_head_seed",
        )
        payload["source_checkpoint_path"] = relative_path(item["source_checkpoint"])
        payload["source_checkpoint_sha256"] = item["source_checkpoint_sha256"]
        payload["notes"] = (
            "PatchTST second-backbone fixed-mask replication. Fixed-t8 improves NLL and CRPS "
            "on all three datasets but misses the 1% point-noninferiority gate on Electricity."
        )
        add(rows, payload)


def build_gate4c(rows: list[dict[str, Any]]) -> None:
    point_source = "reports/gate4c_minimal_results.csv"
    point_rows = [
        item
        for item in read_csv_rows(point_source)
        if item.get("head") in {"gaussian_head", "student_t_head"} and item.get("method") == "native"
    ]
    status_by_head: dict[str, dict[str, Any]] = {}
    run_by_head: dict[str, str] = {}
    for item in point_rows:
        run_dir = relative_path(item["run_dir"])
        status = run_status(run_dir)
        status_by_head[item["head"]] = status
        run_by_head[item["head"]] = run_dir
        payload = {
            "stage": "gate4c_probability_head",
            "gate": "Gate 4c",
            "dataset": "etth1",
            "missing_type": "random",
            "missing_rate": 0.3,
            "experiment_seed": 2025,
            "seed_scope": "single_probability_head_seed",
            "variant": item["head"],
            "interval_method": "point_metrics",
            "coverage_level": "NA",
            "is_formal": True,
            "is_smoke": False,
            "is_reused": False,
            "is_valid": True,
            "paper_evidence": True,
            "legacy_unmasked_invalid": False,
            "experiment_id": item["experiment_id"],
            "run_dir": run_dir,
            "source_report": point_source,
            "source_row_or_key": source_key(item, ["head", "method"]),
            "mse": as_float(item["test_mse"]),
            "mae": as_float(item["test_mae"]),
            "rmse": as_float(item["test_rmse"]),
            "trainable_parameter_count": as_int(item["trainable_parameter_count"]),
            "best_epoch": as_int(item["best_epoch"]),
            "validation_metric": as_float(item["best_selection_nll"]),
            "status": "SUCCESS",
            "formal_decision": FORMAL_DECISIONS["gate4c"],
            "notes": "Independently retrained Gate 4c probability head under the strict raw-time split.",
        }
        payload.update(checkpoint_fields(run_dir, status))
        payload.update(mask_fields(status))
        payload.update(audit_fields(status, masked=True))
        payload.update(protocol_fields(run_dir, status, "scripts/gate4c_train_and_calibrate.py"))
        add(rows, payload)

    interval_source = "reports/gate4c_interval_metrics.csv"
    for item in read_csv_rows(interval_source):
        head = item["head"]
        status = status_by_head[head]
        run_dir = run_by_head[head]
        payload = {
            "stage": "gate4c_interval_calibration",
            "gate": "Gate 4c",
            "dataset": "etth1",
            "missing_type": "random",
            "missing_rate": 0.3,
            "experiment_seed": 2025,
            "seed_scope": "single_probability_head_seed",
            "variant": head,
            "interval_method": item["method"],
            "coverage_level": as_float(item["coverage"]),
            "is_formal": True,
            "is_smoke": False,
            "is_reused": False,
            "is_valid": True,
            "paper_evidence": True,
            "legacy_unmasked_invalid": False,
            "experiment_id": item["experiment_id"],
            "run_dir": run_dir,
            "source_report": interval_source,
            "source_row_or_key": source_key(item, ["head", "method", "coverage"]),
            "picp": as_float(item["picp"]),
            "mpiw": as_float(item["mpiw"]),
            "coverage_error": as_float(item["coverage_gap"]),
            "absolute_coverage_error": as_float(item["absolute_coverage_error"]),
            "interval_score": as_float(item["interval_score"]),
            "status": "SUCCESS",
            "formal_decision": FORMAL_DECISIONS["gate4c"],
            "notes": "Elementwise marginal interval metric; not joint coverage.",
        }
        payload.update(checkpoint_fields(run_dir, status))
        payload.update(mask_fields(status))
        payload.update(audit_fields(status, masked=True))
        payload.update(protocol_fields(run_dir, status, "scripts/gate4c_train_and_calibrate.py"))
        add(rows, payload)

    qhat_rows = read_csv_rows("reports/gate4c_calibration_quantiles.csv")
    expansion_rows = read_csv_rows("reports/gate4c_mechanism_expansion_cost.csv")
    oracle_rows = read_csv_rows("reports/gate4c_mechanism_counterfactual_shrinkage.csv")
    mechanism_source = "reports/gate4c_mechanism_native_shift.csv"
    for item in read_csv_rows(mechanism_source):
        head = item["head"]
        coverage = as_float(item["coverage"])
        qhat = next(row for row in qhat_rows if row["head"] == head and row["method"] == "marginal" and as_float(row["coverage"]) == coverage)
        expansion = next(row for row in expansion_rows if row["head"] == head and row["method"] == "marginal" and as_float(row["coverage"]) == coverage)
        oracle = next(row for row in oracle_rows if row["head"] == head and as_float(row["coverage"]) == coverage)
        status = status_by_head[head]
        run_dir = run_by_head[head]
        payload = {
            "stage": "gate4c_mechanism_audit",
            "gate": "Gate 4c",
            "dataset": "etth1",
            "missing_type": "random",
            "missing_rate": 0.3,
            "experiment_seed": 2025,
            "seed_scope": "single_probability_head_seed",
            "variant": head,
            "interval_method": "mechanism_audit",
            "coverage_level": coverage,
            "is_formal": True,
            "is_smoke": False,
            "is_reused": True,
            "is_valid": True,
            "paper_evidence": True,
            "legacy_unmasked_invalid": False,
            "experiment_id": status["experiment_id"],
            "run_dir": run_dir,
            "source_report": mechanism_source,
            "source_row_or_key": source_key(item, ["head", "coverage"]),
            "calibration_picp": as_float(item["calibration_picp"]),
            "test_picp": as_float(item["test_picp"]),
            "coverage_shift": as_float(item["coverage_shift_test_minus_calibration"]),
            "formal_qhat": as_float(qhat["q_hat"]),
            "delta_mpiw": as_float(expansion["width_increase"]),
            "delta_absolute_coverage_error": as_float(expansion["absolute_coverage_error_change"]),
            "oracle_signed_delta": as_float(oracle["oracle_signed_delta"]),
            "counterfactual_only": True,
            "uses_test_labels": True,
            "status": "SUCCESS",
            "formal_decision": FORMAL_DECISIONS["gate4c_mechanism"],
            "notes": "Oracle delta is COUNTERFACTUAL_ONLY; USES_TEST_LABELS; NOT_FORMAL_CALIBRATION.",
        }
        payload.update(checkpoint_fields(run_dir, status))
        payload.update(mask_fields(status))
        payload.update(audit_fields(status, masked=True))
        payload.update(protocol_fields(run_dir, status, "scripts/gate4c_mechanism_audit.py"))
        add(rows, payload)


def write_registry(rows: list[dict[str, Any]]) -> None:
    rows.sort(
        key=lambda row: (
            row["stage"],
            row["dataset"],
            str(row["missing_type"]),
            str(row["missing_rate"]),
            str(row["experiment_seed"]),
            row["variant"],
            row["interval_method"],
            str(row["coverage_level"]),
        )
    )
    payload = {
        "status": "BUILT_PENDING_VERIFICATION",
        "generated_at": generated_at(),
        "row_count": len(rows),
        "valid_formal_count": sum(bool(row["is_valid"] and row["is_formal"]) for row in rows),
        "invalid_legacy_count": sum(bool(row["legacy_unmasked_invalid"]) for row in rows),
        "paper_evidence_count": sum(bool(row["paper_evidence"]) for row in rows),
        "rows": rows,
    }
    write_json("results/final_evidence_registry.json", payload)
    write_csv("reports/final_evidence_registry.csv", rows, REGISTRY_FIELDS)
    selected_fields = [
        "evidence_id",
        "stage",
        "dataset",
        "missing_type",
        "missing_rate",
        "experiment_seed",
        "variant",
        "interval_method",
        "coverage_level",
        "is_valid",
        "paper_evidence",
        "experiment_id",
        "source_report",
        "status",
    ]
    text = "# Final Evidence Registry\n\n"
    text += f"Rows: {len(rows)}; valid formal: {payload['valid_formal_count']}; invalid legacy retained: {payload['invalid_legacy_count']}.\n\n"
    text += markdown_table(rows, selected_fields) + "\n"
    (ROOT / "reports/final_evidence_registry.md").write_text(text, encoding="utf-8")


def write_manifest(rows: list[dict[str, Any]]) -> None:
    dynamic_paths: set[str] = set(SOURCE_FILES)
    for row in rows:
        dynamic_paths.update(
            value
            for value in (
                row.get("source_report"),
                row.get("checkpoint_path"),
                row.get("source_checkpoint_path"),
                row.get("validation_mask_path"),
                row.get("calibration_mask_path"),
                row.get("test_mask_path"),
            )
            if value
        )
        run_dir = resolve_path(row.get("run_dir"))
        if run_dir is not None:
            for name in ("status.json", "metrics.json", "resolved_config.yaml"):
                candidate = run_dir / name
                if candidate.is_file():
                    dynamic_paths.add(relative_path(candidate))
    manifest = source_manifest(dynamic_paths)
    payload = {
        "status": "PASS" if len(manifest) == len({relative_path(path) for path in dynamic_paths if resolve_path(path) and resolve_path(path).is_file()}) else "FAILED",
        "generated_at": generated_at(),
        "source_count": len(manifest),
        "read_only_sources": True,
        "rows": manifest,
    }
    write_json("results/final_source_manifest.json", payload)
    write_csv(
        "reports/final_source_manifest.csv",
        manifest,
        ["relative_path", "sha256", "file_size", "modified_time", "stage", "purpose", "read_only_source"],
    )


def snapshot_protected_sources() -> None:
    target = ROOT / "results/final_formal_source_hashes_before.json"
    hashes = {path: sha256_file(path) for path in PROTECTED_FORMAL_SOURCES}
    if target.exists():
        existing = read_json(target)
        if existing.get("hashes") != hashes:
            raise RuntimeError("Protected formal sources differ from the retained pre-synthesis snapshot")
        return
    write_json(
        target,
        {
            "generated_at": generated_at(),
            "purpose": "Immutable pre-synthesis SHA256 snapshot",
            "hashes": hashes,
        },
    )


def main() -> None:
    snapshot_protected_sources()
    rows: list[dict[str, Any]] = []
    build_gate2(rows)
    build_gate3b(rows)
    build_gate4a(rows)
    build_legacy_gate4b(rows)
    build_probability_stages(rows)
    build_gate4c(rows)
    build_gate5a(rows)
    build_gate5b(rows)
    write_registry(rows)
    write_manifest(rows)
    print(json.dumps({"status": "BUILT_PENDING_VERIFICATION", "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
