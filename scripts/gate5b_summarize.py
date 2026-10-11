#!/usr/bin/env python3
"""Summarize completed Gate 5b backbone and probability queues."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import DATASETS, dump, load, relative, sha256_file, write_csv, write_report  # noqa: E402


def state(path):
    return load(ROOT / path) if (ROOT / path).is_file() else {"status": "NOT_STARTED", "tasks": {}}


def complete_runs(queue_state):
    if queue_state.get("status") != "COMPLETE":
        return None
    runs = {}
    for key, task in queue_state["tasks"].items():
        if task.get("status") != "SUCCEEDED" or not task.get("run_dir"):
            return None
        run = Path(task["run_dir"])
        if not run.is_absolute():
            run = ROOT / run
        runs[key] = run
    return runs


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |\n"
    separator = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n" for row in rows)
    return header + separator + body


def main():
    backbone_state = state("runtime/gate5b_patchtst_backbones_scheduler_state.json")
    probability_state = state("runtime/gate5b_patchtst_probability_scheduler_state.json")
    backbone_runs = complete_runs(backbone_state)
    probability_runs = complete_runs(probability_state)
    if backbone_runs is None:
        raise SystemExit("Gate 5b0 backbone queue is not COMPLETE")
    backbone_rows = []
    manifest_rows = {}
    for run in backbone_runs.values():
        metrics = load(run / "metrics.json")
        dataset = metrics["dataset"]
        checkpoint = run / "checkpoints/best_backbone.pt"
        row = {
            "dataset": dataset, "experiment_id": run.name, "best_epoch": metrics["best_epoch"],
            "clean_validation_mse": metrics["best_validation_mse"],
            "clean_test_mse": metrics["clean_test_mse"], "clean_test_mae": metrics["clean_test_mae"],
            "clean_test_rmse": metrics["clean_test_rmse"],
            "masked_random0.3_mse": metrics["masked_random0.3_test_mse"],
            "masked_random0.3_mae": metrics["masked_random0.3_test_mae"],
            "masked_random0.3_rmse": metrics["masked_random0.3_test_rmse"],
            "checkpoint": relative(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "status": "SUCCESS",
        }
        backbone_rows.append(row)
        manifest_rows[dataset] = {
            "experiment_id": run.name, "run_dir": relative(run), "checkpoint": relative(checkpoint),
            "checkpoint_sha256": row["checkpoint_sha256"], "best_epoch": row["best_epoch"],
            "selection_metric": "validation_mse", "test_used_for_model_selection": False,
            "strict_reload": metrics["checkpoint_strict_reload"],
        }
    backbone_rows.sort(key=lambda row: DATASETS.index(row["dataset"]))
    backbone_status = {
        "status": "PASS", "GATE5B_BACKBONE_PREREQUISITE_PASS": True,
        "tasks_succeeded": "3/3", "datasets": backbone_rows,
    }
    manifest = {
        "status": "PASS", "backbone": "patchtst", "datasets": manifest_rows,
        "selection_metric": "validation_mse", "test_used_for_model_selection": False,
    }
    dump(ROOT / "results/gate5b_patchtst_backbone_status.json", backbone_status)
    dump(ROOT / "results/gate5b_patchtst_backbone_manifest.json", manifest)
    write_csv(ROOT / "reports/gate5b_patchtst_clean_baselines.csv", backbone_rows)
    (ROOT / "reports/gate5b_patchtst_clean_baselines.md").write_text(
        "# Gate 5b PatchTST Clean Baselines\n\n" + markdown_table(backbone_rows, [
            "dataset", "best_epoch", "clean_test_mse", "clean_test_mae", "masked_random0.3_mse", "status"
        ])
    )

    if probability_runs is None:
        status_payload = {
            "status": "GATE5B0_COMPLETE_GATE5B1_PENDING", "backbone_tasks_succeeded": "3/3",
            "probability_tasks_succeeded": "0/6", "GATE5B_BACKBONE_PREREQUISITE_PASS": True,
            "GATE5B_PROBABILITY_MATRIX_COMPLETE": False,
        }
        dump(ROOT / "results/gate5b_status.json", status_payload)
        write_report(ROOT / "reports/gate5b_status.md", "Gate 5b Status", status_payload)
        print(json.dumps(status_payload, indent=2))
        return 0

    probability_rows = []
    for run in probability_runs.values():
        metrics = load(run / "metrics.json")
        probability_rows.append({
            "dataset": metrics["dataset"], "head": metrics["distribution"],
            "variant": metrics["variant"], "experiment_id": run.name, "run_dir": relative(run),
            "experiment_seed": metrics["experiment_seed"], "missing_type": "random", "missing_rate": 0.3,
            "nll": metrics["test_mean_nll"], "crps": metrics["test_crps"],
            "mse": metrics["test_mse"], "mae": metrics["test_mae"], "rmse": metrics["test_rmse"],
            "test_mean_nll": metrics["test_mean_nll"], "test_crps": metrics["test_crps"],
            "test_mse": metrics["test_mse"], "test_mae": metrics["test_mae"],
            "test_rmse": metrics["test_rmse"],
            "coverage_error": metrics["mean_absolute_coverage_error"],
            "interval_width": metrics["mean_interval_width"], "interval_score": metrics["mean_interval_score"],
            "mean_absolute_coverage_error": metrics["mean_absolute_coverage_error"],
            "mean_interval_width": metrics["mean_interval_width"],
            "mean_interval_score": metrics["mean_interval_score"],
            "best_epoch": metrics["best_epoch"],
            "best_validation_nll": metrics["best_validation_nll"],
            "trainable_parameters": metrics["trainable_parameter_count"],
            "trainable_parameter_count": metrics["trainable_parameter_count"],
            "source_checkpoint": metrics["source_backbone_checkpoint"],
            "source_checkpoint_sha256": metrics["source_backbone_checkpoint_sha256"],
            "checkpoint_sha256": metrics["source_backbone_checkpoint_sha256"],
            "validation_mask": metrics["validation_mask"],
            "validation_mask_sha256": metrics["validation_mask_sha256"],
            "test_mask": metrics["test_mask"], "test_mask_sha256": metrics["test_mask_sha256"],
            "training_protocol_hash": metrics["training_protocol_hash"],
            "metric_protocol_hash": metrics["metric_protocol_hash"],
            "input_mask_applied": metrics["input_mask_applied"],
            "mask_application_verified": metrics["mask_application_verified"],
            "batch_y_modified": metrics["batch_y_modified"], "freeze_backbone": True,
            "test_used_for_model_selection": metrics["test_used_for_model_selection"],
            "status": "SUCCESS",
        })
    probability_rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), row["head"]))
    paired = []
    for dataset in DATASETS:
        rows = {row["head"]: row for row in probability_rows if row["dataset"] == dataset}
        gaussian, fixed = rows["gaussian"], rows["fixed_t8"]
        paired.append({
            "dataset": dataset, "delta_nll": fixed["nll"] - gaussian["nll"],
            "delta_crps": fixed["crps"] - gaussian["crps"], "delta_mse": fixed["mse"] - gaussian["mse"],
            "delta_mae": fixed["mae"] - gaussian["mae"],
            "delta_coverage_error": fixed["coverage_error"] - gaussian["coverage_error"],
            "delta_interval_score": fixed["interval_score"] - gaussian["interval_score"],
            "nll_win": fixed["nll"] < gaussian["nll"],
            "crps_noninferior": fixed["crps"] <= gaussian["crps"],
            "point_noninferior": fixed["mse"] <= gaussian["mse"] * 1.01,
        })
        paired[-1]["full_pair_pass"] = all((
            paired[-1]["nll_win"], paired[-1]["crps_noninferior"], paired[-1]["point_noninferior"]
        ))
    write_csv(ROOT / "reports/gate5b_patchtst_probability_results.csv", probability_rows)
    write_csv(ROOT / "reports/gate5b_patchtst_paired_deltas.csv", paired)
    (ROOT / "reports/gate5b_patchtst_probability_results.md").write_text(
        "# Gate 5b PatchTST Probability Results\n\n" + markdown_table(
            probability_rows, ["dataset", "head", "nll", "crps", "mse", "mae", "coverage_error", "interval_score"]
        )
    )
    (ROOT / "reports/gate5b_patchtst_paired_deltas.md").write_text(
        "# Gate 5b PatchTST Paired Deltas\n\n" + markdown_table(
            paired, ["dataset", "delta_nll", "delta_crps", "delta_mse", "nll_win", "crps_noninferior", "point_noninferior"]
        )
    )
    status_payload = {
        "status": "MATRIX_COMPLETE", "backbone_tasks_succeeded": "3/3",
        "probability_tasks_succeeded": "6/6", "GATE5B_BACKBONE_PREREQUISITE_PASS": True,
        "GATE5B_PROBABILITY_MATRIX_COMPLETE": True,
    }
    dump(ROOT / "results/gate5b_status.json", status_payload)
    write_report(ROOT / "reports/gate5b_status.md", "Gate 5b Status", status_payload)
    print(json.dumps(status_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
