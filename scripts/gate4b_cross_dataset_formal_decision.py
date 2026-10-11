#!/usr/bin/env python
"""Apply the predeclared Gate 4b cross-dataset gates."""
import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def main():
    rows = list(csv.DictReader((ROOT / "reports/gate4b_cross_dataset_results.csv").open()))
    deltas = list(csv.DictReader((ROOT / "reports/gate4b_cross_dataset_deltas.csv").open()))
    queue = load(ROOT / "configs/queues/gate4b_cross_dataset.yaml")
    state = load(ROOT / "runtime/gate4b_cross_dataset_scheduler_state.json")
    probability = [row for row in rows if row["variant"] != "official"]
    task_by_key = {task["task_id"]: task for task in queue["tasks"]}
    run_audits = []
    for row in probability:
        metrics = load(Path(row["run_dir"]) / "metrics.json")
        task = task_by_key[f"{row['dataset']}__random_rate0.3__{row['variant']}"]
        run_audits.append(
            metrics.get("input_mask_applied") is True
            and metrics.get("mask_application_verified") is True
            and metrics.get("legacy_unmasked_invalid") is False
            and metrics.get("source_checkpoint_sha256") == task["source_checkpoint_sha256"]
            and metrics.get("validation_mask_sha256") == task["validation_mask_sha256"]
            and metrics.get("test_mask_sha256") == task["test_mask_sha256"]
            and metrics.get("test_used_for_model_selection") is False
            and metrics.get("test_mask_used_during_training") is False
            and metrics.get("batch_y_modified") is False
            and metrics.get("optimizer_backbone_parameter_count") == 0
            and metrics.get("backbone_non_null_gradient_count") == 0
        )
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream/TimePerceiver"), "status", "--short"], text=True).splitlines()
    tracked_diff_empty = not any(not line.startswith("??") for line in tracked)
    implementation = (
        len(rows) == 6 and len(probability) == 4 and state.get("status") == "COMPLETE"
        and all(item.get("status") == "SUCCEEDED" for item in state["tasks"].values())
        and all(row["status"] == "SUCCESS" for row in probability) and all(run_audits)
        and all(math.isfinite(float(row[key])) for row in probability
                for key in ("test_mean_nll", "test_crps", "test_mse", "scale_min"))
        and all(float(row["scale_min"]) > 0 for row in probability)
        and all(float(row["min_df"]) > 2.01 for row in probability if row["variant"] == "student_t_head")
        and tracked_diff_empty
    )
    nll_count = sum(row["student_t_nll_win"] == "True" for row in deltas)
    crps_count = sum(row["student_t_crps_noninferior"] == "True" for row in deltas)
    point_count = sum(row["student_t_point_noninferior"] == "True" for row in deltas)
    dataset_pass_count = sum(
        row["student_t_nll_win"] == "True" and row["student_t_crps_noninferior"] == "True"
        and row["student_t_point_noninferior"] == "True" for row in deltas)
    signal = implementation and nll_count == 2 and crps_count == 2 and point_count == 2
    partial = implementation and dataset_pass_count == 1
    if not implementation:
        decision = "CROSS_DATASET_IMPLEMENTATION_INCOMPLETE"
    elif point_count < 2:
        decision = "CROSS_DATASET_PROBABILISTIC_POINT_TRADEOFF"
    elif signal:
        decision = "STUDENT_T_CROSS_DATASET_SIGNAL_POSITIVE"
    elif partial:
        decision = "STUDENT_T_PARTIAL_CROSS_DATASET_SIGNAL"
    elif nll_count == 2 and crps_count < 2:
        decision = "CROSS_DATASET_TAIL_LIKELIHOOD_ONLY"
    else:
        decision = "STUDENT_T_CROSS_DATASET_SIGNAL_NEGATIVE"
    output = {
        "status": decision,
        "GATE4B_CROSS_DATASET_IMPLEMENTATION_PASS": implementation,
        "GATE4B_CROSS_DATASET_NLL_WIN_COUNT": nll_count,
        "GATE4B_CROSS_DATASET_CRPS_NONINFERIOR_COUNT": crps_count,
        "GATE4B_CROSS_DATASET_POINT_NONINFERIOR_COUNT": point_count,
        "GATE4B_CROSS_DATASET_SIGNAL_POSITIVE": signal,
        "GATE4B_CROSS_DATASET_PARTIAL_SIGNAL": partial,
        "ENTER_GATE4B_MULTI_SEED_CONFIRMATION": implementation and signal,
        "ENTER_GATE4C": False,
        "seed_count": 1,
        "legacy_unmasked_results_excluded": True,
        "upstream_tracked_diff_empty": tracked_diff_empty,
    }
    (ROOT / "results/gate4b_cross_dataset_formal_decision.json").write_text(json.dumps(output, indent=2) + "\n")
    with (ROOT / "reports/gate4b_cross_dataset_formal_decision.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(output)
        writer.writerow(output.values())
    (ROOT / "reports/gate4b_cross_dataset_formal_decision.md").write_text(
        f"# Gate 4b Cross-Dataset Formal Decision\n\nDecision: **{decision}**\n\n"
        f"- NLL wins: `{nll_count}/2`\n- CRPS noninferior: `{crps_count}/2`\n"
        f"- Point noninferior: `{point_count}/2`\n"
        f"- Enter multi-seed confirmation: `{str(output['ENTER_GATE4B_MULTI_SEED_CONFIRMATION']).lower()}`\n"
        "- Enter Gate 4c: `false`\n\nSingle seed only; no significance claim.\n")
    status_path = ROOT / "results/gate4b_cross_dataset_status.json"
    status = load(status_path)
    status.update(output)
    status.update({
        "status": decision,
        "queue": "COMPLETE",
        "scheduler": "COMPLETE",
        "active_tasks": 0,
        "succeeded_tasks": 4,
        "waiting_resource_tasks": 0,
        "running_tasks": 0,
        "current_task": None,
        "current_required_free_mib": None,
        "status_verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    resume = status.get("formal_resume")
    if isinstance(resume, dict):
        resume.update({
            "status": "COMPLETE_4_OF_4_SUCCEEDED",
            "next_task": None,
            "completed_formal_tasks": [
                "weather__random_rate0.3__gaussian_head",
                "weather__random_rate0.3__student_t_head",
                "electricity__random_rate0.3__gaussian_head",
                "electricity__random_rate0.3__student_t_head",
            ],
            "remaining_formal_task": None,
            "running_experiment_id": None,
            "running_pid": None,
        })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    (ROOT / "reports/gate4b_cross_dataset_status.md").write_text(
        f"# Gate 4b Cross-Dataset Status\n\nDecision: **{decision}**\n\n"
        "All four probability tasks completed or were legally reused. Historical unmasked evidence was excluded.\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
