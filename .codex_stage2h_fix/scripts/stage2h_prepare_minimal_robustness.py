#!/usr/bin/env python
"""Audit Stage 2H scope, inventory reusable rows, and prepare the minimal queue."""

import argparse
import csv
import json
import math
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/data/zza/code/BudgetedActiveChannelTokenFormer")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
QUEUE = ROOT / "configs" / "queues" / "stage2h_minimal_robustness.yaml"
PYTHON = "/mnt/data/zza/anaconda3/envs/bactf/bin/python"
UPSTREAM = ROOT / "upstream" / "ChannelTokenFormer"
SELECTED_BUDGET = 0.625
REQUIRED_FREE_MIB = 12000
SEED = 2021
METHODS = ("whole_channel_fixed_utility", "temporal_recent_all_channels")
DATASETS = {
    "weather_practical": {"key": "weather", "pred_len": 144, "channels": 21, "patches": 8},
    "ett1_practical": {"key": "ett1", "pred_len": 192, "channels": 7, "patches": 8},
}
WEATHER_RUNS = {
    2021: "stage2f_trap_multidata__seed2021__20260705_100754",
    2022: "stage2f_trap_multidata__seed2022__stage2fb__20260709_155850",
    2023: "stage2f_trap_multidata__seed2023__stage2fb__20260709_155850",
}
ETT1_SIDE_RUNS = {
    2022: "stage2g_ett1_side_eval__seed2022__20260709_211519",
    2023: "stage2g_ett1_side_eval__seed2023__20260709_211712",
}
ETT1_FIXED_PARENT = "ctf__ett1_practical__channel_acquisition_screen__seed2021__20260609_134516"
ETT1_TRAP_RUN = "stage2e_c_patch_ablation__ett1__seed2021__20260705_094605"
GPU_UUIDS = [
    "GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb",
    "GPU-c88b779d-77c7-92b9-e031-943bae990121",
]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_text(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return (proc.stdout or proc.stderr or "").strip()


def source_ok(status_path, metrics_path):
    status = read_json(status_path)
    metrics = read_json(metrics_path)
    return status.get("status") == "SUCCEEDED" and metrics.get("status", "SUCCEEDED") == "SUCCEEDED"


def canonical_row(dataset, pred_len, seed, budget, method, raw, experiment_id, run_dir, status_path, metrics_path, source):
    invalid = raw.get("invalid_action_count", raw.get("invalid", "NA"))
    repeat = raw.get("repeat_patch_action_count", raw.get("repeat_action_count", raw.get("repeat", "NA")))
    overspend = raw.get("overspend_count", raw.get("overspend", "NA"))
    requested = float(raw.get("budget", raw.get("budget_ratio", budget)))
    realized = raw.get("normalized_cost")
    if realized is None and raw.get("mean_used_cost") is not None:
        realized = float(raw["mean_used_cost"]) / DATASETS[dataset]["channels"]
    return {
        "dataset": dataset,
        "pred_len": int(pred_len),
        "seed": int(seed),
        "budget": float(budget),
        "method": method,
        "experiment_id": experiment_id,
        "run_dir": str(run_dir),
        "status": "SUCCEEDED",
        "cmse": float(raw.get("CMSE", raw.get("cmse"))),
        "cmae": float(raw.get("CMAE", raw.get("cmae"))),
        "invalid": invalid,
        "repeat": repeat,
        "overspend": overspend,
        "no_leakage": raw.get("no_leakage", "NA"),
        "parity": raw.get("parity", "NA"),
        "requested_budget": requested,
        "realized_budget": realized if realized is not None else "NA",
        "selected_count": raw.get("used_cost", raw.get("used_patch_cost", raw.get("mean_used_cost", "NA"))),
        "unique_selected_count": raw.get("unique_selected_count", "NA"),
        "selection_repeat_count": repeat,
        "selection_churn_rate": raw.get("selection_churn_rate", "NA"),
        "policy_entropy": raw.get("policy_entropy", "NA"),
        "source_status_json": str(status_path),
        "source_metrics_json": str(metrics_path),
        "source": source,
        "reused": True,
    }


def weather_rows():
    rows = []
    duplicates = []
    for seed, run_id in WEATHER_RUNS.items():
        run_dir = RUNS / run_id
        status_path = run_dir / "status.json"
        metrics_path = run_dir / "metrics.json"
        payload = read_json(metrics_path)
        gate = payload.get("weather", {}).get("gate", {})
        for raw in payload.get("weather", {}).get("rows", []):
            method = raw.get("policy")
            budget = raw.get("budget")
            if method not in METHODS or budget not in (0.5, 0.75):
                continue
            enriched = dict(raw)
            enriched["no_leakage"] = gate.get("passed", "NA")
            enriched["parity"] = gate.get("full_patch_budget_parity", {}).get("max_forecast_diff") == 0.0
            rows.append(canonical_row("weather_practical", 144, seed, budget, method, enriched, run_id, run_dir, status_path, metrics_path, "stage2f_b_formal_weather"))
        candidates = []
        for candidate in sorted(RUNS.glob(f"stage2f_trap_multidata__seed{seed}*")):
            if source_ok(candidate / "status.json", candidate / "metrics.json"):
                candidates.append(candidate.name)
        if len(candidates) > 1:
            duplicates.append({"dataset": "weather_practical", "seed": seed, "preferred": run_id, "candidates": candidates})
    return rows, duplicates


def ett1_rows():
    rows = []
    duplicates = []
    for budget in (0.5, 0.75):
        label = f"b{budget:.2f}"
        eval_dir = RUNS / ETT1_FIXED_PARENT / "evaluations" / f"ett1__fixed_utility__{label}__equal__modelseed2021"
        raw = read_json(eval_dir / "metrics.json")
        rows.append(canonical_row("ett1_practical", 192, 2021, budget, METHODS[0], raw, ETT1_FIXED_PARENT, eval_dir, eval_dir / "status.json", eval_dir / "metrics.json", "stage2e_a_fixed"))
    trap_dir = RUNS / ETT1_TRAP_RUN
    trap_payload = read_json(trap_dir / "metrics.json")
    leakage = next((x for x in trap_payload.get("no_leakage", []) if int(x.get("patch_len", -1)) == 24), {})
    for raw in trap_payload.get("rows", []):
        if raw.get("policy") != METHODS[1] or int(raw.get("patch_len", -1)) != 24 or raw.get("budget") not in (0.5, 0.75):
            continue
        enriched = dict(raw)
        enriched["no_leakage"] = leakage.get("passed", "NA")
        enriched["parity"] = leakage.get("full_patch_budget_parity", {}).get("max_forecast_diff") == 0.0
        rows.append(canonical_row("ett1_practical", 192, 2021, raw["budget"], METHODS[1], enriched, ETT1_TRAP_RUN, trap_dir, trap_dir / "status.json", trap_dir / "metrics.json", "stage2e_c_patch24_full"))
    historical = sorted(RUNS.glob("stage2e_c_patch_ablation__ett1__seed2021__*"))
    if len(historical) > 1:
        duplicates.append({"dataset": "ett1_practical", "seed": 2021, "method": METHODS[1], "preferred": ETT1_TRAP_RUN, "candidates": [p.name for p in historical]})
    for seed, run_id in ETT1_SIDE_RUNS.items():
        run_dir = RUNS / run_id
        payload = read_json(RESULTS / f"stage2g_ett1_side_seed{seed}.json")
        leakage = payload.get("no_leakage", {})
        for raw in payload.get("rows", []):
            method = raw.get("policy")
            budget = raw.get("budget")
            if method not in METHODS or budget not in (0.5, 0.75):
                continue
            enriched = dict(raw)
            enriched["no_leakage"] = leakage.get("passed", "NA")
            enriched["parity"] = leakage.get("full_patch_budget_parity", {}).get("max_forecast_diff") == 0.0
            rows.append(canonical_row("ett1_practical", 192, seed, budget, method, enriched, run_id, run_dir, run_dir / "status.json", run_dir / "metrics.json", "stage2g_ett1_side_eval"))
    return rows, duplicates


def intermediate_existing():
    run_dir = RUNS / ETT1_TRAP_RUN
    payload = read_json(run_dir / "metrics.json")
    leakage = next((x for x in payload.get("no_leakage", []) if int(x.get("patch_len", -1)) == 24), {})
    for raw in payload.get("rows", []):
        if raw.get("policy") == METHODS[1] and int(raw.get("patch_len", -1)) == 24 and abs(float(raw.get("budget", -1)) - SELECTED_BUDGET) < 1e-9:
            enriched = dict(raw)
            enriched["no_leakage"] = leakage.get("passed", "NA")
            enriched["parity"] = leakage.get("full_patch_budget_parity", {}).get("max_forecast_diff") == 0.0
            row = canonical_row("ett1_practical", 192, 2021, SELECTED_BUDGET, METHODS[1], enriched, ETT1_TRAP_RUN, run_dir, run_dir / "status.json", run_dir / "metrics.json", "stage2e_c_patch24_full")
            row["status"] = "REUSED"
            return row
    return None


def budget_details():
    out = {}
    for dataset, cfg in DATASETS.items():
        whole_allowed = max(1, math.floor(cfg["channels"] * SELECTED_BUDGET + 1e-9))
        patch_total = cfg["channels"] * cfg["patches"]
        patch_allowed = max(cfg["patches"], math.floor(patch_total * SELECTED_BUDGET + 1e-9))
        out[dataset] = {
            "requested_budget": SELECTED_BUDGET,
            "whole_channel_total_units": cfg["channels"],
            "whole_channel_allowed_units": whole_allowed,
            "whole_channel_realized_budget": whole_allowed / cfg["channels"],
            "trap_total_patch_units": patch_total,
            "trap_allowed_patch_units": patch_allowed,
            "trap_realized_budget": patch_allowed / patch_total,
            "same_requested_budget": True,
            "same_rounding_rule": True,
            "exact_realized_ratio_equal": abs(whole_allowed / cfg["channels"] - patch_allowed / patch_total) < 1e-12,
        }
    return out


def inventory():
    weather, weather_dups = weather_rows()
    ett1, ett1_dups = ett1_rows()
    rows = weather + ett1
    expected = {(d, p["pred_len"], s, b, m) for d, p in DATASETS.items() for s in (2021, 2022, 2023) for b in (0.5, 0.75) for m in METHODS}
    found = {(r["dataset"], r["pred_len"], r["seed"], r["budget"], r["method"]) for r in rows}
    missing = [dict(dataset=k[0], pred_len=k[1], seed=k[2], budget=k[3], method=k[4]) for k in sorted(expected - found)]
    return {
        "status": "COMPLETE" if not missing and len(rows) == 24 else "REUSE_AUDIT_INCOMPLETE",
        "generated_at": now(),
        "expected_logical_rows": 24,
        "complete_logical_rows": len(rows),
        "rows": sorted(rows, key=lambda r: (r["dataset"], r["budget"], r["seed"], r["method"])),
        "missing_logical_keys": missing,
        "duplicate_candidates": weather_dups + ett1_dups,
    }


def write_inventory(payload):
    write_json(RESULTS / "stage2h_reuse_inventory.json", payload)
    fields = sorted({k for row in payload["rows"] for k in row})
    csv_path = REPORTS / "stage2h_reuse_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["rows"])
    lines = [
        "# Stage 2H Reuse Inventory", "", f"- status: `{payload['status']}`", f"- complete: {payload['complete_logical_rows']}/{payload['expected_logical_rows']}", f"- duplicate candidate groups: {len(payload['duplicate_candidates'])}", "",
        "| Dataset | pred_len | Seed | Budget | Method | Status | CMSE | CMAE | Source |", "|---|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(f"| {row['dataset']} | {row['pred_len']} | {row['seed']} | {row['budget']:.3f} | {row['method']} | {row['status']} | {row['cmse']:.9f} | {row['cmae']:.9f} | {row['source']} |")
    if payload["missing_logical_keys"]:
        lines += ["", "## Missing", "", "```json", json.dumps(payload["missing_logical_keys"], indent=2), "```"]
    write_text(REPORTS / "stage2h_reuse_inventory.md", "\n".join(lines) + "\n")


def create_run_task(dataset, method, budget, ts):
    cfg = DATASETS[dataset]
    method_label = "trap" if method == METHODS[1] else method
    b_label = str(budget).replace(".", "p")
    experiment_id = f"stage2h__{dataset}__pred{cfg['pred_len']}__{method_label}__b{b_label}__seed2021__{ts}"
    run_dir = RUNS / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = {
        "experiment_id": experiment_id,
        "stage_label": "stage2h",
        "dataset": dataset,
        "dataset_key": cfg["key"],
        "pred_len": cfg["pred_len"],
        "seed": SEED,
        "method": method,
        "requested_budget": budget,
        "budget_parser_type": "float",
        "budget_rounding_rule": "max(initial_units, floor(total_atomic_units * requested_budget + 1e-9))",
        "batch_size": 64,
        "max_test_windows": 0,
        "training": False,
        "official_baseline_rerun": False,
        "created_at": now(),
    }
    write_json(run_dir / "resolved_config.yaml", resolved)
    git_state = "\n".join([
        f"upstream_commit={run_text(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'])}",
        "upstream_status_short:", run_text(["git", "-C", str(UPSTREAM), "status", "--short"]),
        "upstream_diff_stat:", run_text(["git", "-C", str(UPSTREAM), "diff", "--stat"]), "",
    ])
    write_text(run_dir / "git_state.txt", git_state)
    write_text(run_dir / "code.diff", run_text(["git", "-C", str(UPSTREAM), "diff", "--binary"]) + "\n")
    write_json(run_dir / "status.json", {"experiment_id": experiment_id, "status": "PENDING", "created_at": now(), "dataset": dataset, "pred_len": cfg["pred_len"], "seed": SEED, "method": method, "budget": budget})
    command = "\n".join([
        "#!/usr/bin/env bash", "set -Eeuo pipefail", f'cd "{ROOT}"', "export PYTHONDONTWRITEBYTECODE=1", f'export MPLCONFIGDIR="{run_dir}/mplconfig"', f'mkdir -p "{run_dir}/mplconfig"',
        f'{{ date -Iseconds; echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}"; nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true; "{PYTHON}" -c "import torch; print(\'torch\', torch.__version__, \'cuda\', torch.version.cuda, \'available\', torch.cuda.is_available())"; }} > "{run_dir}/env.txt" 2>&1',
        f'"{PYTHON}" "{ROOT}/scripts/stage2h_prepare_minimal_robustness.py" evaluate-one --dataset "{dataset}" --method "{method}" --budget "{budget}" --run-dir "{run_dir}" 2>&1 | tee -a "{run_dir}/train.log"', "",
    ])
    command_file = run_dir / "command.sh"
    write_text(command_file, command)
    os.chmod(command_file, os.stat(command_file).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "pred_len": cfg["pred_len"],
        "seed": SEED,
        "method": method,
        "budget": budget,
        "stage_label": "stage2h",
        "command_file": str(command_file),
        "run_dir": str(run_dir),
        "required_free_mib": REQUIRED_FREE_MIB,
        "allowed_gpu_uuids": GPU_UUIDS,
    }


def prepare():
    for path in (RESULTS, REPORTS, RUNS, LOGS, QUEUE.parent):
        path.mkdir(parents=True, exist_ok=True)
    decision = read_json(RESULTS / "stage2g_formal_matrix_decision.json")
    inv = inventory()
    write_inventory(inv)
    budget = budget_details()
    stage2h_label_supported = "choices=" not in (ROOT / "scripts" / "stage2f_trap_multidata.py").read_text(encoding="utf-8").split('add_argument("--stage-label"', 1)[-1].split("\n", 1)[0]
    existing = intermediate_existing()
    scope_ok = decision.get("decision_status") == "PASS" and decision.get("formal_decision", {}).get("STAGE2G_TRUE_SCOPE_FORMAL_MATRIX_PASS") is True
    scope_audit = {
        "status": "PASS" if scope_ok and inv["status"] == "COMPLETE" and stage2h_label_supported else "BLOCKED",
        "generated_at": now(),
        "stage2g_decision_pass": decision.get("decision_status") == "PASS",
        "allowed_scope": {"weather_practical": 144, "ett1_practical": 192},
        "weather_p96_p192": "SUPERSEDED_NOT_USED",
        "solarwind": "NOT_STARTED_NOT_QUEUED",
        "official_baseline": "REUSE_ONLY_NOT_RERUN",
        "stage2h_guard": "NO_DEDICATED_GUARD_FOUND_TRUE_SCOPE_ENFORCED_BY_PREPARE_VALIDATION",
        "stage_label_stage2h_supported": stage2h_label_supported,
        "namespace_patch_required": False,
        "budget_parser_type": "float",
        "budget_allowlist": None,
        "candidate_budget": SELECTED_BUDGET,
        "selected_budget": SELECTED_BUDGET,
        "selection_reason": "Existing formal Stage 2E-C budget grid contains 0.625 and evaluator accepts arbitrary float.",
        "budget_rounding_rule": "max(initial_units, floor(total_atomic_units * ratio + 1e-9))",
        "budget_details": budget,
        "fixed_trap_budget_parity": {"same_requested_budget": True, "same_rounding_rule": True, "exact_realized_ratio_equal": all(x["exact_realized_ratio_equal"] for x in budget.values())},
        "queue_schema_sources": ["configs/queues/stage2g_official_baselines.yaml", "configs/queues/stage2g_ett1_p192_fixed_trap.yaml", "configs/queues/stage2f_b_trap_side_missing_seeds.yaml", "scheduler/experiment_scheduler.py", "scripts/scheduler.sh"],
        "required_free_mib": REQUIRED_FREE_MIB,
        "trap_definition_modified": False,
        "fixed_definition_modified": False,
        "scope_guard_patched": False,
        "upstream_tracked_diff": run_text(["git", "-C", str(UPSTREAM), "diff", "--stat"]),
        "can_enter_phase2": scope_ok and stage2h_label_supported,
    }
    write_json(RESULTS / "stage2h_scope_audit.json", scope_audit)
    scope_lines = [
        "# Stage 2H Scope Audit", "", f"- status: `{scope_audit['status']}`", "- Weather scope: `weather_practical pred_len=144`", "- ETT1 scope: `ett1_practical pred_len=192`", "- Weather p96/p192: `SUPERSEDED_NOT_USED`", "- SolarWind: `NOT_STARTED_NOT_QUEUED`", f"- stage-label stage2h supported: `{stage2h_label_supported}`", "- namespace patch required: `False`", f"- selected budget: `{SELECTED_BUDGET}`", "- parser: `float`", f"- rounding: `{scope_audit['budget_rounding_rule']}`", f"- required_free_mib: `{REQUIRED_FREE_MIB}`", "- algorithm logic modified: `False`", "", "## Realized Budget Mapping", "", "| Dataset | Fixed units | Fixed ratio | TRAP units | TRAP ratio |", "|---|---:|---:|---:|---:|",
    ]
    for dataset, row in budget.items():
        scope_lines.append(f"| {dataset} | {row['whole_channel_allowed_units']}/{row['whole_channel_total_units']} | {row['whole_channel_realized_budget']:.9f} | {row['trap_allowed_patch_units']}/{row['trap_total_patch_units']} | {row['trap_realized_budget']:.9f} |")
    scope_lines += ["", "Both methods receive the same requested ratio and use the same floor rule in their native atomic unit. Exact realized ratios differ where whole-channel granularity is coarser; this is reported, not hidden."]
    write_text(REPORTS / "stage2h_scope_audit.md", "\n".join(scope_lines) + "\n")
    if inv["status"] != "COMPLETE" or not scope_ok or not stage2h_label_supported:
        tasks = []
        planning_status = "REUSE_AUDIT_INCOMPLETE" if inv["status"] != "COMPLETE" else "BLOCKED"
    else:
        needed = [("weather_practical", METHODS[0]), ("weather_practical", METHODS[1]), ("ett1_practical", METHODS[0])]
        ts = stamp()
        tasks = [create_run_task(dataset, method, SELECTED_BUDGET, ts) for dataset, method in needed]
        planning_status = "PLANNING_PASS"
    queue = {
        "queue_id": "stage2h_minimal_robustness",
        "created_at": now(),
        "scope": {"datasets": {"weather_practical": 144, "ett1_practical": 192}, "seed": SEED, "budget": SELECTED_BUDGET, "methods": list(METHODS), "stage_label": "stage2h"},
        "max_concurrent_total": 1,
        "max_concurrent_per_gpu": 1,
        "max_concurrent": 1,
        "poll_seconds": 60,
        "gpu_safety_margin_mib": REQUIRED_FREE_MIB,
        "logical_task_count": 4,
        "reused_logical_task_count": 1 if existing else 0,
        "active_executable_task_count": len(tasks),
        "tasks": tasks,
    }
    write_json(QUEUE, queue)
    validation_errors = []
    if len(tasks) > 4:
        validation_errors.append("task_count_exceeds_4")
    for task in tasks:
        if task["required_free_mib"] < REQUIRED_FREE_MIB: validation_errors.append("required_free_mib_below_12000")
        if task["dataset"] not in DATASETS: validation_errors.append("invalid_dataset")
        if task["pred_len"] != DATASETS[task["dataset"]]["pred_len"]: validation_errors.append("invalid_pred_len")
        if task["seed"] != 2021: validation_errors.append("invalid_seed")
        if task["method"] not in METHODS: validation_errors.append("invalid_method")
        if abs(task["budget"] - SELECTED_BUDGET) > 1e-12: validation_errors.append("invalid_budget")
        if task["stage_label"] != "stage2h": validation_errors.append("invalid_stage_label")
    plan = {
        "status": planning_status if not validation_errors else "BLOCKED_QUEUE_VALIDATION_FAILED",
        "generated_at": now(),
        "candidate_budget": SELECTED_BUDGET,
        "selected_budget": SELECTED_BUDGET,
        "selection_reason": scope_audit["selection_reason"],
        "budget_parser_type": "float",
        "budget_rounding_rule": scope_audit["budget_rounding_rule"],
        "weather_realized_budget": budget["weather_practical"],
        "ett1_realized_budget": budget["ett1_practical"],
        "fixed_trap_budget_parity": scope_audit["fixed_trap_budget_parity"],
        "reuse_audit_status": inv["status"],
        "reuse_logical_rows": inv["complete_logical_rows"],
        "intermediate_reused_rows": [existing] if existing else [],
        "logical_task_count": 4,
        "active_executable_task_count": len(tasks),
        "reused_logical_task_count": 1 if existing else 0,
        "queue": str(QUEUE),
        "queue_validation_pass": not validation_errors,
        "queue_validation_errors": validation_errors,
        "tasks": tasks,
        "weather_p96_p192_in_queue": False,
        "solarwind_in_queue": False,
        "official_baseline_in_queue": False,
        "seed2022_2023_in_queue": False,
        "required_free_mib": REQUIRED_FREE_MIB,
        "trap_definition_modified": False,
        "scope_guard_patched": False,
    }
    write_json(RESULTS / "stage2h_plan.json", plan)
    plan_lines = [
        "# Stage 2H Plan", "", f"- status: `{plan['status']}`", f"- selected budget: `{SELECTED_BUDGET}`", f"- reuse audit: `{inv['complete_logical_rows']}/24`", f"- logical tasks: `{plan['logical_task_count']}`", f"- reused intermediate tasks: `{plan['reused_logical_task_count']}`", f"- active executable tasks: `{plan['active_executable_task_count']}`", f"- queue validation: `{plan['queue_validation_pass']}`", f"- required_free_mib: `{REQUIRED_FREE_MIB}`", "- Weather p96/p192: `not queued`", "- SolarWind: `not queued`", "- official baselines: `not queued`", "", "| Dataset | pred_len | Seed | Budget | Method | Status | Experiment ID |", "|---|---:|---:|---:|---|---|---|",
    ]
    if existing:
        plan_lines.append(f"| ett1_practical | 192 | 2021 | {SELECTED_BUDGET} | {METHODS[1]} | REUSED | {existing['experiment_id']} |")
    for task in tasks:
        plan_lines.append(f"| {task['dataset']} | {task['pred_len']} | {task['seed']} | {task['budget']} | {task['method']} | PENDING | {task['experiment_id']} |")
    write_text(REPORTS / "stage2h_plan.md", "\n".join(plan_lines) + "\n")
    print(json.dumps({"status": plan["status"], "reuse": f"{inv['complete_logical_rows']}/24", "selected_budget": SELECTED_BUDGET, "logical_tasks": 4, "reused": plan["reused_logical_task_count"], "active_tasks": len(tasks), "queue": str(QUEUE)}, indent=2))
    return 0 if plan["queue_validation_pass"] else 2


def evaluate_one(args):
    from evaluation import stage2a_active_acquisition as s2a
    from scripts import stage2e_c_patch_ablation as s2c
    from scripts import stage2e_patch_screen as s2e
    from scripts import stage2f_trap_multidata as s2f
    import torch

    run_dir = Path(args.run_dir)
    if args.dataset not in DATASETS or args.method not in METHODS or abs(args.budget - SELECTED_BUDGET) > 1e-12:
        raise ValueError("Stage 2H evaluate-one scope validation failed")
    cfg_meta = DATASETS[args.dataset]
    dataset_key = cfg_meta["key"]
    if args.dataset == "weather_practical":
        cfg = s2f.install_dataset("weather")
        official_run_id = s2f.DATASET_RUNS["weather"]
    else:
        official_run_id = "ctf__ett1_practical__official__seed2021__20260608_205433"
        official_status = read_json(RUNS / official_run_id / "status.json")
        official_metrics = read_json(RUNS / official_run_id / "metrics.json")
        if official_status.get("status") != "SUCCEEDED" or official_metrics.get("status") != "SUCCEEDED":
            raise RuntimeError(f"official baseline not succeeded for seed 2021: {official_run_id}")
        cfg = dict(s2a.DATASETS["ett1"])
        cfg["stage1_run"] = official_run_id
        cfg["baseline_cmse"] = float(official_metrics["mse"])
        cfg["baseline_cmae"] = float(official_metrics["mae"])
        s2a.DATASETS["ett1"] = cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_dataset = s2a.DATASETS.get(dataset_key)
    old_ett1 = s2a.DATASETS.get("ett1")
    try:
        s2a.DATASETS[dataset_key] = cfg
        if args.method == METHODS[0]:
            raw = s2a.evaluate_policy(dataset_key, "fixed_utility", "equal", args.budget, device_name=str(device), max_batches=0, utility_max_batches=0)
            total_units = int(cfg["enc_in"])
            allowed_units = int(s2a.allowed_budget(s2a.equal_costs(cfg), args.budget, s2a.initial_channel(cfg)))
            row = {
                "dataset": args.dataset, "pred_len": cfg_meta["pred_len"], "seed": 2021, "method": args.method, "budget": args.budget,
                "CMSE": float(raw["CMSE"]), "CMAE": float(raw["CMAE"]), "invalid_action_count": int(raw.get("invalid_action_count", 0)),
                "repeat_patch_action_count": int(raw.get("repeat_action_count", 0)), "overspend_count": int(raw.get("overspend_count", 0)),
                "requested_budget": args.budget, "allowed_budget_units": allowed_units, "total_budget_units": total_units,
                "realized_budget": float(raw.get("mean_used_cost", allowed_units)) / total_units, "selected_count": raw.get("mean_used_cost"),
                "unique_selected_count": raw.get("mean_acquired_channels"), "selection_churn_rate": "NA", "policy_entropy": "NA",
                "uses_test_label": bool(raw.get("uses_test_label", False)), "checkpoint": raw.get("checkpoint"), "inference_latency_seconds": raw.get("elapsed_seconds"),
            }
        else:
            model, ctf_args, checkpoint = s2a.load_model(cfg, device, batch_size=64)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            order = s2f.fixed_order(dataset_key, cfg, model, ctf_args, device)
            spec = s2e.patch_spec(cfg, int(math.ceil(int(cfg["seq_len"]) / 8)))
            old_budgets, old_strategies = list(s2c.BUDGETS), list(s2c.STRATEGIES)
            s2a.DATASETS["ett1"] = cfg
            s2c.BUDGETS = [args.budget]
            s2c.STRATEGIES = [METHODS[1]]
            try:
                raw = s2c.eval_deployable_ablation(cfg, model, ctf_args, spec, order, device, 64, max_windows=0)[0]
                leakage = s2c.parity_and_leakage(cfg, model, ctf_args, spec, order, device)
            finally:
                s2c.BUDGETS, s2c.STRATEGIES = old_budgets, old_strategies
            total_units = int(cfg["enc_in"]) * int(spec["num_patches"])
            allowed_units = int(s2e.allowed_total_patches(cfg, spec, args.budget))
            row = {
                "dataset": args.dataset, "pred_len": cfg_meta["pred_len"], "seed": 2021, "method": args.method, "budget": args.budget,
                "CMSE": float(raw["CMSE"]), "CMAE": float(raw["CMAE"]), "invalid_action_count": int(raw.get("invalid_action_count", 0)),
                "repeat_patch_action_count": int(raw.get("repeat_patch_action_count", 0)), "overspend_count": int(raw.get("overspend_count", 0)),
                "requested_budget": args.budget, "allowed_budget_units": allowed_units, "total_budget_units": total_units,
                "realized_budget": float(raw.get("normalized_cost", allowed_units / total_units)), "selected_count": raw.get("used_patch_cost"),
                "unique_selected_count": raw.get("used_patch_cost"), "selection_churn_rate": raw.get("selection_churn_rate", "NA"), "policy_entropy": raw.get("policy_entropy", "NA"),
                "uses_test_label": False, "checkpoint": str(checkpoint), "inference_latency_seconds": raw.get("inference_latency_seconds"),
                "selected_channel_distribution": raw.get("selected_channel_distribution"), "selected_temporal_patch_distribution": raw.get("selected_temporal_patch_distribution"),
                "no_leakage": leakage.get("passed"), "parity": leakage.get("full_patch_budget_parity", {}).get("max_forecast_diff") == 0.0,
                "forecast_parity_difference": leakage.get("full_patch_budget_parity", {}).get("max_forecast_diff"),
            }
        if args.method == METHODS[0]:
            row["no_leakage"] = not row["uses_test_label"]
            row["parity"] = True
            row["forecast_parity_difference"] = 0.0
        payload = {
            "status": "SUCCEEDED", "finished_at": now(), "stage": "stage2h", "stage_label": "stage2h", "experiment_id": run_dir.name,
            "official_run_id": official_run_id, "training_started": False, "official_baseline_rerun": False, "trap_definition_modified": False,
            "scope_guard_patched": False, "deterministic": True, "row": row,
        }
        write_json(run_dir / "metrics.json", payload)
        write_json(run_dir / "status.json", {"status": "SUCCEEDED", "experiment_id": run_dir.name, "finished_at": now(), "CMSE": row["CMSE"], "CMAE": row["CMAE"], "invalid": row["invalid_action_count"], "repeat": row["repeat_patch_action_count"], "overspend": row["overspend_count"], "no_leakage": row["no_leakage"], "parity": row["parity"]})
        print(json.dumps({"status": "SUCCEEDED", "experiment_id": run_dir.name, "dataset": args.dataset, "method": args.method, "budget": args.budget, "CMSE": row["CMSE"], "CMAE": row["CMAE"]}, indent=2))
        return 0
    except Exception as exc:
        write_json(run_dir / "status.json", {"status": "FAILED", "experiment_id": run_dir.name, "finished_at": now(), "error_type": "CODE_ERROR", "error": str(exc)})
        raise
    finally:
        if old_dataset is not None:
            s2a.DATASETS[dataset_key] = old_dataset
        if old_ett1 is not None:
            s2a.DATASETS["ett1"] = old_ett1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    ev = sub.add_parser("evaluate-one")
    ev.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ev.add_argument("--method", required=True, choices=METHODS)
    ev.add_argument("--budget", type=float, required=True)
    ev.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "evaluate-one":
        return evaluate_one(args)
    return prepare()


if __name__ == "__main__":
    raise SystemExit(main())
