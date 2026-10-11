#!/usr/bin/env python
"""Summarize Stage 2H reuse and interpolation-budget evidence."""

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/data/zza/code/BudgetedActiveChannelTokenFormer")
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
METHODS = ("whole_channel_fixed_utility", "temporal_recent_all_channels")
SELECTED_BUDGET = 0.625


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def stat(values):
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_existing(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], float(row["budget"]))].append(row)
    summaries = []
    seed_rows = []
    for (dataset, budget), group in sorted(grouped.items()):
        by = {(int(r["seed"]), r["method"]): r for r in group}
        fixed = [by[(seed, METHODS[0])] for seed in (2021, 2022, 2023) if (seed, METHODS[0]) in by]
        trap = [by[(seed, METHODS[1])] for seed in (2021, 2022, 2023) if (seed, METHODS[1]) in by]
        paired = []
        for seed in (2021, 2022, 2023):
            if (seed, METHODS[0]) not in by or (seed, METHODS[1]) not in by:
                continue
            f, t = by[(seed, METHODS[0])], by[(seed, METHODS[1])]
            item = {
                "dataset": dataset, "pred_len": f["pred_len"], "budget": budget, "seed": seed,
                "fixed_cmse": f["cmse"], "trap_cmse": t["cmse"], "delta_cmse": t["cmse"] - f["cmse"],
                "fixed_cmae": f["cmae"], "trap_cmae": t["cmae"], "delta_cmae": t["cmae"] - f["cmae"],
                "cmse_win": t["cmse"] < f["cmse"], "cmae_win": t["cmae"] < f["cmae"],
            }
            paired.append(item)
            seed_rows.append(item)
        if len(fixed) != 3 or len(trap) != 3:
            summaries.append({"dataset": dataset, "budget": budget, "status": "INCOMPLETE", "paired_seeds": len(paired)})
            continue
        fixed_cmse, trap_cmse = stat([r["cmse"] for r in fixed]), stat([r["cmse"] for r in trap])
        fixed_cmae, trap_cmae = stat([r["cmae"] for r in fixed]), stat([r["cmae"] for r in trap])
        dcmse, dcmae = [r["delta_cmse"] for r in paired], [r["delta_cmae"] for r in paired]
        summaries.append({
            "dataset": dataset, "pred_len": fixed[0]["pred_len"], "budget": budget, "status": "COMPLETE", "paired_seeds": len(paired),
            "fixed_cmse": fixed_cmse, "trap_cmse": trap_cmse, "fixed_cmae": fixed_cmae, "trap_cmae": trap_cmae,
            "paired_delta_cmse_mean": statistics.fmean(dcmse), "paired_delta_cmse_std": statistics.pstdev(dcmse),
            "paired_delta_cmae_mean": statistics.fmean(dcmae), "paired_delta_cmae_std": statistics.pstdev(dcmae),
            "cmse_seed_win_count": sum(x < 0 for x in dcmse), "cmae_seed_win_count": sum(x < 0 for x in dcmae),
            "worst_seed_delta_cmse": max(dcmse), "worst_seed_delta_cmae": max(dcmae),
            "relative_cmse_improvement": (fixed_cmse["mean"] - trap_cmse["mean"]) / fixed_cmse["mean"],
            "relative_cmae_improvement": (fixed_cmae["mean"] - trap_cmae["mean"]) / fixed_cmae["mean"],
            "all_seeds_complete": len(paired) == 3,
            "direction_reversal_cmse": any(x >= 0 for x in dcmse), "direction_reversal_cmae": any(x >= 0 for x in dcmae),
            "outlier_assessment": "NO_FORMAL_OUTLIER_TEST_N3",
        })
    return summaries, seed_rows


def minimal_rows(plan):
    rows = []
    for reused in plan.get("intermediate_reused_rows", []):
        row = dict(reused)
        row["status"] = "REUSED"
        rows.append(row)
    for task in plan.get("tasks", []):
        status = read_json(Path(task["run_dir"]) / "status.json")
        metrics = read_json(Path(task["run_dir"]) / "metrics.json")
        raw = metrics.get("row", {})
        rows.append({
            "dataset": task["dataset"], "pred_len": task["pred_len"], "seed": task["seed"], "budget": task["budget"], "method": task["method"],
            "experiment_id": task["experiment_id"], "run_dir": task["run_dir"], "status": status.get("status", "PENDING"),
            "cmse": raw.get("CMSE"), "cmae": raw.get("CMAE"), "invalid": raw.get("invalid_action_count", status.get("invalid", "NA")),
            "repeat": raw.get("repeat_patch_action_count", status.get("repeat", "NA")), "overspend": raw.get("overspend_count", status.get("overspend", "NA")),
            "no_leakage": raw.get("no_leakage", status.get("no_leakage", "NA")), "parity": raw.get("parity", status.get("parity", "NA")),
            "requested_budget": raw.get("requested_budget", task["budget"]), "realized_budget": raw.get("realized_budget", "NA"),
            "selected_count": raw.get("selected_count", "NA"), "unique_selected_count": raw.get("unique_selected_count", "NA"),
            "selection_churn_rate": raw.get("selection_churn_rate", "NA"), "policy_entropy": raw.get("policy_entropy", "NA"),
            "source_status_json": str(Path(task["run_dir"]) / "status.json"), "source_metrics_json": str(Path(task["run_dir"]) / "metrics.json"), "reused": False,
        })
    return sorted(rows, key=lambda r: (r["dataset"], r["method"]))


def budget_monotonicity(summaries):
    result = []
    for dataset in sorted({r["dataset"] for r in summaries}):
        for method, prefix in ((METHODS[0], "fixed"), (METHODS[1], "trap")):
            by = {r["budget"]: r for r in summaries if r["dataset"] == dataset and r.get("status") == "COMPLETE"}
            if 0.5 not in by or 0.75 not in by:
                result.append({"dataset": dataset, "method": method, "status": "NA"})
                continue
            cmse_ok = by[0.75][f"{prefix}_cmse"]["mean"] <= by[0.5][f"{prefix}_cmse"]["mean"]
            cmae_ok = by[0.75][f"{prefix}_cmae"]["mean"] <= by[0.5][f"{prefix}_cmae"]["mean"]
            result.append({"dataset": dataset, "method": method, "cmse_monotonic": cmse_ok, "cmae_monotonic": cmae_ok, "status": "PASS" if cmse_ok and cmae_ok else "NON_MONOTONIC_BUDGET_RESPONSE"})
    return result


def main():
    inventory = read_json(RESULTS / "stage2h_reuse_inventory.json")
    plan = read_json(RESULTS / "stage2h_plan.json")
    summaries, seeds = summarize_existing(inventory.get("rows", []))
    minimal = minimal_rows(plan)
    complete_minimal = sum(r["status"] in ("SUCCEEDED", "REUSED") for r in minimal)
    waiting = sum(r["status"] == "WAITING_RESOURCE" for r in minimal)
    running = sum(r["status"] == "RUNNING" for r in minimal)
    failed = sum(r["status"] == "FAILED" for r in minimal)
    hygiene_rows = [r for r in inventory.get("rows", []) + minimal if r.get("status") in ("SUCCEEDED", "REUSED")]
    invalid = sum(int(r["invalid"]) for r in hygiene_rows if r.get("invalid") not in (None, "", "NA"))
    repeat = sum(int(r["repeat"]) for r in hygiene_rows if r.get("repeat") not in (None, "", "NA"))
    overspend = sum(int(r["overspend"]) for r in hygiene_rows if r.get("overspend") not in (None, "", "NA"))
    no_leakage_values = [r.get("no_leakage") for r in hygiene_rows if r.get("no_leakage") not in (None, "", "NA")]
    parity_values = [r.get("parity") for r in hygiene_rows if r.get("parity") not in (None, "", "NA")]
    monotonicity = budget_monotonicity(summaries)
    if inventory.get("complete_logical_rows") != 24:
        status = "REUSE_AUDIT_INCOMPLETE"
    elif failed:
        status = "FAILED"
    elif complete_minimal == 4:
        status = "READY_FOR_FORMAL_DECISION"
    elif waiting and not running:
        status = "WAITING_RESOURCE"
    elif running:
        status = "RUNNING"
    else:
        status = "PLANNING_PASS"
    payload = {
        "status": status, "generated_at": now(), "reuse_audit_status": inventory.get("status"),
        "reuse_logical_rows_complete": inventory.get("complete_logical_rows", 0), "reuse_logical_rows_expected": 24,
        "existing_robustness": summaries, "seed_rows": seeds, "budget_monotonicity": monotonicity,
        "minimal_matrix": minimal, "minimal_logical_tasks": 4, "minimal_complete": complete_minimal, "minimal_running": running,
        "minimal_waiting_resource": waiting, "minimal_failed": failed,
        "policy_gates": {"no_leakage": all(v is True for v in no_leakage_values) if no_leakage_values else "NA", "parity": all(v is True for v in parity_values) if parity_values else "NA", "invalid": invalid, "repeat": repeat, "overspend": overspend, "budget_parity": plan.get("fixed_trap_budget_parity"), "missing_diagnostics": ["selection_churn_rate", "policy_entropy"]},
        "queue": read_json(ROOT / "configs" / "queues" / "stage2h_minimal_robustness.yaml"),
    }
    write_json(RESULTS / "stage2h_robustness_status.json", payload)
    seed_fields = list(seeds[0]) if seeds else ["dataset", "pred_len", "budget", "seed"]
    with (REPORTS / "stage2h_seed_robustness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fields)
        writer.writeheader(); writer.writerows(seeds)
    seed_md = ["# Stage 2H Seed Robustness", "", "| Dataset | Budget | Seed | Fixed CMSE | TRAP CMSE | Delta CMSE | Fixed CMAE | TRAP CMAE | Delta CMAE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in seeds:
        seed_md.append(f"| {row['dataset']} | {row['budget']:.2f} | {row['seed']} | {row['fixed_cmse']:.9f} | {row['trap_cmse']:.9f} | {row['delta_cmse']:.9f} | {row['fixed_cmae']:.9f} | {row['trap_cmae']:.9f} | {row['delta_cmae']:.9f} |")
    write_text(REPORTS / "stage2h_seed_robustness.md", "\n".join(seed_md) + "\n")
    result_rows = []
    for row in summaries:
        result_rows.append({"row_type": "existing_summary", "dataset": row["dataset"], "pred_len": row.get("pred_len"), "budget": row["budget"], "method": "paired_fixed_vs_trap", "status": row["status"], "fixed_cmse_mean": row.get("fixed_cmse", {}).get("mean"), "trap_cmse_mean": row.get("trap_cmse", {}).get("mean"), "mean_delta_cmse": row.get("paired_delta_cmse_mean"), "cmse_wins": row.get("cmse_seed_win_count"), "worst_delta_cmse": row.get("worst_seed_delta_cmse"), "fixed_cmae_mean": row.get("fixed_cmae", {}).get("mean"), "trap_cmae_mean": row.get("trap_cmae", {}).get("mean"), "mean_delta_cmae": row.get("paired_delta_cmae_mean"), "cmae_wins": row.get("cmae_seed_win_count"), "worst_delta_cmae": row.get("worst_seed_delta_cmae")})
    for row in minimal:
        result_rows.append({"row_type": "minimal_intermediate", "dataset": row["dataset"], "pred_len": row["pred_len"], "seed": row["seed"], "budget": row["budget"], "method": row["method"], "status": row["status"], "experiment_id": row["experiment_id"], "cmse": row.get("cmse"), "cmae": row.get("cmae"), "realized_budget": row.get("realized_budget")})
    fields = sorted({k for row in result_rows for k in row})
    with (REPORTS / "stage2h_robustness_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(result_rows)
    lines = ["# Stage 2H Robustness Results", "", f"- status: `{status}`", f"- reuse audit: `{inventory.get('complete_logical_rows', 0)}/24`", f"- minimal complete: `{complete_minimal}/4`", "", "## Existing Robustness", "", "| Dataset | Budget | Fixed CMSE mean/std | TRAP CMSE mean/std | Mean delta | Wins | Worst delta |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row.get("status") != "COMPLETE": continue
        lines.append(f"| {row['dataset']} | {row['budget']:.2f} | {row['fixed_cmse']['mean']:.9f}/{row['fixed_cmse']['std']:.9f} | {row['trap_cmse']['mean']:.9f}/{row['trap_cmse']['std']:.9f} | {row['paired_delta_cmse_mean']:.9f} | {row['cmse_seed_win_count']}/3 | {row['worst_seed_delta_cmse']:.9f} |")
    lines += ["", "## Minimal Intermediate Budget", "", "| Dataset | pred_len | Seed | Budget | Method | Status | CMSE | CMAE | Realized budget |", "|---|---:|---:|---:|---|---|---:|---:|---:|"]
    for row in minimal:
        lines.append(f"| {row['dataset']} | {row['pred_len']} | {row['seed']} | {row['budget']} | {row['method']} | {row['status']} | {row.get('cmse','')} | {row.get('cmae','')} | {row.get('realized_budget','')} |")
    lines += ["", "Missing policy diagnostics remain `NA`: selection churn and entropy were not emitted by the existing evaluator; TRAP logic was not modified to manufacture them."]
    write_text(REPORTS / "stage2h_robustness_results.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": status, "reuse": f"{inventory.get('complete_logical_rows', 0)}/24", "minimal_complete": f"{complete_minimal}/4", "running": running, "waiting_resource": waiting, "failed": failed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
