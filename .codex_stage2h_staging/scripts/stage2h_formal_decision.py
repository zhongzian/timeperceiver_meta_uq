#!/usr/bin/env python
"""Generate the Stage 2H formal robustness decision from persisted evidence."""

import csv
import json
import time
from pathlib import Path

ROOT = Path("/mnt/data/zza/code/BudgetedActiveChannelTokenFormer")
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
METHODS = ("whole_channel_fixed_utility", "temporal_recent_all_channels")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path, payload):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); tmp.replace(path)


def write_text(path, text):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8"); tmp.replace(path)


def main():
    status = read_json(RESULTS / "stage2h_robustness_status.json")
    plan = read_json(RESULTS / "stage2h_plan.json")
    scope = read_json(RESULTS / "stage2h_scope_audit.json")
    existing = status.get("existing_robustness", [])
    minimal = status.get("minimal_matrix", [])
    existing_primary = all(r.get("status") == "COMPLETE" and r.get("trap_cmse", {}).get("mean", 1e99) < r.get("fixed_cmse", {}).get("mean", -1e99) and r.get("cmse_seed_win_count", 0) >= 2 for r in existing) and len(existing) == 4
    minimal_by = {(r.get("dataset"), r.get("method")): r for r in minimal if r.get("status") in ("SUCCEEDED", "REUSED")}
    minimal_complete = len(minimal_by) == 4
    intermediate_primary = minimal_complete and all(minimal_by[(dataset, METHODS[1])].get("cmse", 1e99) <= minimal_by[(dataset, METHODS[0])].get("cmse", -1e99) for dataset in ("weather_practical", "ett1_practical"))
    gates = status.get("policy_gates", {})
    hygiene = gates.get("no_leakage") is True and gates.get("parity") is True and gates.get("invalid") == 0 and gates.get("repeat") == 0 and gates.get("overspend") == 0
    scope_clean = scope.get("trap_definition_modified") is False and scope.get("scope_guard_patched") is False and plan.get("weather_p96_p192_in_queue") is False and plan.get("solarwind_in_queue") is False and plan.get("official_baseline_in_queue") is False
    reuse_complete = status.get("reuse_logical_rows_complete") == 24
    if not reuse_complete:
        decision = "REUSE_AUDIT_INCOMPLETE"
    elif status.get("minimal_failed", 0):
        decision = "FAILED"
    elif not minimal_complete:
        decision = status.get("status", "RUNNING")
    elif existing_primary and intermediate_primary and hygiene and scope_clean:
        decision = "STAGE2H_ROBUSTNESS_PASS"
    else:
        decision = "STAGE2H_ROBUSTNESS_MIXED"
    payload = {
        "decision_status": decision, "generated_at": now(),
        "STAGE2H_ROBUSTNESS_PASS": decision == "STAGE2H_ROBUSTNESS_PASS",
        "ENTER_NEXT_STAGE": decision == "STAGE2H_ROBUSTNESS_PASS",
        "gates": {
            "reuse_audit_24_of_24": reuse_complete, "existing_primary_cmse_gate": existing_primary,
            "minimal_matrix_4_of_4": minimal_complete, "intermediate_primary_cmse_gate": intermediate_primary,
            "no_leakage": gates.get("no_leakage"), "parity": gates.get("parity"), "invalid": gates.get("invalid"),
            "repeat": gates.get("repeat"), "overspend": gates.get("overspend"), "scope_clean": scope_clean,
            "same_requested_budget": plan.get("fixed_trap_budget_parity", {}).get("same_requested_budget"),
            "same_rounding_rule": plan.get("fixed_trap_budget_parity", {}).get("same_rounding_rule"),
            "exact_cross_granularity_realized_ratio_equal": plan.get("fixed_trap_budget_parity", {}).get("exact_realized_ratio_equal"),
        },
        "existing_robustness": existing, "minimal_matrix": minimal,
        "diagnostic_notes": ["No p-value or formal confidence interval is claimed for n=3.", "Exact realized ratios can differ because fixed uses whole-channel units while TRAP uses patch units; requested ratio and floor semantics are identical.", "Budget monotonicity is diagnostic, not a hard gate.", "Missing churn/entropy fields remain NA; core policy logic was not modified."],
        "recommended_next": "Enter a tightly scoped ablation/extension decision only after PASS." if decision == "STAGE2H_ROBUSTNESS_PASS" else "Complete or diagnose the Stage 2H minimal matrix before expanding scope.",
    }
    write_json(RESULTS / "stage2h_formal_decision.json", payload)
    csv_rows = []
    for row in existing:
        csv_rows.append({"section": "existing", "dataset": row.get("dataset"), "budget": row.get("budget"), "status": row.get("status"), "fixed_cmse_mean": row.get("fixed_cmse", {}).get("mean"), "trap_cmse_mean": row.get("trap_cmse", {}).get("mean"), "delta_cmse": row.get("paired_delta_cmse_mean"), "cmse_wins": row.get("cmse_seed_win_count")})
    for row in minimal:
        csv_rows.append({"section": "intermediate", "dataset": row.get("dataset"), "budget": row.get("budget"), "method": row.get("method"), "status": row.get("status"), "cmse": row.get("cmse"), "cmae": row.get("cmae"), "experiment_id": row.get("experiment_id")})
    fields = sorted({k for row in csv_rows for k in row})
    with (REPORTS / "stage2h_formal_decision.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(csv_rows)
    lines = ["# Stage 2H Formal Robustness Decision", "", f"## Decision: `{decision}`", "", f"- `STAGE2H_ROBUSTNESS_PASS={payload['STAGE2H_ROBUSTNESS_PASS']}`", f"- `ENTER_NEXT_STAGE={payload['ENTER_NEXT_STAGE']}`", "", "## Gates", ""]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Interpretation", "", "This decision uses paired three-seed summaries and the seed2021 intermediate-budget check. It does not claim statistical significance.", "", "Requested budget and rounding semantics match across methods. Realized ratios are reported separately because the acquisition units differ."]
    write_text(REPORTS / "stage2h_formal_decision.md", "\n".join(lines) + "\n")
    print(json.dumps({"decision_status": decision, "STAGE2H_ROBUSTNESS_PASS": payload["STAGE2H_ROBUSTNESS_PASS"], "ENTER_NEXT_STAGE": payload["ENTER_NEXT_STAGE"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
