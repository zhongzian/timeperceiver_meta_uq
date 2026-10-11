#!/usr/bin/env python3
"""Verify Gate 5b postprocessing, protected sources, and paper integration."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import PATCHTST, PATCHTST_COMMIT, TIMEPERCEIVER, TIMEPERCEIVER_COMMIT, dump, load, sha256_file, write_report  # noqa: E402


REQUIRED_GATE5B = (
    "results/gate5b_formal_decision.json",
    "reports/gate5b_formal_decision.csv",
    "reports/gate5b_patchtst_probability_results.csv",
    "reports/gate5b_patchtst_paired_deltas.csv",
    "results/gate5b_tail_diagnostic.json",
    "reports/gate5b_patchtst_tail_nll_gain.csv",
    "results/gate5b_paper_update.json",
    "paper/tables/table10_second_backbone_replication.csv",
    "paper/tables/table10_second_backbone_replication.md",
    "paper/tables/table10_second_backbone_replication.tex",
    "paper/figures/gate5b/fig_gate5b_patchtst_tail_nll_gain.png",
    "paper/figures/gate5b/fig_gate5b_patchtst_tail_nll_gain.pdf",
    "paper/figures/gate5b/fig_gate5b_patchtst_tail_nll_gain_data.csv",
    "paper/figures/gate5b/fig_gate5b_cross_backbone_deltas.png",
    "paper/figures/gate5b/fig_gate5b_cross_backbone_deltas.pdf",
    "paper/figures/gate5b/fig_gate5b_cross_backbone_deltas_data.csv",
    "paper/sections/second_backbone_replication.md",
    "results/final_evidence_registry.json",
    "results/final_evidence_registry_verification.json",
    "results/final_metrics_consistency.json",
    "results/final_claim_evidence_registry.json",
    "results/paper_artifact_verification.json",
    "results/final_project_status.json",
)


def csv_rows(path):
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def tracked_state(path):
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    lines = subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True).splitlines()
    return commit, [line for line in lines if not line.startswith("??")], [line for line in lines if line.startswith("??")]


def related_processes():
    output = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,args="], text=True)
    markers = ("gate5b_scheduler.py", "gate5b_train_backbone.py", "gate5b_train_probability.py", "gate5b_tail_nll_gain.py")
    rows = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or int(fields[0]) == os.getpid():
            continue
        if any(marker in fields[2] for marker in markers):
            rows.append({"pid": int(fields[0]), "ppid": int(fields[1]), "command": fields[2]})
    return rows


def main():
    missing = [path for path in REQUIRED_GATE5B if not (ROOT / path).is_file() or (ROOT / path).stat().st_size == 0]
    protected = load(ROOT / "results/gate5b_protected_assets_preflight.json")
    protected_rows = []
    for row in protected["protected_assets"]:
        current = sha256_file(ROOT / row["path"])
        protected_rows.append({**row, "current_sha256": current, "match": current == row["sha256"]})
    backbone = load(ROOT / "runtime/gate5b_patchtst_backbones_scheduler_state.json")
    probability = load(ROOT / "runtime/gate5b_patchtst_probability_scheduler_state.json")
    decision = load(ROOT / "results/gate5b_formal_decision.json")
    registry = load(ROOT / "results/final_evidence_registry.json")
    gate5b_rows = [row for row in registry["rows"] if row["stage"] == "gate5b_second_backbone"]
    claims = load(ROOT / "results/final_claim_evidence_registry.json")
    tp_commit, tp_tracked, tp_untracked = tracked_state(TIMEPERCEIVER)
    pt_commit, pt_tracked, pt_untracked = tracked_state(PATCHTST)
    processes = related_processes()
    checks = {
        "required_artifacts_nonempty": not missing,
        "protected_gate4a_to_gate5a_sources_unchanged": all(row["match"] for row in protected_rows),
        "backbone_queue_complete_3_of_3": backbone.get("status") == "COMPLETE" and all(task.get("status") == "SUCCEEDED" for task in backbone.get("tasks", {}).values()) and len(backbone.get("tasks", {})) == 3,
        "probability_queue_complete_6_of_6": probability.get("status") == "COMPLETE" and all(task.get("status") == "SUCCEEDED" for task in probability.get("tasks", {}).values()) and len(probability.get("tasks", {})) == 6,
        "implementation_pass": decision.get("GATE5B_IMPLEMENTATION_PASS") is True,
        "point_tradeoff_decision_preserved": decision.get("decision") == "SECOND_BACKBONE_POINT_TRADEOFF" and decision.get("GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE") is False,
        "tail_diagnostic_secondary": load(ROOT / "results/gate5b_tail_diagnostic.json").get("formal_gate_evidence") is False,
        "paper_update_pass": load(ROOT / "results/gate5b_paper_update.json").get("status") == "PASS",
        "registry_gate5b_rows_6": len(gate5b_rows) == 6 and all(row.get("is_formal") is True and row.get("is_valid") is True and row.get("paper_evidence") is True for row in gate5b_rows),
        "registry_verification_pass": load(ROOT / "results/final_evidence_registry_verification.json").get("FINAL_EVIDENCE_REGISTRY_PASS") is True,
        "metrics_consistency_pass": load(ROOT / "results/final_metrics_consistency.json").get("FINAL_METRICS_CONSISTENCY_PASS") is True,
        "paper_verification_pass": load(ROOT / "results/paper_artifact_verification.json").get("FINAL_PAPER_ARTIFACT_PASS") is True,
        "unsupported_claims_zero": claims.get("unsupported_claim_count") == 0 and any(claim.get("claim_id") == "C9" for claim in claims.get("claims", [])),
        "invalid_legacy_paper_rows_zero": not any(row.get("legacy_unmasked_invalid") is True and row.get("paper_evidence") is True for row in registry["rows"]),
        "smoke_paper_rows_zero": not any(row.get("is_smoke") is True and row.get("paper_evidence") is True for row in registry["rows"]),
        "timeperceiver_commit_and_tracked_diff": tp_commit == TIMEPERCEIVER_COMMIT and not tp_tracked,
        "patchtst_commit_and_tracked_diff": pt_commit == PATCHTST_COMMIT and not pt_tracked,
        "related_process_count_zero": not processes,
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE5B_FORMAL_INTEGRITY_PASS": passed,
        "checks": checks,
        "missing_artifacts": missing,
        "protected_assets": protected_rows,
        "invalid_legacy_paper_rows": sum(row.get("legacy_unmasked_invalid") is True and row.get("paper_evidence") is True for row in registry["rows"]),
        "smoke_paper_rows": sum(row.get("is_smoke") is True and row.get("paper_evidence") is True for row in registry["rows"]),
        "unsupported_claims": claims.get("unsupported_claim_count"),
        "hash_mismatches": sum(not row["match"] for row in protected_rows),
        "related_processes": processes,
        "timeperceiver": {"commit": tp_commit, "tracked": tp_tracked, "untracked": tp_untracked},
        "patchtst": {"commit": pt_commit, "tracked": pt_tracked, "untracked": pt_untracked},
    }
    dump(ROOT / "results/gate5b_postprocess_integrity.json", payload)
    write_report(ROOT / "reports/gate5b_postprocess_integrity.md", "Gate 5b Postprocess Integrity", payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
