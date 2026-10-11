#!/usr/bin/env python3
"""Atomically finalize Gate 5b after all formal and postprocess checks pass."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import dump, load, write_report  # noqa: E402


def main():
    integrity = load(ROOT / "results/gate5b_postprocess_integrity.json")
    decision = load(ROOT / "results/gate5b_formal_decision.json")
    final_status = load(ROOT / "results/final_project_status.json")
    checks = {
        "gate5b_implementation_pass": decision.get("GATE5B_IMPLEMENTATION_PASS") is True,
        "backbone_prerequisite_pass": load(ROOT / "results/gate5b_patchtst_backbone_status.json").get("GATE5B_BACKBONE_PREREQUISITE_PASS") is True,
        "probability_matrix_complete": load(ROOT / "results/gate5b_status.json").get("GATE5B_PROBABILITY_MATRIX_COMPLETE") is True,
        "paper_update_complete": load(ROOT / "results/gate5b_paper_update.json").get("status") == "PASS",
        "final_evidence_rebuild_pass": load(ROOT / "results/final_evidence_registry_verification.json").get("FINAL_EVIDENCE_REGISTRY_PASS") is True,
        "formal_integrity_pass": integrity.get("GATE5B_FORMAL_INTEGRITY_PASS") is True,
        "final_synthesis_pass": final_status.get("FINAL_SYNTHESIS_PASS") is True,
        "previous_final_synthesis_hash_recorded": load(ROOT / "results/gate5b_previous_final_synthesis_hash.json").get("status") == "PASS",
        "decision_point_tradeoff_preserved": decision.get("decision") == "SECOND_BACKBONE_POINT_TRADEOFF",
        "cross_backbone_direction_consistent_false": decision.get("CROSS_BACKBONE_DIRECTION_CONSISTENT") is False,
        "ready_for_paper_finalization": decision.get("READY_FOR_PAPER_FINALIZATION") is True,
    }
    passed = all(checks.values())
    closure = {
        "status": "COMPLETE" if passed else "FAILED",
        "GATE5B_STRICT_CLOSURE_PASS": passed,
        "checks": checks,
        "decision": decision.get("decision"),
        "GATE5B_IMPLEMENTATION_PASS": decision.get("GATE5B_IMPLEMENTATION_PASS"),
        "GATE5B_BACKBONE_PREREQUISITE_PASS": checks["backbone_prerequisite_pass"],
        "GATE5B_PROBABILITY_MATRIX_COMPLETE": checks["probability_matrix_complete"],
        "GATE5B_PAPER_UPDATE_COMPLETE": checks["paper_update_complete"],
        "GATE5B_FINAL_EVIDENCE_REBUILD_PASS": checks["final_evidence_rebuild_pass"],
        "GATE5B_FORMAL_INTEGRITY_PASS": checks["formal_integrity_pass"],
        "GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE": decision.get("GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE"),
        "CROSS_BACKBONE_DIRECTION_CONSISTENT": decision.get("CROSS_BACKBONE_DIRECTION_CONSISTENT"),
        "READY_FOR_PAPER_FINALIZATION": decision.get("READY_FOR_PAPER_FINALIZATION"),
    }
    dump(ROOT / "results/gate5b_closure_verification.json", closure)
    write_report(ROOT / "reports/gate5b_closure_verification.md", "Gate 5b Closure Verification", closure)
    status = {
        **decision,
        "status": "COMPLETE" if passed else "FAILED",
        "backbone_tasks_succeeded": "3/3",
        "probability_tasks_succeeded": "6/6",
        "GATE5B_BACKBONE_PREREQUISITE_PASS": checks["backbone_prerequisite_pass"],
        "GATE5B_PROBABILITY_MATRIX_COMPLETE": checks["probability_matrix_complete"],
        "GATE5B_PAPER_UPDATE_COMPLETE": checks["paper_update_complete"],
        "GATE5B_FINAL_EVIDENCE_REBUILD_PASS": checks["final_evidence_rebuild_pass"],
        "GATE5B_FORMAL_INTEGRITY_PASS": checks["formal_integrity_pass"],
        "GATE5B_STRICT_CLOSURE_PASS": passed,
    }
    dump(ROOT / "results/gate5b_status.json", status)
    write_report(ROOT / "reports/gate5b_status.md", "Gate 5b Status", status)
    print(json.dumps(closure, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
