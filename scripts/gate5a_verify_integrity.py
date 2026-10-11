#!/usr/bin/env python3
"""Verify Gate 5a paper integration without changing formal source evidence."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_common import dump, load, sha256_file, write_json_markdown


IMMUTABLE = (
    "results/gate4a_formal_decision.json",
    "results/gate4b_multiseed_formal_decision.json",
    "results/gate4c_formal_decision.json",
    "reports/gate4b_multiseed_results.csv",
    "reports/gate4b_multiseed_paired_deltas.csv",
)


def csv_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    before = load(ROOT / "results/gate5a_protected_sources_before.json")
    before_map = {row["path"]: row["sha256"] for row in before["files"]}
    immutable_after = {path: sha256_file(ROOT / path) for path in IMMUTABLE}
    immutable_unchanged = all(before_map.get(path) == digest for path, digest in immutable_after.items())
    previous_final = load(ROOT / "results/gate5a_previous_final_synthesis_hash.json")
    registry = load(ROOT / "results/final_evidence_registry.json")
    paper_rows = [row for row in registry["rows"] if row.get("paper_evidence") is True]
    gate5a_rows = [row for row in registry["rows"] if row["stage"] == "gate5a_distribution_ablation"]
    table8 = csv_rows(ROOT / "paper/tables/table8_distribution_head_ablation.csv")
    table9 = csv_rows(ROOT / "paper/tables/table9_distribution_efficiency.csv")
    valid_ids = {row["evidence_id"] for row in gate5a_rows}
    figures = (
        "fig_gate5a_pit_histograms", "fig_gate5a_qq", "fig_gate5a_tail_nll_gain", "fig_gate5a_kurtosis_df",
    )
    figure_trace = all(
        all((ROOT / f"paper/figures/gate5a/{name}{suffix}").is_file() for suffix in (".png", ".pdf", "_data.csv"))
        for name in figures
    )
    upstream_lines = subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream/TimePerceiver"), "status", "--short"], text=True
    ).splitlines()
    tracked = [line for line in upstream_lines if not line.startswith("??")]
    checks = {
        "gate4a_to_gate4c_decisions_unchanged": immutable_unchanged,
        "gate4b_primary_csv_unchanged": all(before_map.get(path) == immutable_after[path] for path in IMMUTABLE if "gate4b_multiseed" in path),
        "previous_final_synthesis_hash_recorded": previous_final.get("sha256") == before_map.get("reports/final_project_synthesis.md"),
        "old_unmasked_paper_rows_zero": not any(row.get("legacy_unmasked_invalid") is True for row in paper_rows),
        "smoke_paper_rows_zero": not any(row.get("is_smoke") is True for row in paper_rows),
        "gate5a_formal_rows_12": len(gate5a_rows) == 12 and all(row.get("is_formal") is True and row.get("is_valid") is True for row in gate5a_rows),
        "table8_traceable": len(table8) == 12 and all(row.get("source_evidence_ids") in valid_ids for row in table8),
        "table9_traceable": len(table9) == 12 and all(row.get("source_evidence_ids") in valid_ids for row in table9),
        "gate5a_figures_traceable": figure_trace,
        "canonical_registry_pass": registry.get("status") == "PASS",
        "final_synthesis_pass": load(ROOT / "results/final_project_status.json").get("FINAL_SYNTHESIS_PASS") is True,
        "upstream_tracked_diff_empty": not tracked,
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE5A_ARTIFACT_VERIFICATION_PASS": passed,
        "checks": checks,
        "immutable_before": {path: before_map.get(path) for path in IMMUTABLE},
        "immutable_after": immutable_after,
        "upstream_tracked_status": tracked,
    }
    dump(ROOT / "results/gate5a_artifact_verification.json", payload)
    write_json_markdown(ROOT / "reports/gate5a_artifact_verification.md", "Gate 5a Artifact Verification", payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
