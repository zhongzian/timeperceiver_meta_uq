#!/usr/bin/env python3
"""Perform the terminal Gate 6 manuscript and formal-integrity checks."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "paper/submission"
TIMEPERCEIVER_COMMIT = "7e30cc07b51c709f408409fd60a34c81ae8990be"
PATCHTST_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
REQUIRED_TEXT_ASSETS = {
    "abstract.md",
    "contributions.md",
    "experimental_setup.md",
    "results.md",
    "limitations.md",
    "conclusion.md",
}
TABLE10_REQUIRED_FIELDS = {
    "dataset",
    "backbone",
    "gaussian_nll",
    "fixed_t8_nll",
    "delta_nll",
    "gaussian_crps",
    "fixed_t8_crps",
    "delta_crps",
    "gaussian_mse",
    "fixed_t8_mse",
    "delta_mse",
    "relative_mse_change_pct",
    "nll_win",
    "crps_noninferior",
    "point_noninferior",
    "full_pair_pass",
}


def generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha_snapshot(relative: str) -> dict[str, str]:
    rows = {}
    path = ROOT / relative
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        rows[filename.strip()] = digest
    return rows


def git_state(relative: str, expected_commit: str) -> dict[str, object]:
    repo = ROOT / relative
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
    ).splitlines()
    untracked = [
        line
        for line in subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True
        ).splitlines()
        if line.startswith("??")
    ]
    return {
        "path": relative,
        "commit": commit,
        "expected_commit": expected_commit,
        "commit_matches": commit == expected_commit,
        "tracked_diff_empty": not tracked,
        "tracked_status": tracked,
        "untracked": untracked,
    }


def verify_manifest() -> tuple[bool, dict[str, object]]:
    manifest_path = SUBMISSION / "submission_manifest.json"
    if not manifest_path.is_file():
        return False, {"missing_manifest": True, "mismatches": ["submission_manifest.json missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    expected_files = set()
    for row in manifest.get("files", []):
        relative = row["file"]
        expected_files.add(relative)
        target = SUBMISSION / relative
        source = ROOT / row["source"]
        if not target.is_file():
            mismatches.append(f"missing asset: {relative}")
            continue
        if sha256(target) != row["sha256"]:
            mismatches.append(f"manifest hash: {relative}")
        if target.stat().st_size != row["size"]:
            mismatches.append(f"manifest size: {relative}")
        if row.get("anonymous") is not True:
            mismatches.append(f"anonymous flag: {relative}")
        if not source.is_file() or sha256(source) != sha256(target):
            mismatches.append(f"source copy mismatch: {relative}")
    actual_files = {
        str(path.relative_to(SUBMISSION))
        for path in SUBMISSION.rglob("*")
        if path.is_file() and path.name not in {"submission_manifest.json", "submission_manifest.md"}
    }
    if actual_files != expected_files:
        mismatches.append(
            f"manifest inventory difference: missing={sorted(expected_files - actual_files)} extra={sorted(actual_files - expected_files)}"
        )
    details = {
        "file_count": len(expected_files),
        "actual_file_count": len(actual_files),
        "text_sections": sum(path in expected_files for path in REQUIRED_TEXT_ASSETS),
        "table_files": sum(path.startswith("tables/") for path in expected_files),
        "figure_files": sum(path.startswith("figures/") for path in expected_files),
        "appendix_files": sum(path.startswith("appendix/") for path in expected_files),
        "mismatches": mismatches,
    }
    return not mismatches and REQUIRED_TEXT_ASSETS.issubset(expected_files), details


def verify_manuscript() -> tuple[bool, dict[str, object]]:
    abstract = (ROOT / "paper/sections/abstract_draft.md").read_text(encoding="utf-8")
    contributions = (ROOT / "paper/sections/contributions.md").read_text(encoding="utf-8")
    setup = (ROOT / "paper/sections/experimental_setup.md").read_text(encoding="utf-8")
    main = (ROOT / "paper/sections/main_results.md").read_text(encoding="utf-8")
    second = (ROOT / "paper/sections/second_backbone_replication.md").read_text(encoding="utf-8")
    limitations = (ROOT / "paper/sections/limitations.md").read_text(encoding="utf-8")
    conclusion = (ROOT / "paper/sections/conclusion.md").read_text(encoding="utf-8")
    negatives = (ROOT / "paper/sections/ablation_and_negative_results.md").read_text(encoding="utf-8")
    table10 = read_csv("paper/tables/table10_second_backbone_replication.csv")
    figure_rows = read_csv("paper/figures/gate5b/fig_gate5b_cross_backbone_deltas_data.csv")
    electricity_figure = next(
        (row for row in figure_rows if row["dataset"] == "electricity" and row["backbone"] == "PatchTST"),
        None,
    )
    checks = {
        "abstract_protocol_complete": all(
            phrase in abstract
            for phrase in ("train-only normalization", "fixed validation and test masks", "historical inputs")
        ),
        "abstract_fixed_t8_and_capacity": "fixed df=8" in abstract and "same trainable parameter count" in abstract,
        "abstract_timeperceiver_scope": "three datasets and four probability-head training seeds" in abstract,
        "abstract_patchtst_tradeoff": "1.194% Electricity MSE increase" in abstract and "complete second-backbone gate negative" in abstract,
        "three_contributions_exact": len(re.findall(r"(?m)^\d+\. ", contributions)) == 3,
        "setup_no_future_target_masking": "future targets are never masked" in setup,
        "main_results_order": [
            main.find("## 1. Clean Official TimePerceiver Baselines"),
            main.find("## 2. Fixed-Mask Degradation"),
            main.find("## 3. TimePerceiver Student-t Multi-Seed Results"),
            main.find("## 4. Distribution-Head Ablation"),
            main.find("## 5. PatchTST Second-Backbone Replication"),
            main.find("## 6. Final Synthesis"),
        ] == sorted(
            [
                main.find("## 1. Clean Official TimePerceiver Baselines"),
                main.find("## 2. Fixed-Mask Degradation"),
                main.find("## 3. TimePerceiver Student-t Multi-Seed Results"),
                main.find("## 4. Distribution-Head Ablation"),
                main.find("## 5. PatchTST Second-Backbone Replication"),
                main.find("## 6. Final Synthesis"),
            ]
        ) and all(
            main.find(heading) >= 0
            for heading in (
                "## 1. Clean Official TimePerceiver Baselines",
                "## 2. Fixed-Mask Degradation",
                "## 3. TimePerceiver Student-t Multi-Seed Results",
                "## 4. Distribution-Head Ablation",
                "## 5. PatchTST Second-Backbone Replication",
                "## 6. Final Synthesis",
            )
        ),
        "timeperceiver_multiseed_claim_scoped": "12/12 cells" in main and "probability-head training seeds" in main,
        "fixed_df_sufficient": "fixed df=8 is empirically sufficient" in main.lower() and "learned per-channel df does not provide consistent additional value" in main.lower(),
        "gate5b_protocol_integrated": all(
            phrase in second for phrase in ("input length 384", "prediction length 96", "random historical missingness at rate 0.3", "seed 2025")
        ),
        "gate5b_decision_integrated": "SECOND_BACKBONE_POINT_TRADEOFF" in second,
        "electricity_tradeoff_visible": all(
            phrase in second for phrase in ("0.165829 to 0.167809", "1.194%", "pre-registered 1% tolerance")
        ),
        "tail_diagnostic_scoped": all(
            phrase in second for phrase in ("approximately 1.82e8", "descriptive analysis only", "not a formal gate", "not a causal mechanism proof")
        ),
        "limitations_complete": len(re.findall(r"(?m)^\d+\. ", limitations)) >= 10,
        "gate4a_negative_retained": "FiLM adapter" in negatives and "worse than the static adapter" in negatives,
        "gate4c_negative_retained": "additive conformal" in negatives and "worsen absolute coverage error" in negatives,
        "conclusion_complete": "probabilistic-score robustness across the two evaluated backbones" in conclusion and "Electricity MSE tradeoff" in conclusion,
        "table10_fields_complete": bool(table10) and TABLE10_REQUIRED_FIELDS.issubset(table10[0]),
        "table10_electricity_failure_visible": any(
            row["dataset"] == "Electricity"
            and row["backbone"] == "PatchTST"
            and row["relative_mse_change_pct"] == "+1.194%"
            and row["point_noninferior"] == "False"
            and row["full_pair_pass"] == "False"
            for row in table10
        ),
        "cross_backbone_figure_complete": all(
            (ROOT / f"paper/figures/gate5b/fig_gate5b_cross_backbone_deltas.{suffix}").is_file()
            for suffix in ("png", "pdf")
        ) and (ROOT / "paper/figures/gate5b/fig_gate5b_cross_backbone_deltas_caption.md").is_file(),
        "cross_backbone_figure_electricity_positive_mse": electricity_figure is not None and float(electricity_figure["delta_mse"]) > 0,
    }
    return all(checks.values()), checks


def main() -> None:
    gate5b = read_json("results/gate5b_formal_decision.json")
    gate5b_closure = read_json("results/gate5b_closure_verification.json")
    evidence = read_json("results/final_evidence_registry_verification.json")
    claims = read_json("results/final_claim_evidence_registry.json")
    claim_scan = read_json("results/gate6_claim_scan.json")
    number_check = read_json("results/gate6_number_verification.json")
    anonymization = read_json("results/gate6_anonymization_audit.json")
    paper_verification = read_json("results/paper_artifact_verification.json")
    source_snapshot = read_sha_snapshot("artifacts/gate6_pre_finalization/formal_source_hashes.sha256")
    formal_mismatches = []
    for relative, expected in source_snapshot.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            formal_mismatches.append({"path": relative, "expected": expected, "actual": actual})

    timeperceiver = git_state("upstream/TimePerceiver", TIMEPERCEIVER_COMMIT)
    patchtst = git_state("upstream/PatchTST", PATCHTST_COMMIT)
    manifest_pass, manifest_details = verify_manifest()
    manuscript_pass, manuscript_checks = verify_manuscript()

    gate5b_checks = {
        "implementation_pass": gate5b.get("GATE5B_IMPLEMENTATION_PASS") is True,
        "strict_closure_pass": gate5b_closure.get("GATE5B_STRICT_CLOSURE_PASS") is True,
        "decision": gate5b.get("decision") == "SECOND_BACKBONE_POINT_TRADEOFF",
        "nll_wins_3": gate5b.get("GATE5B_NLL_WIN_COUNT") == 3,
        "crps_noninferior_3": gate5b.get("GATE5B_CRPS_NONINFERIOR_COUNT") == 3,
        "point_noninferior_2": gate5b.get("GATE5B_POINT_NONINFERIOR_COUNT") == 2,
        "full_pair_pass_2": gate5b.get("GATE5B_FULL_PAIR_PASS_COUNT") == 2,
        "second_backbone_signal_false": gate5b.get("GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE") is False,
        "complete_direction_false": gate5b.get("CROSS_BACKBONE_DIRECTION_CONSISTENT") is False,
        "ready_for_paper": gate5b.get("READY_FOR_PAPER_FINALIZATION") is True,
    }
    registry_checks = {
        "invalid_legacy_paper_rows_zero": evidence["counts"]["invalid_legacy_rows_in_paper"] == 0,
        "smoke_paper_rows_zero": evidence["counts"]["smoke_rows_in_paper"] == 0,
        "unsupported_claims_zero": claims.get("unsupported_claim_count") == 0,
    }
    formal_integrity_pass = (
        bool(source_snapshot)
        and not formal_mismatches
        and timeperceiver["commit_matches"]
        and timeperceiver["tracked_diff_empty"]
        and patchtst["commit_matches"]
        and patchtst["tracked_diff_empty"]
        and all(gate5b_checks.values())
        and all(registry_checks.values())
        and paper_verification.get("FINAL_PAPER_ARTIFACT_PASS") is True
    )
    claim_pass = claim_scan.get("GATE6_CLAIM_SCAN_PASS") is True and claim_scan.get("prohibited_unsupported_wording_count") == 0
    number_pass = (
        number_check.get("GATE6_NUMBER_VERIFICATION_PASS") is True
        and number_check.get("number_mismatch_count") == 0
        and number_check.get("gate_interpretation_mismatch_count") == 0
    )
    anonymization_pass = (
        anonymization.get("GATE6_ANONYMIZATION_PASS") is True
        and anonymization.get("anonymization_violation_count") == 0
    )
    gates = {
        "GATE6_MANUSCRIPT_INTEGRATION_PASS": manuscript_pass,
        "GATE6_CLAIM_SCAN_PASS": claim_pass,
        "GATE6_NUMBER_VERIFICATION_PASS": number_pass,
        "GATE6_ANONYMIZATION_PASS": anonymization_pass,
        "GATE6_SUBMISSION_ASSET_PASS": manifest_pass and manifest_details.get("text_sections") == 6,
        "GATE6_FORMAL_INTEGRITY_PASS": formal_integrity_pass,
    }
    ready = all(gates.values())
    status = "SUBMISSION_READY" if ready else "BLOCKED_PAPER_FINALIZATION"
    now = generated_at()
    payload = {
        "status": status,
        **gates,
        "GATE6_SUBMISSION_READY": ready,
        "gate5b_decision": gate5b.get("decision"),
        "gate5b_nll_wins": gate5b.get("GATE5B_NLL_WIN_COUNT"),
        "gate5b_crps_noninferior": gate5b.get("GATE5B_CRPS_NONINFERIOR_COUNT"),
        "gate5b_point_noninferior": gate5b.get("GATE5B_POINT_NONINFERIOR_COUNT"),
        "gate5b_full_pair_pass": gate5b.get("GATE5B_FULL_PAIR_PASS_COUNT"),
        "paper_main_claim": "Fixed heavy-tailed probabilistic forecasting improves NLL and CRPS under the evaluated fixed-mask protocol.",
        "paper_second_backbone_interpretation": "The probabilistic-score direction transfers, but PatchTST Electricity prevents complete multi-metric consistency.",
        "limitations_complete": manuscript_checks.get("limitations_complete", False),
        "negative_results_retained": manuscript_checks.get("gate4a_negative_retained", False) and manuscript_checks.get("gate4c_negative_retained", False),
        "generated_at": now,
        "checks": {
            "gate5b": gate5b_checks,
            "registry": registry_checks,
            "manuscript": manuscript_checks,
        },
        "verification_counts": {
            "prohibited_wording": claim_scan.get("prohibited_unsupported_wording_count"),
            "number_mismatches": number_check.get("number_mismatch_count"),
            "gate_interpretation_mismatches": number_check.get("gate_interpretation_mismatch_count"),
            "anonymization_violations": anonymization.get("anonymization_violation_count"),
            "invalid_legacy_paper_rows": evidence["counts"]["invalid_legacy_rows_in_paper"],
            "smoke_paper_rows": evidence["counts"]["smoke_rows_in_paper"],
            "unsupported_claims": claims.get("unsupported_claim_count"),
            "source_hash_mismatches": len(formal_mismatches),
        },
        "submission_assets": manifest_details,
        "formal_integrity": {
            "pre_finalization_snapshot_recorded": (ROOT / "artifacts/gate6_pre_finalization/source_hashes.sha256").is_file(),
            "formal_source_snapshot_count": len(source_snapshot),
            "source_hash_mismatches": formal_mismatches,
            "timeperceiver": timeperceiver,
            "patchtst": patchtst,
        },
    }
    (ROOT / "results/gate6_submission_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results/gate6_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = f"""# Gate 6 Submission Verification

**Status: {status}**

## Final Gates

""" + "\n".join(f"- {key}: {value}" for key, value in gates.items()) + f"""
- GATE6_SUBMISSION_READY: {ready}

## Gate 5b

- Decision: {gate5b.get('decision')}
- NLL wins: {gate5b.get('GATE5B_NLL_WIN_COUNT')}/3
- CRPS noninferior: {gate5b.get('GATE5B_CRPS_NONINFERIOR_COUNT')}/3
- Point noninferior: {gate5b.get('GATE5B_POINT_NONINFERIOR_COUNT')}/3
- Full-pair pass: {gate5b.get('GATE5B_FULL_PAIR_PASS_COUNT')}/3
- Electricity relative MSE change: +1.194%

## Verification Counts

- Prohibited wording: {claim_scan.get('prohibited_unsupported_wording_count')}
- Number mismatches: {number_check.get('number_mismatch_count')}
- Gate interpretation mismatches: {number_check.get('gate_interpretation_mismatch_count')}
- Anonymization violations: {anonymization.get('anonymization_violation_count')}
- Invalid legacy paper rows: {evidence['counts']['invalid_legacy_rows_in_paper']}
- Smoke paper rows: {evidence['counts']['smoke_rows_in_paper']}
- Unsupported claims: {claims.get('unsupported_claim_count')}
- Source hash mismatches: {len(formal_mismatches)}

## Submission Assets

- Text sections: {manifest_details.get('text_sections', 0)}
- Table files: {manifest_details.get('table_files', 0)}
- Figure files: {manifest_details.get('figure_files', 0)}
- Appendix files: {manifest_details.get('appendix_files', 0)}
"""
    (ROOT / "reports/gate6_submission_verification.md").write_text(report, encoding="utf-8")
    (ROOT / "reports/gate6_status.md").write_text(report.replace("Submission Verification", "Status", 1), encoding="utf-8")

    final_status = read_json("results/final_project_status.json")
    final_status.update(
        {
            "status": status,
            **gates,
            "GATE6_SUBMISSION_READY": ready,
            "gate5b_implementation_pass": True,
            "gate5b_strict_closure_pass": True,
            "gate5b_decision": "SECOND_BACKBONE_POINT_TRADEOFF",
            "gate5b_nll_win_count": 3,
            "gate5b_crps_noninferior_count": 3,
            "gate5b_point_noninferior_count": 2,
            "gate5b_full_pair_pass_count": 2,
            "gate5b_second_backbone_signal_positive": False,
            "cross_backbone_direction_consistent": False,
            "probabilistic_direction_across_two_backbones": True,
            "full_metric_consistency_across_two_backbones": False,
            "ready_for_paper_finalization": True,
            "cross_backbone_nll_direction_consistent": True,
            "cross_backbone_crps_direction_consistent": True,
            "cross_backbone_point_noninferiority_complete": False,
            "generated_at": now,
            "gate6_verification_path": "results/gate6_submission_verification.json",
        }
    )
    (ROOT / "results/final_project_status.json").write_text(json.dumps(final_status, indent=2) + "\n", encoding="utf-8")
    final_report = f"""# Final Project Status

**Status: {status}**

- Gate 5b decision: `SECOND_BACKBONE_POINT_TRADEOFF`
- Gate 5b strict closure: PASS
- Gate 5b counts (NLL/CRPS/point/full): 3/3, 3/3, 2/3, 2/3
- Probabilistic direction across two evaluated backbones: True
- Complete multi-metric consistency: False
- Gate 6 submission ready: {ready}
- Formal source hash mismatches: {len(formal_mismatches)}
- TimePerceiver tracked diff empty: {timeperceiver['tracked_diff_empty']}
- PatchTST tracked diff empty: {patchtst['tracked_diff_empty']}
"""
    (ROOT / "reports/final_project_status.md").write_text(final_report, encoding="utf-8")
    print(json.dumps({"status": status, **gates, "GATE6_SUBMISSION_READY": ready}, sort_keys=True))
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
