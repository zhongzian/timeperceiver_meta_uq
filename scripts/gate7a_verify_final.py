#!/usr/bin/env python3
"""Verify all Gate 7A gates, immutable evidence, and upstream repositories."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "gate7a_status.json"
REPORT = ROOT / "reports" / "gate7a_status.md"


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hashes(relative: str) -> dict[str, str]:
    result = {}
    for line in (ROOT / relative).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        result[name.strip()] = expected
    return result


def verify_hashes(relative: str) -> list[dict[str, str]]:
    mismatches = []
    for name, expected in parse_hashes(relative).items():
        path = ROOT / name
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            mismatches.append({"file": name, "expected": expected, "actual": actual})
    return mismatches


def git_state(relative: str, expected_commit: str) -> dict[str, object]:
    path = ROOT / relative
    commit_proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status_proc = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    untracked_proc = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = commit_proc.stdout.strip()
    tracked = [line for line in status_proc.stdout.splitlines() if line.strip()]
    untracked = [line for line in untracked_proc.stdout.splitlines() if line.startswith("??")]
    return {
        "path": relative,
        "commit": commit,
        "expected_commit": expected_commit,
        "commit_matches": commit == expected_commit,
        "tracked_diff_empty": status_proc.returncode == 0 and not tracked,
        "tracked_status": tracked,
        "untracked": untracked,
    }


def main() -> int:
    assembly = load_json("results/gate7a_manuscript_assembly.json")
    references = load_json("results/gate7a_reference_audit.json")
    cross_refs = load_json("results/gate7a_cross_reference_validation.json")
    language = load_json("results/gate7a_language_audit.json")
    numbers = load_json("results/gate7a_number_audit.json")
    anonymization = load_json("results/gate7a_anonymization_audit.json")
    compile_result = load_json("results/gate7a_compile_validation.json")
    package = load_json("results/gate7a_submission_package.json")

    manuscript = ROOT / "paper" / "manuscript"
    required_files = [
        "main.tex", "abstract.tex", "references.bib", "macros.tex", "anonymous_metadata.tex", "README.md",
        *[f"sections/{index:02d}_{name}.tex" for index, name in enumerate([
            "introduction", "related_work", "problem_formulation", "method", "experimental_setup",
            "results", "ablation_negative_results", "limitations", "conclusion",
        ], 1)],
        "appendix/appendix.tex", "appendix/additional_results.tex", "appendix/protocol_details.tex", "appendix/reproducibility.tex",
    ]
    missing_manuscript = [name for name in required_files if not (manuscript / name).is_file()]

    source_mismatches = verify_hashes("artifacts/gate7a_pre_submission/source_hashes.sha256")
    formal_mismatches = verify_hashes("artifacts/gate7a_pre_submission/formal_source_hashes.sha256")
    timeperceiver = git_state("upstream/TimePerceiver", "7e30cc07b51c709f408409fd60a34c81ae8990be")
    patchtst = git_state("upstream/PatchTST", "204c21efe0b39603ad6e2ca640ef5896646ab1a9")
    integrity_pass = all([
        not source_mismatches,
        not formal_mismatches,
        timeperceiver["commit_matches"],
        timeperceiver["tracked_diff_empty"],
        patchtst["commit_matches"],
        patchtst["tracked_diff_empty"],
    ])

    gates = {
        "GATE7A_MANUSCRIPT_ASSEMBLY_PASS": assembly.get("status") == "PASS" and not missing_manuscript,
        "GATE7A_REFERENCE_AUDIT_PASS": references.get("GATE7A_REFERENCE_AUDIT_PASS") is True,
        "GATE7A_CROSS_REFERENCE_PASS": cross_refs.get("GATE7A_CROSS_REFERENCE_PASS") is True,
        "GATE7A_LANGUAGE_AUDIT_PASS": language.get("GATE7A_LANGUAGE_AUDIT_PASS") is True,
        "GATE7A_NUMBER_AUDIT_PASS": numbers.get("GATE7A_NUMBER_AUDIT_PASS") is True,
        "GATE7A_ANONYMIZATION_PASS": anonymization.get("GATE7A_ANONYMIZATION_PASS") is True,
        "GATE7A_COMPILE_PASS": compile_result.get("GATE7A_COMPILE_PASS") is True,
        "GATE7A_SUBMISSION_PACKAGE_PASS": package.get("GATE7A_SUBMISSION_PACKAGE_PASS") is True,
        "GATE7A_FORMAL_INTEGRITY_PASS": integrity_pass,
    }
    ready = all(gates.values())
    title_text = (manuscript / "anonymous_metadata.tex").read_text(encoding="utf-8")
    title_match = re.search(r"\\newcommand\{\\papertitle\}\{([^}]+)\}", title_text)
    abstract_text = (manuscript / "abstract.tex").read_text(encoding="utf-8")
    abstract_words = len(re.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", re.sub(r"\\[%$&_#{}]", "", abstract_text)))

    protected_names = set(parse_hashes("artifacts/gate7a_pre_submission/formal_source_hashes.sha256"))
    decisions_unchanged = not any(
        mismatch["file"].startswith("results/gate") and mismatch["file"].endswith("formal_decision.json")
        for mismatch in formal_mismatches
    )
    primary_metrics_unchanged = not any(mismatch["file"].startswith("reports/") for mismatch in formal_mismatches)
    registry_unchanged = not any(mismatch["file"] == "results/final_evidence_registry.json" for mismatch in formal_mismatches)

    payload = {
        "status": "VENUE_NEUTRAL_MANUSCRIPT_READY" if ready else "BLOCKED_GATE7A_MANUSCRIPT_FINALIZATION",
        **gates,
        "GATE7A_READY_FOR_VENUE_ADAPTATION": ready,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "training": False,
            "inference": False,
            "optimizer": False,
            "backward": False,
            "gpu": False,
            "scheduler": False,
            "formal_metrics_modified": not primary_metrics_unchanged,
            "formal_decisions_modified": not decisions_unchanged,
        },
        "manuscript": {
            "title": title_match.group(1) if title_match else "",
            "abstract_words": abstract_words,
            "main_sections": 9,
            "appendix_sections": 4,
            "tables": assembly.get("table_count"),
            "figures": assembly.get("figure_count"),
            "total_pages": compile_result.get("total_pages"),
            "pdf": compile_result.get("pdf_path"),
            "missing_required_files": missing_manuscript,
        },
        "claims": {
            "fixed_student_t_main_claim": "probabilistic-score improvement under the evaluated fixed-mask protocol",
            "adaptive_df_necessity_claim": False,
            "second_backbone_interpretation": "probabilistic-score direction transfers with a PatchTST Electricity point-prediction tradeoff",
            "electricity_tradeoff_disclosed": "+1.194\\%" in (manuscript / "sections/06_results.tex").read_text(encoding="utf-8"),
            "gate4a_negative_retained": "FiLM" in (manuscript / "sections/07_ablation_negative_results.tex").read_text(encoding="utf-8"),
            "gate4c_negative_retained": "conformal" in (manuscript / "sections/07_ablation_negative_results.tex").read_text(encoding="utf-8").lower(),
            "unsupported_claim_count": language.get("prohibited_unsupported_claim_count"),
        },
        "references": references,
        "cross_references": cross_refs,
        "language": language,
        "numbers": numbers,
        "anonymization": anonymization,
        "compilation": compile_result,
        "submission_package": package,
        "formal_integrity": {
            "protected_file_count": len(protected_names),
            "source_hash_mismatches": source_mismatches,
            "formal_hash_mismatches": formal_mismatches,
            "gate4a_to_gate5b_decisions_unchanged": decisions_unchanged,
            "primary_metrics_unchanged": primary_metrics_unchanged,
            "final_evidence_registry_unchanged": registry_unchanged,
            "timeperceiver": timeperceiver,
            "patchtst": patchtst,
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_lines = [
        "# Gate 7A Status",
        "",
        f"**{payload['status']}**",
        "",
        f"- Ready for venue adaptation: {ready}",
        f"- Manuscript pages: {compile_result.get('total_pages')}",
        f"- Submission package files: {package.get('file_count')}",
        f"- Formal hash mismatches: {len(formal_mismatches)}",
        f"- Frozen source mismatches: {len(source_mismatches)}",
        "",
        "## Final Gates",
        "",
    ]
    report_lines.extend(f"- {name}: {value}" for name, value in gates.items())
    report_lines.append(f"- GATE7A_READY_FOR_VENUE_ADAPTATION: {ready}")
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], **gates, "GATE7A_READY_FOR_VENUE_ADAPTATION": ready}, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
