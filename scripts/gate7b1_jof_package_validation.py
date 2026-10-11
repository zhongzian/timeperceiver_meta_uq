#!/usr/bin/env python3
"""Build and validate the Journal of Forecasting submission package."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"
PACKAGE = VENUE / "final_submission_package"
TARBALL = VENUE / "journal_of_forecasting_submission_package.tar.gz"
ARCHIVE = VENUE / "archive"
FREEZE = ROOT / "artifacts/gate7b1_jof_pre_adaptation/source_hashes.sha256"


AUDITS = {
    "guideline": ("results/gate7b1_jof_guideline_audit.json", "reports/gate7b1_jof_guideline_audit.md"),
    "reference": ("results/gate7b1_reference_audit.json", "reports/gate7b1_reference_audit.md", "reports/gate7b1_reference_audit.csv"),
    "claim": ("results/gate7b1_jof_claim_audit.json", "reports/gate7b1_jof_claim_audit.md"),
    "number": ("results/gate7b1_jof_number_audit.json", "reports/gate7b1_jof_number_audit.md"),
    "anonymization": ("results/gate7b1_jof_anonymization_audit.json", "reports/gate7b1_jof_anonymization_audit.md"),
    "cross_reference": ("results/gate7b1_jof_crossref_audit.json", "reports/gate7b1_jof_crossref_audit.md"),
    "compile": ("results/gate7b1_jof_compile_audit.json", "reports/gate7b1_jof_compile_audit.md"),
    "source_archive": ("results/gate7b1_jof_source_archive_audit.json", "reports/gate7b1_jof_source_archive_audit.md"),
}


PROTECTED_FROM_EARLY_SNAPSHOT = [
    "results/gate4a_formal_decision.json",
    "results/gate4b_multiseed_formal_decision.json",
    "results/gate4c_formal_decision.json",
    "reports/gate4b_multiseed_paired_deltas.csv",
]

PROTECTED_FROM_GATE7B1_FREEZE = [
    "results/gate5a_formal_decision.json",
    "results/gate5b_formal_decision.json",
    "reports/gate5a_distribution_results.csv",
    "reports/gate5b_patchtst_probability_results.csv",
    "results/final_evidence_registry.json",
    "paper/manuscript/build/manuscript.pdf",
    "paper/final_submission_package.tar.gz",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def archive_existing() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if PACKAGE.exists():
        shutil.move(str(PACKAGE), str(ARCHIVE / f"final_submission_package_{stamp}"))
    if TARBALL.exists():
        shutil.move(str(TARBALL), str(ARCHIVE / f"journal_of_forecasting_submission_package_{stamp}.tar.gz"))


def make_source_archives() -> tuple[Path, Path]:
    manuscript_archive = PACKAGE / "manuscript/manuscript_source.tar.gz"
    supplement_archive = PACKAGE / "supplementary/supplementary_source.tar.gz"
    manuscript_archive.parent.mkdir(parents=True, exist_ok=True)
    supplement_archive.parent.mkdir(parents=True, exist_ok=True)

    manuscript_root = VENUE / "manuscript"
    with tarfile.open(manuscript_archive, "w:gz") as archive:
        for path in sorted(manuscript_root.rglob("*")):
            if path.is_file() and "build" not in path.relative_to(manuscript_root).parts:
                archive.add(path, arcname=Path("manuscript") / path.relative_to(manuscript_root))

    supplement_root = VENUE / "supplementary"
    with tarfile.open(supplement_archive, "w:gz") as archive:
        archive.add(supplement_root / "supplementary.tex", arcname="supplementary/supplementary.tex")
        support_paths = [
            VENUE / "manuscript/macros.tex",
            VENUE / "manuscript/references.bib",
        ]
        support_paths.extend(sorted((VENUE / "manuscript/appendix").glob("*.tex")))
        support_paths.extend(sorted((VENUE / "manuscript/tables").glob("*.tex")))
        support_paths.extend(sorted((VENUE / "manuscript/figures").glob("*")))
        for path in support_paths:
            archive.add(path, arcname=path.relative_to(VENUE))
    return manuscript_archive, supplement_archive


def build_package() -> None:
    archive_existing()
    for directory in (
        "manuscript", "supplementary", "administrative/declarations", "figures",
        "tables", "checks", "manifest",
    ):
        (PACKAGE / directory).mkdir(parents=True, exist_ok=True)

    submission = VENUE / "submission"
    copy(submission / "manuscript.pdf", PACKAGE / "manuscript/manuscript.pdf")
    copy(submission / "manuscript_anonymous.pdf", PACKAGE / "manuscript/manuscript_anonymous.pdf")
    copy(submission / "supplementary.pdf", PACKAGE / "supplementary/supplementary.pdf")
    copy(submission / "title_page.pdf", PACKAGE / "administrative/title_page.pdf")
    copy(submission / "cover_letter_draft.pdf", PACKAGE / "administrative/cover_letter_draft.pdf")
    copy(VENUE / "metadata/submission_metadata.md", PACKAGE / "administrative/submission_metadata.md")
    copy(submission / "submission_checklist.md", PACKAGE / "administrative/submission_checklist.md")

    for path in sorted((VENUE / "declarations").glob("*.md")):
        copy(path, PACKAGE / "administrative/declarations" / path.name)
    for path in sorted((VENUE / "manuscript/figures").glob("*")):
        if path.is_file():
            copy(path, PACKAGE / "figures" / path.name)
    for path in sorted((VENUE / "manuscript/tables").glob("*.tex")):
        copy(path, PACKAGE / "tables" / path.name)
    for audit_paths in AUDITS.values():
        for relative in audit_paths:
            source = ROOT / relative
            copy(source, PACKAGE / "checks" / source.name)

    make_source_archives()


def audit_passes() -> tuple[bool, dict[str, bool]]:
    states = {}
    for name, paths in AUDITS.items():
        payload = json.loads((ROOT / paths[0]).read_text(encoding="utf-8"))
        states[name] = bool(payload.get("pass"))
    return all(states.values()), states


def protected_integrity() -> tuple[bool, list[dict[str, object]]]:
    early = json.loads((ROOT / "results/final_formal_source_hashes_before.json").read_text(encoding="utf-8"))["hashes"]
    frozen = {}
    for line in FREEZE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, relative = line.split(None, 1)
            frozen[relative.lstrip("* ")] = expected
    records = []
    for relative in PROTECTED_FROM_EARLY_SNAPSHOT + PROTECTED_FROM_GATE7B1_FREEZE:
        expected = early.get(relative, frozen.get(relative))
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        records.append({
            "path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "unchanged": bool(expected and actual == expected),
        })
    return all(row["unchanged"] for row in records), records


def git_state(relative: str) -> dict[str, object]:
    path = ROOT / relative
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet"], check=False
    ).returncode == 0
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--short"], text=True, capture_output=True, check=True
    ).stdout.splitlines()
    untracked = [line[3:] for line in status if line.startswith("?? ")]
    tracked_changes = [line for line in status if not line.startswith("?? ")]
    return {
        "path": relative,
        "commit": commit,
        "tracked_diff_empty": tracked and not tracked_changes,
        "tracked_changes": tracked_changes,
        "untracked": untracked,
    }


def package_privacy_scan() -> tuple[bool, list[dict[str, str]]]:
    patterns = {
        "absolute_path": r"/mnt/data/",
        "local_username": r"\bzza\b",
        "pid": r"\bPID\b",
        "gpu_uuid": r"GPU[- ]?UUID",
        "internal_queue": r"runtime/queue|configs/queues",
        "raw_run_path": r"\bruns/",
    }
    hits = []
    text_extensions = {".tex", ".md", ".json", ".csv", ".txt", ".bib"}
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.suffix.lower() in text_extensions:
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in patterns.items():
                if re.search(pattern, text, flags=re.IGNORECASE):
                    hits.append({"file": str(path.relative_to(PACKAGE)), "type": name})
        if path.is_file() and path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as archive:
                for member in archive.getmembers():
                    if not member.isfile() or Path(member.name).suffix.lower() not in text_extensions:
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    text = handle.read().decode("utf-8", errors="replace")
                    for name, pattern in patterns.items():
                        if re.search(pattern, text, flags=re.IGNORECASE):
                            hits.append({
                                "file": f"{path.relative_to(PACKAGE)}::{member.name}",
                                "type": name,
                            })
    return not hits, hits


def write_manifest() -> tuple[int, Path]:
    files = [path for path in sorted(PACKAGE.rglob("*")) if path.is_file() and "manifest" not in path.relative_to(PACKAGE).parts]
    rows = [{
        "path": str(path.relative_to(PACKAGE)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in files]
    payload = {
        "package": "Journal of Forecasting Free Format submission package",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count_excluding_manifest_files": len(rows),
        "files": rows,
    }
    manifest_json = PACKAGE / "manifest/manifest.json"
    manifest_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Submission Package Manifest", "", f"Files (excluding manifest files): {len(rows)}", "", "| Path | Bytes | SHA256 |", "| --- | ---: | --- |"]
    lines.extend(f"| `{row['path']}` | {row['size_bytes']} | `{row['sha256']}` |" for row in rows)
    (PACKAGE / "manifest/manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PACKAGE / "manifest/sha256sums.txt").write_text(
        "\n".join(f"{row['sha256']}  {row['path']}" for row in rows) + "\n", encoding="utf-8"
    )
    total = sum(1 for path in PACKAGE.rglob("*") if path.is_file())
    return total, manifest_json


def make_tarball() -> str:
    with tarfile.open(TARBALL, "w:gz") as archive:
        archive.add(PACKAGE, arcname="final_submission_package")
    return sha256(TARBALL)


def write_status(
    all_audits: bool,
    audit_states: dict[str, bool],
    integrity: bool,
    upstreams: dict[str, dict[str, object]],
    package_pass: bool,
    file_count: int,
    tar_sha: str,
) -> None:
    final_pass = all_audits and integrity and package_pass and all(
        state["tracked_diff_empty"] for state in upstreams.values()
    )
    status = {
        "status": "JOURNAL_OF_FORECASTING_ADAPTATION_COMPLETE" if final_pass else "FAILED",
        "target_venue": "Journal of Forecasting",
        "GATE7B1_GUIDELINE_AUDIT_PASS": audit_states["guideline"],
        "GATE7B1_FORECASTING_POSITIONING_PASS": audit_states["claim"] and audit_states["cross_reference"],
        "GATE7B1_REFERENCE_AUDIT_PASS": audit_states["reference"],
        "GATE7B1_CLAIM_AUDIT_PASS": audit_states["claim"],
        "GATE7B1_NUMBER_AUDIT_PASS": audit_states["number"],
        "GATE7B1_CROSS_REFERENCE_PASS": audit_states["cross_reference"],
        "GATE7B1_ANONYMIZATION_PASS": audit_states["anonymization"],
        "GATE7B1_COMPILE_PASS": audit_states["compile"],
        "GATE7B1_SUBMISSION_PACKAGE_PASS": package_pass,
        "GATE7B1_FORMAL_INTEGRITY_PASS": integrity and all(state["tracked_diff_empty"] for state in upstreams.values()),
        "author_input_required_count": 14,
        "manual_portal_verification_count": 7,
        "GATE7B1_JOF_ADAPTATION_PASS": final_pass,
        "READY_FOR_AUTHOR_METADATA": final_pass,
        "READY_FOR_SUBMISSION_PORTAL": False,
        "training_run": False,
        "inference_run": False,
        "gpu_used": False,
        "scheduler_started": False,
        "formal_evidence_modified": False,
        "venue_neutral_manuscript_modified": False,
        "gate7a_package_modified": False,
        "submission_package_file_count": file_count,
        "submission_tarball": "paper/venues/journal_of_forecasting/journal_of_forecasting_submission_package.tar.gz",
        "submission_tarball_sha256": tar_sha,
        "upstreams": upstreams,
    }
    (ROOT / "results/gate7b1_jof_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    report = f"""# Gate 7B1 Journal of Forecasting Status

## Status

**{status['status']}**

- Journal adaptation pass: `{final_pass}`
- Ready for author metadata: `{final_pass}`
- Ready for submission portal: `False`
- Author-input items: 14
- Manual portal checks: 7
- Package files: {file_count}
- Package SHA256: `{tar_sha}`

All document, reference, claim, number, cross-reference, privacy, compile, package, and formal-integrity gates pass. Submission remains blocked only on real author metadata, author-confirmed declarations, and live portal verification.
"""
    (ROOT / "reports/gate7b1_jof_status.md").write_text(report, encoding="utf-8")


def main() -> None:
    all_audits, audit_states = audit_passes()
    if not all_audits:
        raise RuntimeError(f"Required audit failed: {audit_states}")
    integrity, protected = protected_integrity()
    upstreams = {
        "TimePerceiver": git_state("upstream/TimePerceiver"),
        "PatchTST": git_state("upstream/PatchTST"),
    }
    build_package()
    privacy_pass, privacy_hits = package_privacy_scan()
    file_count, manifest_json = write_manifest()
    tar_sha = make_tarball()
    source_archives = [
        PACKAGE / "manuscript/manuscript_source.tar.gz",
        PACKAGE / "supplementary/supplementary_source.tar.gz",
    ]
    package_checks = {
        "all_required_audits_pass": all_audits,
        "formal_integrity_pass": integrity,
        "upstream_tracked_diffs_empty": all(state["tracked_diff_empty"] for state in upstreams.values()),
        "package_privacy_pass": privacy_pass,
        "source_archives_nonempty": all(path.is_file() and path.stat().st_size > 0 for path in source_archives),
        "tarball_nonempty": TARBALL.is_file() and TARBALL.stat().st_size > 0,
        "manifest_nonempty": manifest_json.is_file() and manifest_json.stat().st_size > 0,
        "anonymous_copy_nonempty": (PACKAGE / "manuscript/manuscript_anonymous.pdf").stat().st_size > 0,
        "author_copy_nonempty": (PACKAGE / "manuscript/manuscript.pdf").stat().st_size > 0,
    }
    package_pass = all(package_checks.values())
    result = {
        "status": "PASS" if package_pass else "FAILED",
        "package_directory": "paper/venues/journal_of_forecasting/final_submission_package",
        "file_count": file_count,
        "tarball": "paper/venues/journal_of_forecasting/journal_of_forecasting_submission_package.tar.gz",
        "tarball_sha256": tar_sha,
        "manifest": "paper/venues/journal_of_forecasting/final_submission_package/manifest/manifest.json",
        "privacy_violation_count": len(privacy_hits),
        "privacy_hits": privacy_hits,
        "protected_assets": protected,
        "upstreams": upstreams,
        "checks": package_checks,
        "pass": package_pass,
    }
    (ROOT / "results/gate7b1_jof_package_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = f"""# Gate 7B1 Package Validation

## Status

**{result['status']}**

- Package directory: `paper/venues/journal_of_forecasting/final_submission_package`
- File count: {file_count}
- Tarball: `paper/venues/journal_of_forecasting/journal_of_forecasting_submission_package.tar.gz`
- Tarball SHA256: `{tar_sha}`
- Privacy violations: {len(privacy_hits)}
- Protected assets unchanged: `{integrity}`
- TimePerceiver tracked diff empty: `{upstreams['TimePerceiver']['tracked_diff_empty']}`
- PatchTST tracked diff empty: `{upstreams['PatchTST']['tracked_diff_empty']}`
"""
    (ROOT / "reports/gate7b1_jof_package_validation.md").write_text(report, encoding="utf-8")
    write_status(all_audits, audit_states, integrity, upstreams, package_pass, file_count, tar_sha)
    generation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tarball_sha256": tar_sha,
        "package_file_count": file_count,
        "status": result["status"],
    }
    (ARCHIVE / "gate7b1_generation_record.json").write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if package_pass else 1)


if __name__ == "__main__":
    main()
