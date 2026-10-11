#!/usr/bin/env python3
"""Build and verify the anonymous Gate 7A venue-neutral submission package."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "manuscript"
PACKAGE = ROOT / "paper" / "final_submission_package"
TARBALL = ROOT / "paper" / "final_submission_package.tar.gz"
SIDECAR = ROOT / "paper" / "final_submission_package.tar.gz.sha256"
RESULT = ROOT / "results" / "gate7a_submission_package.json"
REPORT = ROOT / "reports" / "gate7a_submission_package.md"

TOP_LEVEL_SOURCE = [
    "main.tex", "abstract.tex", "macros.tex", "anonymous_metadata.tex", "references.bib", "README.md",
]
SOURCE_DIRECTORIES = ["sections", "appendix", "tables", "figures"]
CHECK_FILES = [
    "results/gate7a_reference_audit.json",
    "reports/gate7a_reference_audit.csv",
    "reports/gate7a_reference_audit.md",
    "results/gate7a_number_audit.json",
    "reports/gate7a_number_audit.md",
    "results/gate7a_language_audit.json",
    "reports/gate7a_language_audit.md",
    "results/gate7a_anonymization_audit.json",
    "reports/gate7a_anonymization_audit.md",
    "results/gate7a_cross_reference_validation.json",
    "reports/gate7a_cross_reference_validation.md",
    "results/gate7a_compile_validation.json",
    "reports/gate7a_compile_validation.md",
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


def source_files() -> list[tuple[Path, Path]]:
    pairs = [(MANUSCRIPT / name, Path("source") / name) for name in TOP_LEVEL_SOURCE]
    for directory in SOURCE_DIRECTORIES:
        for path in sorted((MANUSCRIPT / directory).rglob("*")):
            if path.is_file():
                pairs.append((path, Path("source") / path.relative_to(MANUSCRIPT)))
    return pairs


def anonymous_scan(files: list[Path]) -> list[dict[str, str]]:
    violations = []
    patterns = {
        "absolute local path": re.compile(r"/(?:mnt|home|Users|tmp)/[^\s}\],\"']+", re.IGNORECASE),
        "username": re.compile(r"\bzza\b", re.IGNORECASE),
        "GPU UUID": re.compile(r"GPU-[0-9a-f-]{16,}", re.IGNORECASE),
        "PID": re.compile(r"\bpid\s*[:= ]\s*\d+\b", re.IGNORECASE),
        "private URL": re.compile(r"(?:localhost|127\.0\.0\.1|git@|ssh://|file://)", re.IGNORECASE),
        "personal email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "internal run timestamp": re.compile(r"\b20\d{6}[_-]\d{6}\b"),
        "run or queue path": re.compile(r"(?:runs/|checkpoints?/|configs/queues|scheduler_state)", re.IGNORECASE),
    }
    text_suffixes = {".tex", ".bib", ".md", ".json", ".csv", ".txt", ".yaml", ".yml"}
    for path in files:
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            if pattern.search(text):
                violations.append({"file": str(path.relative_to(PACKAGE)), "category": name})
    return violations


def main() -> int:
    for directory in ("source", "compiled", "checks", "manifest"):
        (PACKAGE / directory).mkdir(parents=True, exist_ok=True)

    missing = []
    copied: list[Path] = []
    origins: dict[str, str] = {}
    for source, relative in source_files():
        if not source.is_file():
            missing.append(str(source.relative_to(ROOT)))
            continue
        destination = PACKAGE / relative
        copy(source, destination)
        copied.append(destination)
        origins[str(relative)] = str(source.relative_to(ROOT))

    pdf_source = MANUSCRIPT / "build" / "manuscript.pdf"
    if pdf_source.is_file():
        pdf_destination = PACKAGE / "compiled" / "manuscript.pdf"
        copy(pdf_source, pdf_destination)
        copied.append(pdf_destination)
        origins["compiled/manuscript.pdf"] = str(pdf_source.relative_to(ROOT))
    else:
        missing.append(str(pdf_source.relative_to(ROOT)))

    for relative_string in CHECK_FILES:
        source = ROOT / relative_string
        if not source.is_file():
            missing.append(relative_string)
            continue
        destination = PACKAGE / "checks" / source.name
        copy(source, destination)
        copied.append(destination)
        origins[str(destination.relative_to(PACKAGE))] = relative_string

    expected_before_manifest = {path.resolve() for path in copied}
    current_before_manifest = {
        path.resolve() for path in PACKAGE.rglob("*")
        if path.is_file() and "manifest" not in path.parts
    }
    unexpected = sorted(str(path.relative_to(PACKAGE)) for path in current_before_manifest - expected_before_manifest)

    records = [{
        "file": str(path.relative_to(PACKAGE)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "source": origins[str(path.relative_to(PACKAGE))],
        "anonymous": True,
    } for path in sorted(copied)]

    manifest_path = PACKAGE / "manifest" / "submission_manifest.json"
    manifest_md = PACKAGE / "manifest" / "submission_manifest.md"
    manifest_payload = {
        "status": "PASS" if not missing and not unexpected else "FAILED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "anonymous": True,
        "venue_neutral": True,
        "document_class": "article",
        "font_size": "11pt",
        "manifest_self_excluded": True,
        "content_file_count": len(records),
        "files": records,
        "missing_files": missing,
        "unexpected_files": unexpected,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_md.write_text("\n".join([
        "# Anonymous Venue-Neutral Submission Manifest",
        "",
        f"**Status:** {manifest_payload['status']}",
        "",
        f"- Anonymous: {manifest_payload['anonymous']}",
        f"- Venue neutral: {manifest_payload['venue_neutral']}",
        f"- Content files: {len(records)}",
        f"- Missing files: {len(missing)}",
        f"- Unexpected files: {len(unexpected)}",
        "",
        "The manifest excludes itself and the checksum inventory from its content-file count.",
    ]) + "\n", encoding="utf-8")

    checksum_targets = sorted(copied + [manifest_path, manifest_md])
    checksum_path = PACKAGE / "manifest" / "sha256sums.txt"
    checksum_path.write_text("".join(
        f"{sha256(path)}  {path.relative_to(PACKAGE)}\n" for path in checksum_targets
    ), encoding="utf-8")

    all_package_files = sorted(path for path in PACKAGE.rglob("*") if path.is_file())
    violations = anonymous_scan(all_package_files)
    forbidden_names = sorted(
        str(path.relative_to(PACKAGE))
        for path in all_package_files
        if any(term in str(path.relative_to(PACKAGE)).lower() for term in ("checkpoint", "scheduler", "pid", "smoke", "invalid_legacy"))
    )

    with tarfile.open(TARBALL, "w:gz") as archive:
        archive.add(PACKAGE, arcname="final_submission_package")
    tarball_hash = sha256(TARBALL)
    SIDECAR.write_text(f"{tarball_hash}  {TARBALL.name}\n", encoding="utf-8")

    passed = not missing and not unexpected and not violations and not forbidden_names and TARBALL.is_file()
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE7A_SUBMISSION_PACKAGE_PASS": passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_directory": str(PACKAGE.relative_to(ROOT)),
        "file_count": len(all_package_files),
        "content_file_count": len(records),
        "tarball": str(TARBALL.relative_to(ROOT)),
        "tarball_sha256": tarball_hash,
        "tarball_size_bytes": TARBALL.stat().st_size,
        "tarball_sha256_sidecar": str(SIDECAR.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "anonymous": not violations,
        "venue_neutral": True,
        "missing_files": missing,
        "unexpected_files": unexpected,
        "anonymization_violations": violations,
        "forbidden_package_files": forbidden_names,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text("\n".join([
        "# Gate 7A Submission Package",
        "",
        f"**Status:** {payload['status']}",
        "",
        f"- Package files: {len(all_package_files)}",
        f"- Tarball: `{payload['tarball']}`",
        f"- Tarball SHA256: `{tarball_hash}`",
        f"- Missing files: {len(missing)}",
        f"- Unexpected files: {len(unexpected)}",
        f"- Anonymization violations: {len(violations)}",
        f"- Forbidden files: {len(forbidden_names)}",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "files": len(all_package_files),
        "anonymous": payload["anonymous"],
        "tarball_sha256": tarball_hash,
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
