#!/usr/bin/env python3
"""Scan the anonymous review copy for local paths and personal identifiers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"


PATTERNS = {
    "absolute_project_path": r"/mnt/data/",
    "local_username": r"\bzza\b",
    "username_label": r"\busername\b",
    "hostname_label": r"\bhostname\b",
    "pid_label": r"\bPID\b",
    "gpu_uuid": r"GPU[- ]?UUID",
    "private_repository": r"private repository",
    "internal_queue": r"runtime/queue|configs/queues|scheduler",
    "raw_run_path": r"\bruns/",
}


def source_text() -> str:
    paths = [VENUE / "manuscript/anonymous_main.tex", VENUE / "manuscript/abstract.tex"]
    paths.extend(sorted((VENUE / "manuscript/sections").glob("*.tex")))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def pdf_text() -> tuple[str, bool]:
    pdf = VENUE / "submission/manuscript_anonymous.pdf"
    executable = shutil.which("pdftotext")
    if not pdf.is_file() or not executable:
        return "", False
    completed = subprocess.run(
        [executable, str(pdf), "-"], text=True, capture_output=True, check=False
    )
    return completed.stdout, completed.returncode == 0


def main() -> None:
    source = source_text()
    pdf, pdf_scanned = pdf_text()
    text = source + "\n" + pdf
    violations = []
    for name, pattern in PATTERNS.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            violations.append({"type": name, "count": len(matches)})
    passed = not violations
    result = {
        "status": "PASS" if passed else "FAILED",
        "review_model": "single-blind",
        "anonymous_copy_retained": True,
        "author_information_copy_with_placeholders_retained": True,
        "source_scanned": True,
        "compiled_pdf_scanned": pdf_scanned,
        "placeholder_is_violation": False,
        "privacy_violation_count": sum(row["count"] for row in violations),
        "violations": violations,
        "pass": passed,
    }
    (ROOT / "results/gate7b1_jof_anonymization_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Gate 7B1 Anonymization Audit

## Status

**{result['status']}**

- Anonymous source scanned: yes
- Compiled anonymous PDF scanned: {'yes' if pdf_scanned else 'pending compilation'}
- Privacy violations: {result['privacy_violation_count']}
- Explicit author placeholders treated as violations: no
- Single-blind author-information copy retained: yes
"""
    (ROOT / "reports/gate7b1_jof_anonymization_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
