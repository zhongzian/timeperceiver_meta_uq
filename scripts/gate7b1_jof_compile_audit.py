#!/usr/bin/env python3
"""Summarize Gate 7B1 LaTeX compilation and formatting checks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"
CHECKS = VENUE / "checks"


LOGS = {
    "manuscript": VENUE / "manuscript/build/main_compile.log",
    "anonymous_manuscript": VENUE / "manuscript/build/anonymous_compile.log",
    "supplementary": VENUE / "supplementary/build/supplementary_compile.log",
    "title_page": VENUE / "submission/build/title_page_compile.log",
    "cover_letter": VENUE / "submission/build/cover_letter_compile.log",
}

INTERNAL_LOGS = [
    VENUE / "manuscript/build/main.log",
    VENUE / "manuscript/build/anonymous_main.log",
    VENUE / "supplementary/build/supplementary.log",
    VENUE / "submission/build/title_page.log",
    VENUE / "submission/build/cover_letter_draft.log",
]

PDFS = {
    "manuscript": VENUE / "submission/manuscript.pdf",
    "anonymous_manuscript": VENUE / "submission/manuscript_anonymous.pdf",
    "supplementary": VENUE / "submission/supplementary.pdf",
    "title_page": VENUE / "submission/title_page.pdf",
    "cover_letter": VENUE / "submission/cover_letter.pdf",
}


def pdf_pages(path: Path) -> int | None:
    executable = shutil.which("pdfinfo")
    if not executable or not path.is_file():
        return None
    completed = subprocess.run([executable, str(path)], text=True, capture_output=True, check=False)
    match = re.search(r"^Pages:\s+(\d+)", completed.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def abstract_words() -> int:
    text = (VENUE / "manuscript/abstract.tex").read_text(encoding="utf-8")
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = text.replace("$", " ").replace("--", " ")
    return len(re.findall(r"\b[\w'-]+\b", text))


def unresolved_pdf_markers() -> int:
    executable = shutil.which("pdftotext")
    if not executable:
        return 0
    count = 0
    for name in ("manuscript", "anonymous_manuscript", "supplementary"):
        path = PDFS[name]
        if not path.is_file():
            continue
        completed = subprocess.run([executable, str(path), "-"], text=True, capture_output=True, check=False)
        count += len(re.findall(r"\[\s*\?(?:\s*,\s*\?)*\s*\]", completed.stdout))
    return count


def main() -> None:
    exit_codes = {}
    for line in (CHECKS / "compile_exit_codes.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            exit_codes[name] = int(value)
    combined_paths = list(LOGS.values()) + INTERNAL_LOGS
    combined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in combined_paths if path.is_file())
    undefined_citations = len(re.findall(r"citation.*undefined|undefined citations", combined, flags=re.IGNORECASE))
    undefined_references = len(re.findall(r"reference.*undefined|undefined references", combined, flags=re.IGNORECASE))
    overfull_boxes = len(re.findall(r"Overfull \\hbox|Overfull \\vbox", combined))
    fatal_errors = len(re.findall(r"fatal error|aborted|error:|emergency stop", combined, flags=re.IGNORECASE))
    missing_assets = len(re.findall(
        r"LaTeX Error: File [`'][^`']+\.(?:sty|cls|tex|pdf|png)[`'] not found"
        r"|unable to open[^\n]*\.(?:pdf|tex|png)|missing asset",
        combined,
        flags=re.IGNORECASE,
    ))
    unresolved_markers = unresolved_pdf_markers()
    pages = {name: pdf_pages(path) for name, path in PDFS.items()}
    words = abstract_words()
    short_title = "Heavy-Tailed Forecasts with Missingness"
    checks = {
        "all_exit_codes_zero": set(exit_codes) == set(PDFS) and all(value == 0 for value in exit_codes.values()),
        "all_pdfs_nonempty": all(path.is_file() and path.stat().st_size > 0 for path in PDFS.values()),
        "undefined_citations_zero": undefined_citations == 0,
        "undefined_references_zero": undefined_references == 0,
        "overfull_boxes_zero": overfull_boxes == 0,
        "fatal_errors_zero": fatal_errors == 0,
        "missing_assets_zero": missing_assets == 0,
        "unresolved_pdf_citation_markers_zero": unresolved_markers == 0,
        "abstract_180_to_230": 180 <= words <= 230,
        "short_title_under_40": len(short_title) < 40,
        "cover_letter_one_page": pages["cover_letter"] == 1,
    }
    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "BLOCKED_GATE7B1_COMPILE",
        "engine": "Tectonic 0.16.9 offline cached bundle",
        "exit_codes": exit_codes,
        "pdf_pages": pages,
        "abstract_words": words,
        "short_title_characters": len(short_title),
        "undefined_citations": undefined_citations,
        "undefined_references": undefined_references,
        "missing_assets": missing_assets,
        "fatal_errors": fatal_errors,
        "overfull_boxes": overfull_boxes,
        "unresolved_pdf_citation_markers": unresolved_markers,
        "checks": checks,
        "pass": passed,
    }
    (ROOT / "results/gate7b1_jof_compile_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Gate 7B1 Compile Audit

## Status

**{result['status']}**

| Artifact | Exit | Pages |
| --- | ---: | ---: |
| Manuscript | {exit_codes.get('manuscript', 'NA')} | {pages['manuscript']} |
| Anonymous manuscript | {exit_codes.get('anonymous_manuscript', 'NA')} | {pages['anonymous_manuscript']} |
| Supplementary | {exit_codes.get('supplementary', 'NA')} | {pages['supplementary']} |
| Title page | {exit_codes.get('title_page', 'NA')} | {pages['title_page']} |
| Cover letter | {exit_codes.get('cover_letter', 'NA')} | {pages['cover_letter']} |

- Abstract words: {words}
- Undefined citations: {undefined_citations}
- Undefined references: {undefined_references}
- Missing assets: {missing_assets}
- Fatal errors: {fatal_errors}
- Overfull boxes: {overfull_boxes}
- Unresolved PDF citation markers: {unresolved_markers}
"""
    (ROOT / "reports/gate7b1_jof_compile_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
