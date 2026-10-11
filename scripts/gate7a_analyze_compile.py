#!/usr/bin/env python3
"""Classify Gate 7A LaTeX compilation diagnostics and validate the PDF."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "paper" / "manuscript" / "build"
RESULT = ROOT / "results" / "gate7a_compile_validation.json"
REPORT = ROOT / "reports" / "gate7a_compile_validation.md"


def matches(pattern: str, text: str) -> list[str]:
    return sorted(set(match.group(0).strip() for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()

    console_log = BUILD / "compile.log"
    engine_log = BUILD / "main.log"
    text = console_log.read_text(encoding="utf-8", errors="ignore") if console_log.is_file() else ""
    if engine_log.is_file():
        text += "\n" + engine_log.read_text(encoding="utf-8", errors="ignore")

    undefined_citations = matches(r"(?:Citation [`'][^\n]+?undefined|There were undefined citations)", text)
    undefined_references = matches(r"(?:Reference [`'][^\n]+?undefined|There were undefined references)", text)
    missing_figures = matches(r"(?:File [`'][^\n]*\.(?:pdf|png|jpg|jpeg|eps)[^\n]*not found)", text)
    missing_fonts = matches(r"(?:font[^\n]*(?:not found|missing)|missing character)", text)
    fatal_errors = matches(r"(?:Emergency stop|Fatal error|^! LaTeX Error:[^\n]*|error: [^\n]*)", text)
    overfull_boxes = matches(r"Overfull \\[hv]box[^\n]*", text)
    underfull_boxes = matches(r"Underfull \\[hv]box[^\n]*", text)
    latex_warnings = matches(r"(?:LaTeX|Package [^\n]+) Warning:[^\n]*", text)

    pdf_path = BUILD / "manuscript.pdf"
    page_count = 0
    pdfinfo_error = ""
    if pdf_path.is_file():
        proc = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            match = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.MULTILINE)
            page_count = int(match.group(1)) if match else 0
        else:
            pdfinfo_error = proc.stderr.strip()

    passed = all([
        args.exit_code == 0,
        pdf_path.is_file(),
        pdf_path.stat().st_size > 0 if pdf_path.is_file() else False,
        page_count > 0,
        not undefined_citations,
        not undefined_references,
        not missing_figures,
        not fatal_errors,
    ])
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE7A_COMPILE_PASS": passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "compile_exit_code": args.exit_code,
        "pdf_path": str(pdf_path.relative_to(ROOT)),
        "pdf_exists": pdf_path.is_file(),
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.is_file() else 0,
        "total_pages": page_count,
        "undefined_citation_count": len(undefined_citations),
        "undefined_reference_count": len(undefined_references),
        "missing_figure_count": len(missing_figures),
        "missing_font_count": len(missing_fonts),
        "fatal_error_count": len(fatal_errors),
        "overfull_box_count": len(overfull_boxes),
        "underfull_box_count": len(underfull_boxes),
        "latex_warning_count": len(latex_warnings),
        "undefined_citations": undefined_citations,
        "undefined_references": undefined_references,
        "missing_figures": missing_figures,
        "missing_fonts": missing_fonts,
        "fatal_errors": fatal_errors,
        "overfull_boxes": overfull_boxes,
        "underfull_boxes": underfull_boxes,
        "latex_warnings": latex_warnings,
        "pdfinfo_error": pdfinfo_error,
        "compile_log": str(console_log.relative_to(ROOT)),
        "engine_log": str(engine_log.relative_to(ROOT)) if engine_log.is_file() else None,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Gate 7A Compile Validation",
        "",
        f"**Status:** {payload['status']}",
        "",
        f"- Engine: {args.engine}",
        f"- Exit code: {args.exit_code}",
        f"- Pages: {page_count}",
        f"- Undefined citations: {len(undefined_citations)}",
        f"- Undefined references: {len(undefined_references)}",
        f"- Missing figures: {len(missing_figures)}",
        f"- Missing fonts: {len(missing_fonts)}",
        f"- Fatal errors: {len(fatal_errors)}",
        f"- Overfull boxes: {len(overfull_boxes)}",
        f"- Underfull boxes: {len(underfull_boxes)}",
        f"- Other LaTeX warnings: {len(latex_warnings)}",
        f"- PDF: `{payload['pdf_path']}`",
    ]
    if overfull_boxes:
        lines.extend(["", "## Overfull Boxes", ""] + [f"- {item}" for item in overfull_boxes])
    if latex_warnings:
        lines.extend(["", "## Other Warnings", ""] + [f"- {item}" for item in latex_warnings])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "engine": args.engine,
        "exit_code": args.exit_code,
        "pages": page_count,
        "undefined_citations": len(undefined_citations),
        "undefined_references": len(undefined_references),
        "overfull_boxes": len(overfull_boxes),
        "fatal_errors": len(fatal_errors),
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
