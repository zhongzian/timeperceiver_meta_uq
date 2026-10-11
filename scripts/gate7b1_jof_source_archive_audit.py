#!/usr/bin/env python3
"""Independently extract and compile both packaged LaTeX source archives."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"
PACKAGE = VENUE / "final_submission_package"
TECTONIC = Path("/mnt/data/zza/anaconda3/pkgs/tectonic-0.16.9-ha39f199_0/bin/tectonic")
ICU = "/mnt/data/zza/anaconda3/pkgs/icu-78.3-h84d19a5_0/lib"


def safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def compile_source(directory: Path, source: str) -> tuple[int, str, Path]:
    build = directory / "build"
    build.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ICU + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    completed = subprocess.run(
        [str(TECTONIC), "--only-cached", "--keep-intermediates", "--keep-logs", "--outdir", "build", source],
        cwd=directory,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    pdf = build / f"{Path(source).stem}.pdf"
    return completed.returncode, completed.stdout + completed.stderr, pdf


def main() -> None:
    main_archive = PACKAGE / "manuscript/manuscript_source.tar.gz"
    supplement_archive = PACKAGE / "supplementary/supplementary_source.tar.gz"
    with tempfile.TemporaryDirectory(prefix="gate7b1_source_audit_") as temporary:
        destination = Path(temporary)
        safe_extract(main_archive, destination)
        safe_extract(supplement_archive, destination)
        main_exit, main_log, main_pdf = compile_source(destination / "manuscript", "main.tex")
        supplement_exit, supplement_log, supplement_pdf = compile_source(destination / "supplementary", "supplementary.tex")
        combined = main_log + "\n" + supplement_log
        overfull = len(re.findall(r"Overfull \\hbox|Overfull \\vbox", combined))
        undefined = len(re.findall(r"citation.*undefined|reference.*undefined|undefined citations|undefined references", combined, flags=re.IGNORECASE))
        fatal = len(re.findall(r"fatal error|aborted|emergency stop|error:", combined, flags=re.IGNORECASE))
        checks = {
            "main_exit_zero": main_exit == 0,
            "supplement_exit_zero": supplement_exit == 0,
            "main_pdf_nonempty": main_pdf.is_file() and main_pdf.stat().st_size > 0,
            "supplement_pdf_nonempty": supplement_pdf.is_file() and supplement_pdf.stat().st_size > 0,
            "undefined_zero": undefined == 0,
            "fatal_zero": fatal == 0,
            "overfull_zero": overfull == 0,
        }
    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAILED",
        "engine": "Tectonic 0.16.9 offline cached bundle",
        "main_source_archive": "manuscript/manuscript_source.tar.gz",
        "supplement_source_archive": "supplementary/supplementary_source.tar.gz",
        "main_compile_exit": main_exit,
        "supplement_compile_exit": supplement_exit,
        "undefined_citations_or_references": undefined,
        "fatal_errors": fatal,
        "overfull_boxes": overfull,
        "checks": checks,
        "pass": passed,
    }
    (ROOT / "results/gate7b1_jof_source_archive_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = f"""# Gate 7B1 Source Archive Audit

## Status

**{result['status']}**

- Manuscript archive compile exit: {main_exit}
- Supplement archive compile exit: {supplement_exit}
- Undefined citations or references: {undefined}
- Fatal errors: {fatal}
- Overfull boxes: {overfull}

Both archives were extracted together in a fresh temporary directory and compiled with the verified offline Tectonic bundle. Temporary absolute paths and compile logs are not included in the submission package.
"""
    (ROOT / "reports/gate7b1_jof_source_archive_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
