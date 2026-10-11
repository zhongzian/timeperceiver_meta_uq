#!/usr/bin/env python3
"""Audit submission assets for local identity and infrastructure leakage."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "paper/submission"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}
PATTERNS = {
    "project_local_path": r"/(?:mnt/data|home)/zza(?:/|\b)",
    "local_username": r"\bzza\b",
    "absolute_local_path": r"(?:^|[\s`'\"(])(?:/home/|/mnt/|/tmp/|/var/tmp/|[A-Za-z]:\\)",
    "personal_email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "github_account_url": r"github\.com/[A-Za-z0-9_.-]+",
    "pid_record": r"\bPID\b\s*(?:[:=]|[0-9])",
    "gpu_uuid": r"GPU-[0-9a-f-]{20,}",
    "internal_queue_path": r"(?:configs/queues/|runtime/(?:queue|locks?)/|scheduler_state)",
    "internal_run_path": r"(?:^|[\s`'\"(])(?:runs/|checkpoints?/)",
    "internal_experiment_timestamp": r"\b20[0-9]{6}_[0-9]{6}\b",
    "hostname_record": r"\bhostname\b\s*[:=]",
}


def generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    violations = []
    scanned = []
    if not SUBMISSION.is_dir():
        violations.append({"path": "paper/submission", "line": 0, "type": "missing_submission_directory", "text": ""})
    else:
        for path in sorted(SUBMISSION.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = str(path.relative_to(ROOT))
            scanned.append(relative)
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                for kind, pattern in PATTERNS.items():
                    if re.search(pattern, line, flags=re.IGNORECASE | re.MULTILINE):
                        violations.append(
                            {
                                "path": relative,
                                "line": line_number,
                                "type": kind,
                                "text": line.strip()[:500],
                            }
                        )

    passed = not violations
    payload = {
        "status": "PASS" if passed else "BLOCKED_ANONYMIZATION",
        "GATE6_ANONYMIZATION_PASS": passed,
        "generated_at": generated_at(),
        "submission_directory": "paper/submission",
        "scanned_text_file_count": len(scanned),
        "anonymization_violation_count": len(violations),
        "patterns": list(PATTERNS),
        "violations": violations,
    }
    (ROOT / "results/gate6_anonymization_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Gate 6 Anonymization Audit",
        "",
        f"**Status: {payload['status']}**",
        "",
        f"- Text assets scanned: {len(scanned)}",
        f"- Anonymization violations: {len(violations)}",
    ]
    if violations:
        report.extend(["", "## Violations", ""])
        report.extend(
            f"- `{item['path']}:{item['line']}` [{item['type']}]: {item['text']}" for item in violations
        )
    (ROOT / "reports/gate6_anonymization_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "violations": len(violations)}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
