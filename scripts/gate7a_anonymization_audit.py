#!/usr/bin/env python3
"""Audit Gate 7A manuscript and frozen submission assets for identity leakage."""

from __future__ import annotations

import getpass
import json
import re
import socket
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "paper" / "manuscript", ROOT / "paper" / "submission"]
RESULT = ROOT / "results" / "gate7a_anonymization_audit.json"
REPORT = ROOT / "reports" / "gate7a_anonymization_audit.md"


def text_files() -> list[Path]:
    files = []
    for target in TARGETS:
        for path in target.rglob("*"):
            if not path.is_file() or "build" in path.parts:
                continue
            if path.suffix.lower() in {".tex", ".bib", ".md", ".json", ".csv", ".txt", ".yaml", ".yml"}:
                files.append(path)
    return sorted(files)


def redact(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", "<email>", text)
    text = re.sub(r"(?:/[^\s:'\"]+)+", "<path>", text)
    return text[:180]


def main() -> int:
    username = getpass.getuser()
    hostname = socket.gethostname()
    patterns = {
        "absolute_local_paths": re.compile(r"/(?:mnt|home|Users|tmp)/[^\s}\],\"']+", re.IGNORECASE),
        "usernames": re.compile(rf"\b{re.escape(username)}\b", re.IGNORECASE) if username else re.compile(r"(?!)"),
        "hostnames": re.compile(rf"\b{re.escape(hostname)}\b", re.IGNORECASE) if hostname else re.compile(r"(?!)"),
        "pids": re.compile(r"\bpid\s*[:= ]\s*\d+\b", re.IGNORECASE),
        "gpu_uuids": re.compile(r"GPU-[0-9a-f-]{16,}", re.IGNORECASE),
        "private_urls": re.compile(r"(?:https?://(?:localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)|git@|ssh://|file://)", re.IGNORECASE),
        "personal_emails": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "internal_timestamps": re.compile(r"\b20\d{6}[_-]\d{6}\b"),
        "absolute_checkpoint_paths": re.compile(r"(?:/(?:mnt|home)/[^\s]+|runs/[^\s]+)/(?:checkpoints?|ckpt)/[^\s]+", re.IGNORECASE),
        "queue_paths": re.compile(r"(?:configs/queues|runtime/.+scheduler|scheduler_state)", re.IGNORECASE),
    }
    violations: dict[str, list[dict[str, object]]] = {name: [] for name in patterns}
    files = text_files()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), 1):
            for name, pattern in patterns.items():
                if pattern.search(line):
                    violations[name].append({
                        "file": str(path.relative_to(ROOT)),
                        "line": line_number,
                        "context": redact(line),
                    })
    counts = {name: len(items) for name, items in violations.items()}
    total = sum(counts.values())
    passed = total == 0
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE7A_ANONYMIZATION_PASS": passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audited_roots": [str(path.relative_to(ROOT)) for path in TARGETS],
        "audited_text_file_count": len(files),
        "violation_count": total,
        "counts": counts,
        "violations": violations,
        "allowed_public_metadata": ["public upstream URLs", "DOIs", "software versions", "dataset names", "publication authors"],
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Gate 7A Anonymization Audit",
        "",
        f"**Status:** {payload['status']}",
        "",
        f"- Audited text files: {len(files)}",
        f"- Total violations: {total}",
    ]
    lines.extend(f"- {name.replace('_', ' ').title()}: {count}" for name, count in counts.items())
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], **counts}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
