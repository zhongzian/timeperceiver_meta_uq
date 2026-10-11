#!/usr/bin/env python3
"""Scan final manuscript text for unsupported or over-broad wording."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}
PATTERNS = {
    "backbone-independent": r"backbone-independent",
    "universal": r"\buniversal\b",
    "universally": r"\buniversally\b",
    "all metrics improve": r"all metrics improve",
    "full second-backbone success": r"full second-backbone success",
    "second-backbone replication positive": r"second-backbone replication positive",
    "point accuracy is preserved across all datasets": r"point accuracy is preserved across all datasets",
    "statistically significant": r"statistically significant",
    "state-of-the-art": r"state-of-the-art",
    "SOTA": r"\bSOTA\b",
    "jointly calibrated": r"jointly calibrated",
    "guaranteed calibration": r"guaranteed calibration",
    "adaptive df is necessary": r"adaptive df is necessary",
}


def generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def targets() -> list[Path]:
    paths = [
        path
        for path in (ROOT / "paper").rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]
    paths.extend(
        [
            ROOT / "reports/final_project_synthesis.md",
            ROOT / "reports/final_claim_evidence_registry.md",
        ]
    )
    return sorted(set(paths))


def main() -> None:
    hits = []
    scanned = []
    for path in targets():
        if not path.is_file():
            hits.append({"path": str(path.relative_to(ROOT)), "line": 0, "phrase": "missing scan target", "text": ""})
            continue
        relative = str(path.relative_to(ROOT))
        scanned.append(relative)
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            for phrase, pattern in PATTERNS.items():
                if re.search(pattern, line, flags=re.IGNORECASE):
                    hits.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "phrase": phrase,
                            "text": line.strip()[:500],
                        }
                    )

    passed = not hits
    payload = {
        "status": "PASS" if passed else "BLOCKED_UNSUPPORTED_WORDING",
        "GATE6_CLAIM_SCAN_PASS": passed,
        "generated_at": generated_at(),
        "scanned_file_count": len(scanned),
        "prohibited_unsupported_wording_count": len(hits),
        "prohibited_patterns": list(PATTERNS),
        "hits": hits,
    }
    result_path = ROOT / "results/gate6_claim_scan.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Gate 6 Claim Scan",
        "",
        f"**Status: {payload['status']}**",
        "",
        f"- Scanned files: {len(scanned)}",
        f"- Prohibited unsupported wording: {len(hits)}",
    ]
    if hits:
        lines.extend(["", "## Hits", ""])
        lines.extend(
            f"- `{hit['path']}:{hit['line']}` [{hit['phrase']}]: {hit['text']}" for hit in hits
        )
    report_path = ROOT / "reports/gate6_claim_scan.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "hits": len(hits)}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
