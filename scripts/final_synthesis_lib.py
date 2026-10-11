#!/usr/bin/env python3
"""Shared, read-only source helpers for the final evidence synthesis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python")
UPSTREAM_COMMIT = "7e30cc07b51c709f408409fd60a34c81ae8990be"
UPSTREAM_ROOT = ROOT / "upstream" / "TimePerceiver"

REGISTRY_FIELDS = [
    "evidence_id",
    "stage",
    "gate",
    "dataset",
    "missing_type",
    "missing_rate",
    "experiment_seed",
    "seed_scope",
    "variant",
    "interval_method",
    "coverage_level",
    "is_formal",
    "is_smoke",
    "is_reused",
    "is_valid",
    "paper_evidence",
    "invalid_reason",
    "legacy_unmasked_invalid",
    "experiment_id",
    "run_dir",
    "source_report",
    "source_row_or_key",
    "checkpoint_path",
    "checkpoint_sha256",
    "source_checkpoint_path",
    "source_checkpoint_sha256",
    "validation_mask_path",
    "validation_mask_sha256",
    "calibration_mask_path",
    "calibration_mask_sha256",
    "test_mask_path",
    "test_mask_sha256",
    "input_mask_applied",
    "mask_application_verified",
    "batch_y_modified",
    "freeze_backbone",
    "test_used_for_model_selection",
    "calibration_used_for_model_selection",
    "training_protocol_hash",
    "metric_protocol_hash",
    "mse",
    "mae",
    "rmse",
    "nll",
    "crps",
    "picp",
    "mpiw",
    "coverage_error",
    "absolute_coverage_error",
    "interval_score",
    "mean_df",
    "median_df",
    "min_df",
    "max_df",
    "trainable_parameter_count",
    "total_parameter_count",
    "parameter_overhead_ratio",
    "best_epoch",
    "validation_metric",
    "calibration_picp",
    "test_picp",
    "coverage_shift",
    "formal_qhat",
    "delta_mpiw",
    "delta_absolute_coverage_error",
    "oracle_signed_delta",
    "counterfactual_only",
    "uses_test_labels",
    "status",
    "formal_decision",
    "notes",
]

PROTECTED_FORMAL_SOURCES = [
    "results/gate4a_formal_decision.json",
    "results/gate4b_expanded_formal_decision.json",
    "results/gate4b_cross_dataset_formal_decision.json",
    "results/gate4b_multiseed_formal_decision.json",
    "results/gate4c_formal_decision.json",
    "reports/gate4b_multiseed_paired_deltas.csv",
    "reports/gate4b_multiseed_dataset_summary.csv",
    "reports/gate4c_interval_metrics.csv",
    "reports/gate4c_conditioned_bin_metrics.csv",
]


def generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def resolve_path(value: Any) -> Path | None:
    if value in (None, "", "NA"):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def relative_path(value: Any) -> str:
    path = resolve_path(value)
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def sha256_file(value: Any) -> str:
    path = resolve_path(value)
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(value: Any) -> Any:
    path = resolve_path(value)
    if path is None:
        raise FileNotFoundError(value)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(value: Any) -> list[dict[str, Any]]:
    path = resolve_path(value)
    if path is None:
        raise FileNotFoundError(value)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row_number, row in enumerate(rows, start=2):
        row["_row_number"] = row_number
    return rows


def as_float(value: Any) -> float | None:
    if value in (None, "", "NA", "None", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return int(number) if number is not None else None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, "", "NA", "None"):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def run_status(run_dir: Any) -> dict[str, Any]:
    path = resolve_path(run_dir)
    if path is None or not (path / "status.json").is_file():
        return {}
    return read_json(path / "status.json")


def run_metrics(run_dir: Any) -> dict[str, Any]:
    path = resolve_path(run_dir)
    if path is None or not (path / "metrics.json").is_file():
        return {}
    return read_json(path / "metrics.json")


def best_checkpoint(run_dir: Any) -> str:
    path = resolve_path(run_dir)
    if path is None:
        return ""
    candidates = [
        path / "checkpoints" / "best_probability_head.pt",
        path / "checkpoints" / "best_prob_head.pt",
        path / "checkpoints" / "best_adapter.pt",
        path / "checkpoints" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return relative_path(candidate)
    files = sorted((path / "checkpoints").glob("**/checkpoint.pth")) if (path / "checkpoints").is_dir() else []
    return relative_path(files[0]) if files else ""


def protocol_hash(run_dir: Any, fallback_file: str) -> str:
    path = resolve_path(run_dir)
    if path is not None and (path / "resolved_config.yaml").is_file():
        return sha256_file(path / "resolved_config.yaml")
    return sha256_file(fallback_file)


def blank_registry_row() -> dict[str, Any]:
    return {field: "" for field in REGISTRY_FIELDS}


def normalize_registry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = blank_registry_row()
    output.update(row)
    return output


def evidence_id(row: Mapping[str, Any]) -> str:
    key = "|".join(
        str(row.get(field, "NA"))
        for field in (
            "stage",
            "dataset",
            "missing_type",
            "missing_rate",
            "experiment_seed",
            "variant",
            "interval_method",
            "coverage_level",
        )
    )
    return "EV-" + hash_text(key)[:16]


def write_json(path: Any, payload: Any) -> None:
    target = resolve_path(path)
    if target is None:
        raise ValueError(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Any, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    target = resolve_path(path)
    if target is None:
        raise ValueError(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def display(value: Any, digits: int = 6) -> str:
    if value in (None, "", "NA"):
        return "NA"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], labels: Mapping[str, str] | None = None) -> str:
    labels = labels or {}
    header = [labels.get(field, field) for field in fields]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = [display(row.get(field, "")).replace("|", "\\|") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def latex_escape(value: Any) -> str:
    text = display(value)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], labels: Mapping[str, str] | None = None) -> str:
    labels = labels or {}
    columns = "l" + "r" * max(0, len(fields) - 1)
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\hline"]
    lines.append(" & ".join(latex_escape(labels.get(field, field)) for field in fields) + r" \\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(field, "")) for field in fields) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_table_bundle(base: Any, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], title: str, note: str = "", labels: Mapping[str, str] | None = None) -> None:
    target = resolve_path(base)
    if target is None:
        raise ValueError(base)
    write_csv(target.with_suffix(".csv"), rows, fields)
    md = f"# {title}\n\n" + markdown_table(rows, fields, labels) + "\n"
    if note:
        md += f"\nNote: {note}\n"
    target.with_suffix(".md").write_text(md, encoding="utf-8")
    tex = latex_table(rows, fields, labels)
    if note:
        tex += "% Note: " + note.replace("\n", " ") + "\n"
    target.with_suffix(".tex").write_text(tex, encoding="utf-8")


def upstream_state() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    untracked = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).splitlines()
    return {
        "commit": commit,
        "tracked_diff_empty": not bool(tracked),
        "tracked_status": tracked.splitlines(),
        "untracked": [line for line in untracked if line.startswith("??")],
    }


def source_manifest(paths: Iterable[Any], stage_map: Mapping[str, tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    stage_map = stage_map or {}
    rows = []
    for value in sorted({relative_path(item) for item in paths if item}):
        path = resolve_path(value)
        if path is None or not path.is_file():
            continue
        stat = path.stat()
        stage, purpose = stage_map.get(value, ("cross_gate", "final synthesis source"))
        rows.append(
            {
                "relative_path": value,
                "sha256": sha256_file(path),
                "file_size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc).astimezone().isoformat(timespec="seconds"),
                "stage": stage,
                "purpose": purpose,
                "read_only_source": True,
            }
        )
    return rows


def exact_close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    a, b = as_float(left), as_float(right)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def sample_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sample_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
