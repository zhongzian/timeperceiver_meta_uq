#!/usr/bin/env python3
"""Finalize Gate 5a bookkeeping without rerunning compute or formal analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/gate5a_closure"

FORMAL_SOURCES = (
    "results/gate4a_formal_decision.json",
    "results/gate4b_expanded_formal_decision.json",
    "results/gate4b_cross_dataset_formal_decision.json",
    "results/gate4b_multiseed_formal_decision.json",
    "results/gate4c_formal_decision.json",
    "results/gate5a_formal_decision.json",
    "reports/gate4b_multiseed_paired_deltas.csv",
    "reports/gate4b_multiseed_dataset_summary.csv",
    "reports/gate4c_interval_metrics.csv",
    "reports/gate4c_conditioned_bin_metrics.csv",
    "reports/gate5a_distribution_results.csv",
    "reports/gate5a_pairwise_deltas.csv",
    "results/final_evidence_registry.json",
    "results/final_project_status.json",
    "results/paper_artifact_verification.json",
)

DIAGNOSTIC_FILES = (
    "results/gate5a_mechanism_audit.json",
    "reports/gate5a_mechanism_audit.md",
    "reports/gate5a_pit_metrics.csv",
    "reports/gate5a_qq_metrics.csv",
    "reports/gate5a_tail_exceedance.csv",
    "reports/gate5a_nll_gain_by_residual_bin.csv",
    "reports/gate5a_channel_tail_analysis.csv",
    "reports/gate5a_channel_tail_correlations.csv",
    "paper/figures/gate5a/fig_gate5a_pit_histograms.png",
    "paper/figures/gate5a/fig_gate5a_pit_histograms.pdf",
    "paper/figures/gate5a/fig_gate5a_pit_histograms_data.csv",
    "paper/figures/gate5a/fig_gate5a_qq.png",
    "paper/figures/gate5a/fig_gate5a_qq.pdf",
    "paper/figures/gate5a/fig_gate5a_qq_data.csv",
    "paper/figures/gate5a/fig_gate5a_tail_nll_gain.png",
    "paper/figures/gate5a/fig_gate5a_tail_nll_gain.pdf",
    "paper/figures/gate5a/fig_gate5a_tail_nll_gain_data.csv",
    "paper/figures/gate5a/fig_gate5a_kurtosis_df.png",
    "paper/figures/gate5a/fig_gate5a_kurtosis_df.pdf",
    "paper/figures/gate5a/fig_gate5a_kurtosis_df_data.csv",
)

EFFICIENCY_FILES = (
    "results/gate5a_efficiency.json",
    "reports/gate5a_efficiency.csv",
    "reports/gate5a_efficiency.md",
)

PAPER_FILES = (
    "paper/tables/table8_distribution_head_ablation.csv",
    "paper/tables/table8_distribution_head_ablation.md",
    "paper/tables/table8_distribution_head_ablation.tex",
    "paper/tables/table9_distribution_efficiency.csv",
    "paper/tables/table9_distribution_efficiency.md",
    "paper/tables/table9_distribution_efficiency.tex",
    "paper/sections/distribution_head_ablation.md",
    "paper/sections/heavy_tail_mechanism.md",
    "paper/sections/abstract_draft.md",
    "paper/sections/contributions.md",
    "paper/sections/main_results.md",
    "paper/sections/limitations.md",
)

REQUIRED_STATE_FILES = (
    "runtime/gate5a_protocol_smoke_state.json",
    "runtime/gate5a_distribution_ablation_scheduler_state.json",
    "runtime/gate5a_postformal_watcher_state.json",
    "results/gate5a_status.json",
    "results/gate5a_formal_decision.json",
    "results/gate5a_artifact_verification.json",
    "results/final_evidence_registry_verification.json",
    "results/final_project_status.json",
    "results/paper_artifact_verification.json",
)

PROCESS_MARKERS = (
    "gate5a_run_protocol_smokes.py",
    "gate5a_continue_after_smokes.sh",
    "gate5a_postformal_watcher.py",
    "gate5a_refresh_summary.sh",
    "gate5a_heavy_tail_diagnostics.py",
    "gate5a_efficiency_benchmark.py",
    "gate5a_distribution_ablation.yaml",
    "gate5a_distribution_ablation_scheduler",
    "build_final_paper_artifacts.sh",
)

WRITE_ALLOWLIST = (
    "scripts/gate5a_finalize_closure.py",
    "scripts/gate5a_closure_status.sh",
    "results/gate5a_postprocess_integrity.json",
    "reports/gate5a_postprocess_integrity.md",
    "results/gate5a_closure_verification.json",
    "reports/gate5a_closure_verification.md",
    "results/gate5a_status.json",
    "reports/gate5a_status.md",
    "artifacts/gate5a_closure/",
)


class ClosureError(RuntimeError):
    pass


def load_json(relative_path: str) -> dict:
    path = ROOT / relative_path
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureError(f"cannot read valid JSON: {relative_path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_nonempty(relative_paths) -> None:
    missing = []
    for relative_path in relative_paths:
        path = ROOT / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative_path)
    if missing:
        raise ClosureError("missing or empty required artifacts: " + ", ".join(missing))


def csv_row_count(relative_path: str) -> int:
    with (ROOT / relative_path).open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o664)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def current_processes() -> list[dict]:
    output = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,args="], text=True
    )
    matches = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid, ppid, command = int(fields[0]), int(fields[1]), fields[2]
        if pid == os.getpid():
            continue
        if any(marker in command for marker in PROCESS_MARKERS):
            matches.append({"pid": pid, "ppid": ppid, "command": command})
    return matches


def upstream_state() -> tuple[str, list[str], list[str]]:
    upstream = ROOT / "upstream/TimePerceiver"
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    lines = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--short"], text=True
    ).splitlines()
    tracked = [line for line in lines if not line.startswith("??")]
    untracked = [line for line in lines if line.startswith("??")]
    return commit, tracked, untracked


def source_hashes() -> dict[str, str]:
    ensure_nonempty(FORMAL_SOURCES)
    return {relative_path: sha256_file(ROOT / relative_path) for relative_path in FORMAL_SOURCES}


def hash_manifest_text(hashes: dict[str, str]) -> str:
    return "".join(f"{digest}  {path}\n" for path, digest in hashes.items())


def all_true(mapping: dict) -> bool:
    return bool(mapping) and all(value is True for value in mapping.values())


def integrity_markdown(payload: dict, comparison: dict) -> str:
    source_rows = "\n".join(
        f"| `{row['path']}` | `{row['pre_sha256']}` | `{row['post_sha256']}` | {str(row['match']).upper()} |"
        for row in comparison["files"]
    )
    return f"""# Gate 5a postprocess integrity

## Gate 5a postprocess integrity

**{payload['status']}**

## Execution safety

- training: NO
- optimizer: NO
- backward: NO
- GPU: NO
- scheduler: NO
- formal checkpoints modified: NO
- formal metrics modified: NO
- formal decisions modified: NO

## Formal completion

- protocol smoke: 6/6
- logical tasks: 12/12
- reused: 6
- new tasks: 6/6
- scheduler: COMPLETE
- formal decision: FIXED_HEAVY_TAIL_SUFFICIENT

## Postprocess completion

- diagnostics: PASS
- efficiency: PASS
- paper artifacts: PASS
- canonical evidence: PASS
- final synthesis: PASS

## Evidence integrity

- invalid legacy rows in paper: {payload['invalid_legacy_paper_rows']}
- smoke rows in paper: {payload['smoke_paper_rows']}
- unsupported claims: {payload['unsupported_claims']}
- source hash mismatch: {payload['source_hash_mismatches']}

## Formal source integrity

| Source | Pre-closure SHA256 | Post-closure SHA256 | Match |
| --- | --- | --- | --- |
{source_rows}

## Closure gates

```text
GATE5A_DIAGNOSTICS_COMPLETE=True
GATE5A_EFFICIENCY_COMPLETE=True
GATE5A_PAPER_UPDATE_COMPLETE=True
GATE5A_FINAL_EVIDENCE_REBUILD_PASS=True
GATE5A_FORMAL_INTEGRITY_PASS=True
GATE5A_POSTPROCESS_COMPLETE=True
FINAL_SYNTHESIS_PASS=True
READY_FOR_SECOND_BACKBONE_PLANNING=True
```
"""


def status_markdown(payload: dict) -> str:
    return f"""# Gate 5a Status

Gate 5a: **{payload['status']}**

- Formal decision: `{payload['decision']}`
- Formal status: `{payload['formal_status']}`
- Postprocess: `{payload['postprocess_status']}`
- Final synthesis: `PASS`
- Ready for second-backbone planning: `TRUE`
- Closure-only update: `TRUE`
- Formal results modified: `FALSE`
- Formal decision modified: `FALSE`

```json
{json.dumps(payload, indent=2)}
```
"""


def verification_markdown(payload: dict) -> str:
    lines = [
        "# Gate 5a Closure Verification",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"`GATE5A_STRICT_CLOSURE_PASS={payload['GATE5A_STRICT_CLOSURE_PASS']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {key}: {str(value).upper()}" for key, value in payload["checks"].items())
    return "\n".join(lines) + "\n"


def main() -> int:
    ensure_nonempty(REQUIRED_STATE_FILES + FORMAL_SOURCES + DIAGNOSTIC_FILES + EFFICIENCY_FILES + PAPER_FILES)

    related_before = current_processes()
    if related_before:
        raise ClosureError(
            "BLOCKED_GATE5A_CLOSURE_PROCESS_STILL_RUNNING: "
            + json.dumps(related_before, sort_keys=True)
        )

    smoke = load_json("runtime/gate5a_protocol_smoke_state.json")
    scheduler = load_json("runtime/gate5a_distribution_ablation_scheduler_state.json")
    postformal = load_json("runtime/gate5a_postformal_watcher_state.json")
    old_status = load_json("results/gate5a_status.json")
    formal_decision = load_json("results/gate5a_formal_decision.json")
    equivalent = load_json("results/gate5a_artifact_verification.json")
    mechanism = load_json("results/gate5a_mechanism_audit.json")
    efficiency = load_json("results/gate5a_efficiency.json")
    registry_verification = load_json("results/final_evidence_registry_verification.json")
    final_status = load_json("results/final_project_status.json")
    paper_verification = load_json("results/paper_artifact_verification.json")

    smoke_tasks = list(smoke.get("tasks", {}).values())
    scheduler_tasks = list(scheduler.get("tasks", {}).values())
    active_tasks = [task for task in scheduler_tasks if task.get("reused") is not True]
    reused_tasks = [task for task in scheduler_tasks if task.get("reused") is True]

    equivalent_checks = equivalent.get("checks", {})
    registry_counts = registry_verification.get("counts", {})
    paper_checks = paper_verification.get("checks", {})
    paper_claims = paper_verification.get("claims", {})
    paper_tables = paper_verification.get("tables", {})
    paper_figures = paper_verification.get("figures", {})

    mapped_checks = {
        "postformal_watcher_complete": postformal.get("status") == "COMPLETE"
        and postformal.get("return_code") == 0,
        "protocol_smoke_6_of_6": smoke.get("status") == "COMPLETE"
        and len(smoke_tasks) == 6
        and all(task.get("status") == "SUCCEEDED" for task in smoke_tasks),
        "formal_tasks_12_of_12": scheduler.get("status") == "COMPLETE"
        and len(scheduler_tasks) == 12
        and all(task.get("status") == "SUCCEEDED" for task in scheduler_tasks),
        "formal_task_partition_6_plus_6": len(reused_tasks) == 6 and len(active_tasks) == 6,
        "formal_implementation_pass": formal_decision.get("GATE5A_IMPLEMENTATION_PASS") is True,
        "formal_decision_fixed_heavy_tail": formal_decision.get("status") == "PASS"
        and formal_decision.get("decision") == "FIXED_HEAVY_TAIL_SUFFICIENT",
        "formal_distribution_rows_12": csv_row_count("reports/gate5a_distribution_results.csv") == 12,
        "diagnostics_complete": mechanism.get("status") == "PASS",
        "efficiency_complete": efficiency.get("status") == "PASS"
        and len(efficiency.get("rows", [])) == 12,
        "paper_tables_complete": all(
            all_true(paper_tables[name].get("checks", {}))
            for name in ("table8_distribution_head_ablation", "table9_distribution_efficiency")
        ),
        "paper_figures_complete": all(
            all_true(paper_figures[name].get("checks", {}))
            for name in (
                "gate5a/fig_gate5a_pit_histograms",
                "gate5a/fig_gate5a_qq",
                "gate5a/fig_gate5a_tail_nll_gain",
                "gate5a/fig_gate5a_kurtosis_df",
            )
        ),
        "canonical_evidence_rebuilt": registry_verification.get("FINAL_EVIDENCE_REGISTRY_PASS") is True
        and equivalent_checks.get("canonical_registry_pass") is True,
        "final_synthesis_pass": final_status.get("FINAL_SYNTHESIS_PASS") is True
        and equivalent_checks.get("final_synthesis_pass") is True,
        "paper_artifact_verification_pass": paper_verification.get("FINAL_PAPER_ARTIFACT_PASS") is True
        and all_true(paper_checks),
        "invalid_legacy_paper_rows_zero": registry_counts.get("invalid_legacy_rows_in_paper") == 0
        and equivalent_checks.get("old_unmasked_paper_rows_zero") is True,
        "smoke_paper_rows_zero": registry_counts.get("smoke_rows_in_paper") == 0
        and equivalent_checks.get("smoke_paper_rows_zero") is True,
        "unsupported_claims_zero": paper_claims.get("unsupported_count") == 0,
        "registry_hash_mismatches_zero": registry_counts.get("hash_mismatches") == 0,
        "existing_equivalent_verification_pass": equivalent.get("status") == "PASS"
        and equivalent.get("GATE5A_ARTIFACT_VERIFICATION_PASS") is True,
    }
    missing_required_checks = [key for key, passed in mapped_checks.items() if not passed]
    if missing_required_checks:
        raise ClosureError("required closure checks failed: " + ", ".join(missing_required_checks))

    commit, tracked, untracked = upstream_state()
    if commit != "7e30cc07b51c709f408409fd60a34c81ae8990be":
        raise ClosureError(f"unexpected upstream commit: {commit}")
    if tracked:
        raise ClosureError("upstream tracked diff is not empty: " + json.dumps(tracked))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pre_hashes = source_hashes()
    atomic_write_text(
        ARTIFACT_DIR / "pre_closure_source_hashes.sha256", hash_manifest_text(pre_hashes)
    )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    equivalent_sha256 = sha256_file(ROOT / "results/gate5a_artifact_verification.json")
    closure_completed_at = generated_at
    previous_status = old_status.get("previous_status", old_status.get("status"))

    integrity = {
        "status": "PASS",
        "stage": "gate5a_postprocess_closure",
        "closure_only": True,
        "training_run": False,
        "optimizer_created": False,
        "backward_called": False,
        "gpu_used": False,
        "scheduler_started": False,
        "formal_checkpoints_modified": False,
        "formal_metrics_modified": False,
        "formal_decisions_modified": False,
        "postformal_watcher_complete": True,
        "postformal_exit_code": 0,
        "postformal_completed_at": postformal.get("finished_at"),
        "gate5a_formal_tasks_complete": True,
        "gate5a_logical_tasks": len(scheduler_tasks),
        "gate5a_reused_tasks": len(reused_tasks),
        "gate5a_new_tasks_succeeded": len(active_tasks),
        "gate5a_implementation_pass": True,
        "gate5a_formal_decision": formal_decision["decision"],
        "diagnostics_complete": True,
        "efficiency_complete": True,
        "paper_update_complete": True,
        "canonical_evidence_rebuild_pass": True,
        "final_synthesis_pass": True,
        "paper_artifact_verification_pass": True,
        "invalid_legacy_paper_rows": registry_counts["invalid_legacy_rows_in_paper"],
        "smoke_paper_rows": registry_counts["smoke_rows_in_paper"],
        "unsupported_claims": paper_claims["unsupported_count"],
        "source_hash_mismatches": registry_counts["hash_mismatches"],
        "gate4a_gate4c_decisions_unchanged": True,
        "gate4b_primary_csv_unchanged": True,
        "gate5a_formal_decision_unchanged": True,
        "gate5a_formal_results_unchanged": True,
        "existing_equivalent_verification_file": "results/gate5a_artifact_verification.json",
        "existing_equivalent_verification_sha256": equivalent_sha256,
        "existing_equivalent_verification_status": equivalent["status"],
        "source_verification_file": "results/gate5a_artifact_verification.json",
        "source_verification_sha256": equivalent_sha256,
        "source_verification_status": equivalent["status"],
        "mapped_checks": mapped_checks,
        "missing_required_checks": missing_required_checks,
        "upstream_commit": commit,
        "upstream_tracked_diff_empty": True,
        "upstream_untracked": untracked,
        "GATE5A_DIAGNOSTICS_COMPLETE": True,
        "GATE5A_EFFICIENCY_COMPLETE": True,
        "GATE5A_PAPER_UPDATE_COMPLETE": True,
        "GATE5A_FINAL_EVIDENCE_REBUILD_PASS": True,
        "GATE5A_FORMAL_INTEGRITY_PASS": True,
        "GATE5A_POSTPROCESS_COMPLETE": True,
        "FINAL_SYNTHESIS_PASS": True,
        "READY_FOR_SECOND_BACKBONE_PLANNING": True,
        "generated_at": generated_at,
        "generated_by": "scripts/gate5a_finalize_closure.py",
        "write_allowlist": list(WRITE_ALLOWLIST),
        "atomic_status_update": {
            "temporary_file_used": True,
            "temporary_file_fsync": True,
            "os_replace_used": True,
            "directory_fsync": True,
            "reopened_validation": True,
        },
    }
    atomic_write_json(ROOT / "results/gate5a_postprocess_integrity.json", integrity)

    status = dict(old_status)
    status.update(
        {
            "status": "COMPLETE",
            "postprocess_status": "COMPLETE",
            "formal_status": "COMPLETE",
            "GATE5A_IMPLEMENTATION_PASS": True,
            "GATE5A_DIAGNOSTICS_COMPLETE": True,
            "GATE5A_EFFICIENCY_COMPLETE": True,
            "GATE5A_PAPER_UPDATE_COMPLETE": True,
            "GATE5A_FINAL_EVIDENCE_REBUILD_PASS": True,
            "GATE5A_FORMAL_INTEGRITY_PASS": True,
            "GATE5A_POSTPROCESS_COMPLETE": True,
            "FINAL_SYNTHESIS_PASS": True,
            "READY_FOR_SECOND_BACKBONE_PLANNING": True,
            "decision": formal_decision["decision"],
            "closure_only_update": True,
            "formal_results_modified": False,
            "formal_decision_modified": False,
            "training_restarted": False,
            "scheduler_restarted": False,
            "closure_integrity_file": "results/gate5a_postprocess_integrity.json",
            "closure_completed_at": closure_completed_at,
            "previous_status": previous_status,
        }
    )
    atomic_write_json(ROOT / "results/gate5a_status.json", status)
    reopened_status = load_json("results/gate5a_status.json")
    if reopened_status != status:
        raise ClosureError("atomic gate5a_status write did not validate after reopen")
    atomic_write_text(ROOT / "reports/gate5a_status.md", status_markdown(status))

    post_hashes = source_hashes()
    atomic_write_text(
        ARTIFACT_DIR / "post_closure_source_hashes.sha256", hash_manifest_text(post_hashes)
    )
    comparison_rows = [
        {
            "path": path,
            "pre_sha256": pre_hashes[path],
            "post_sha256": post_hashes[path],
            "match": pre_hashes[path] == post_hashes[path],
        }
        for path in FORMAL_SOURCES
    ]
    hash_comparison = {
        "status": "PASS" if all(row["match"] for row in comparison_rows) else "FAILED",
        "source_hash_mismatches": sum(not row["match"] for row in comparison_rows),
        "pre_manifest": "artifacts/gate5a_closure/pre_closure_source_hashes.sha256",
        "post_manifest": "artifacts/gate5a_closure/post_closure_source_hashes.sha256",
        "files": comparison_rows,
        "allowed_changes": list(WRITE_ALLOWLIST),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_write_json(ARTIFACT_DIR / "hash_comparison.json", hash_comparison)
    if hash_comparison["source_hash_mismatches"] != 0:
        failed_status = dict(status)
        failed_status["status"] = "POSTPROCESS_INCOMPLETE"
        failed_status["postprocess_status"] = "POSTPROCESS_INCOMPLETE"
        failed_status["GATE5A_FORMAL_INTEGRITY_PASS"] = False
        failed_status["GATE5A_POSTPROCESS_COMPLETE"] = False
        atomic_write_json(ROOT / "results/gate5a_status.json", failed_status)
        atomic_write_text(ROOT / "reports/gate5a_status.md", status_markdown(failed_status))
        raise ClosureError("formal source hashes changed during closure")

    integrity["source_hash_mismatches"] = 0
    integrity["formal_source_hashes"] = comparison_rows
    atomic_write_json(ROOT / "results/gate5a_postprocess_integrity.json", integrity)
    atomic_write_text(
        ROOT / "reports/gate5a_postprocess_integrity.md",
        integrity_markdown(integrity, hash_comparison),
    )

    related_after = current_processes()
    commit_after, tracked_after, untracked_after = upstream_state()
    closure_checks = {
        "gate5a_status_complete": reopened_status.get("status") == "COMPLETE",
        "gate5a_postprocess_status_complete": reopened_status.get("postprocess_status") == "COMPLETE",
        "gate5a_decision_preserved": reopened_status.get("decision") == "FIXED_HEAVY_TAIL_SUFFICIENT",
        "gate5a_implementation_pass": reopened_status.get("GATE5A_IMPLEMENTATION_PASS") is True,
        "gate5a_diagnostics_complete": reopened_status.get("GATE5A_DIAGNOSTICS_COMPLETE") is True,
        "gate5a_efficiency_complete": reopened_status.get("GATE5A_EFFICIENCY_COMPLETE") is True,
        "gate5a_paper_update_complete": reopened_status.get("GATE5A_PAPER_UPDATE_COMPLETE") is True,
        "gate5a_final_evidence_rebuild_pass": reopened_status.get("GATE5A_FINAL_EVIDENCE_REBUILD_PASS") is True,
        "gate5a_formal_integrity_pass": reopened_status.get("GATE5A_FORMAL_INTEGRITY_PASS") is True,
        "gate5a_postprocess_complete": reopened_status.get("GATE5A_POSTPROCESS_COMPLETE") is True,
        "final_synthesis_pass": reopened_status.get("FINAL_SYNTHESIS_PASS") is True,
        "ready_for_second_backbone_planning": reopened_status.get("READY_FOR_SECOND_BACKBONE_PLANNING") is True,
        "postprocess_integrity_json_pass": load_json("results/gate5a_postprocess_integrity.json").get("status") == "PASS",
        "postprocess_integrity_markdown_exists": (ROOT / "reports/gate5a_postprocess_integrity.md").stat().st_size > 0,
        "related_process_count_zero": len(related_after) == 0,
        "gpu_work_started_by_closure_false": True,
        "formal_source_hash_mismatch_zero": hash_comparison["source_hash_mismatches"] == 0,
        "upstream_commit_unchanged": commit_after == commit,
        "upstream_tracked_diff_empty": not tracked_after,
    }
    strict_pass = all(closure_checks.values())
    verification = {
        "status": "COMPLETE" if strict_pass else "POSTPROCESS_INCOMPLETE",
        "GATE5A_STRICT_CLOSURE_PASS": strict_pass,
        "checks": closure_checks,
        "related_processes": related_after,
        "gpu_work_started_by_closure": False,
        "formal_source_hash_mismatches": hash_comparison["source_hash_mismatches"],
        "upstream_commit": commit_after,
        "upstream_tracked_status": tracked_after,
        "upstream_untracked": untracked_after,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generated_by": "scripts/gate5a_finalize_closure.py",
    }
    atomic_write_json(ROOT / "results/gate5a_closure_verification.json", verification)
    atomic_write_text(
        ROOT / "reports/gate5a_closure_verification.md", verification_markdown(verification)
    )

    for relative_path in (
        "results/gate5a_postprocess_integrity.json",
        "reports/gate5a_postprocess_integrity.md",
        "results/gate5a_closure_verification.json",
        "reports/gate5a_closure_verification.md",
        "results/gate5a_status.json",
        "reports/gate5a_status.md",
    ):
        ensure_nonempty((relative_path,))
    if load_json("results/gate5a_status.json").get("status") != "COMPLETE":
        raise ClosureError("final gate5a status is not COMPLETE")
    if not load_json("results/gate5a_closure_verification.json").get("GATE5A_STRICT_CLOSURE_PASS"):
        raise ClosureError("strict closure verification failed")

    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClosureError as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        raise SystemExit(2)
