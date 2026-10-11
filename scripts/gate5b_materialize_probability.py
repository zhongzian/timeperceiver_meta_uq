#!/usr/bin/env python3
"""Bind Gate 5b1 probability tasks to immutable Gate 5b0 checkpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import (  # noqa: E402
    DATASETS,
    PATCHTST_COMMIT,
    dump,
    load,
    probability_config,
    resolve,
    sha256_file,
    write_report,
    write_resolved,
)


def main():
    manifest_path = ROOT / "results/gate5b_patchtst_backbone_manifest.json"
    queue_path = ROOT / "configs/queues/gate5b_patchtst_probability.yaml"
    manifest = load(manifest_path)
    queue = load(queue_path)
    if manifest.get("status") != "PASS" or set(manifest.get("datasets", {})) != set(DATASETS):
        raise SystemExit("Gate 5b0 manifest is not 3/3 PASS")
    provenance = []
    for dataset in DATASETS:
        item = manifest["datasets"][dataset]
        checkpoint = resolve(item["checkpoint"])
        actual_hash = sha256_file(checkpoint)
        checkpoint_object = torch.load(checkpoint, map_location="cpu")
        checks = {
            "checkpoint_exists": checkpoint.is_file(),
            "sha256_match": actual_hash == item["checkpoint_sha256"],
            "dataset_match": checkpoint_object.get("dataset") == dataset,
            "seed_match": int(checkpoint_object.get("experiment_seed", -1)) == 2025,
            "patchtst_commit_match": checkpoint_object.get("patchtst_upstream_commit") == PATCHTST_COMMIT,
            "model_state_present": isinstance(checkpoint_object.get("model_state_dict"), dict),
            "selection_metric_validation_mse": item.get("selection_metric") == "validation_mse",
            "test_not_used_for_selection": item.get("test_used_for_model_selection") is False,
            "strict_reload": item.get("strict_reload") is True,
        }
        provenance.append({
            "dataset": dataset, "checkpoint": item["checkpoint"],
            "checkpoint_sha256": actual_hash, "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        })
    if not all(row["status"] == "PASS" for row in provenance):
        raise SystemExit("Gate 5b0 checkpoint provenance audit failed")

    task_order = queue.get("scheduling_order", list(DATASETS))
    tasks = []
    for dataset in task_order:
        item = manifest["datasets"][dataset]
        for distribution in ("gaussian", "fixed_t8"):
            task = probability_config(
                dataset, distribution, smoke=False,
                checkpoint=item["checkpoint"], checkpoint_sha256=item["checkpoint_sha256"],
            )
            previous = next(
                old for old in queue["tasks"]
                if old["dataset"] == dataset and old["distribution"] == distribution
            )
            for key in (
                "required_free_mib", "resource_floor_original_mib", "resource_floor_effective_mib",
                "resource_floor_user_override",
            ):
                if key in previous:
                    task[key] = previous[key]
            task["task_id"] = f"{dataset}__{distribution}__random_rate0.3_seed2025"
            task["reuse_allowed"] = False
            tasks.append(task)
            suffix = "gaussian" if distribution == "gaussian" else "fixed_t8"
            write_resolved(
                ROOT / f"configs/gate5b/heads/patchtst_{dataset}_{suffix}_random0.3_seed2025.yaml",
                task,
            )
    queue["tasks"] = tasks
    queue["launch_enabled"] = False
    queue["blocked_reason"] = "PENDING_DISTRIBUTION_PAIRING_AND_PROBABILITY_SMOKE"
    queue["checkpoint_manifest"] = "results/gate5b_patchtst_backbone_manifest.json"
    queue["checkpoint_manifest_status"] = "PASS"
    dump(queue_path, queue)
    payload = {
        "status": "PASS", "manifest": str(manifest_path.relative_to(ROOT)),
        "datasets": provenance, "probability_tasks_materialized": len(tasks),
        "all_checkpoint_hashes_match": True, "queue_launch_enabled": False,
    }
    dump(ROOT / "results/gate5b_checkpoint_provenance.json", payload)
    write_report(
        ROOT / "reports/gate5b_checkpoint_provenance.md",
        "Gate 5b Checkpoint Provenance", payload,
    )
    print(json.dumps({
        "status": payload["status"], "probability_tasks_materialized": len(tasks),
        "queue_launch_enabled": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

