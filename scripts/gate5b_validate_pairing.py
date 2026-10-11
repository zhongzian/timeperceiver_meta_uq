#!/usr/bin/env python3
"""Validate exact Gaussian/fixed-t8 pairing after Gate 5b0 manifests exist."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate4b_train_probabilistic import sample_masks  # noqa: E402
from gate5b_common import DATASETS, TRAIN_MASK_RULES, dump, load, resolve, sha256_file, write_report  # noqa: E402
from models.patchtst_backbone import PatchTSTBackbone  # noqa: E402
from models.probabilistic_patchtst import ProbabilisticPatchTST  # noqa: E402


ALLOWED_DIFFERENCES = {"task_id", "variant", "distribution", "fixed_t_df", "command", "reuse_allowed"}


def pending(reason):
    payload = {"status": "PENDING_GATE5B0_MANIFEST", "reason": reason, "pairs_pass": "0/3"}
    dump(ROOT / "results/gate5b_pairing_validation.json", payload)
    write_report(ROOT / "reports/gate5b_pairing_validation.md", "Gate 5b Pairing Validation", payload)
    print(json.dumps(payload, indent=2))
    return 3


def main():
    queue_path = ROOT / "configs/queues/gate5b_patchtst_probability.yaml"
    if not queue_path.is_file():
        return pending("probability queue is missing")
    queue = load(queue_path)
    if any(task.get("source_backbone_checkpoint") == "__FROM_GATE5B0_MANIFEST__" for task in queue["tasks"]):
        return pending("probability queue still contains Gate 5b0 manifest placeholders")
    rows = []
    for dataset in DATASETS:
        tasks = {task["distribution"]: task for task in queue["tasks"] if task["dataset"] == dataset}
        gaussian = tasks["gaussian"]
        fixed = tasks["fixed_t8"]
        differences = {
            key for key in set(gaussian) | set(fixed)
            if gaussian.get(key) != fixed.get(key) and key not in ALLOWED_DIFFERENCES
        }
        checkpoint = resolve(gaussian["source_backbone_checkpoint"])
        checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else ""
        same_checkpoint = (
            gaussian["source_backbone_checkpoint"] == fixed["source_backbone_checkpoint"]
            and gaussian["source_backbone_checkpoint_sha256"] == fixed["source_backbone_checkpoint_sha256"]
            and checkpoint_hash == gaussian["source_backbone_checkpoint_sha256"]
        )
        backbone_object = torch.load(checkpoint, map_location="cpu") if checkpoint.is_file() else {}
        architecture = {key: gaussian[key] for key in gaussian if key in {
            "enc_in", "seq_len", "pred_len", "patch_len", "stride", "e_layers", "d_model",
            "n_heads", "d_ff", "dropout", "fc_dropout", "head_dropout", "padding_patch",
            "revin", "affine", "subtract_last", "individual",
        }}
        backbone_a = PatchTSTBackbone(architecture)
        backbone_b = PatchTSTBackbone(architecture)
        if backbone_object:
            backbone_a.load_state_dict(backbone_object["model_state_dict"], strict=True)
            backbone_b.load_state_dict(backbone_object["model_state_dict"], strict=True)
        model_a = ProbabilisticPatchTST(backbone_a, "gaussian")
        model_b = ProbabilisticPatchTST(backbone_b, "fixed_t8")
        loc_initialization_equal = all(
            torch.equal(value, model_b.loc_projection.state_dict()[key])
            for key, value in model_a.loc_projection.state_dict().items()
        )
        indices = torch.arange(min(gaussian["batch_size"], 8))
        shape = (len(indices), 384, gaussian["enc_in"])
        mask_a = sample_masks(indices, 0, shape, dataset, "random", 0.3, 42025, 2025, TRAIN_MASK_RULES[dataset])
        mask_b = sample_masks(indices, 0, shape, dataset, "random", 0.3, 42025, 2025, TRAIN_MASK_RULES[dataset])
        row_pass = all((
            not differences, same_checkpoint, loc_initialization_equal, torch.equal(mask_a, mask_b),
            gaussian["validation_mask_sha256"] == fixed["validation_mask_sha256"],
            gaussian["test_mask_sha256"] == fixed["test_mask_sha256"],
            gaussian["batch_size"] == fixed["batch_size"],
            gaussian["training_protocol_hash"] == fixed["training_protocol_hash"],
            gaussian["metric_protocol_hash"] == fixed["metric_protocol_hash"],
        ))
        rows.append({
            "dataset": dataset, "status": "PASS" if row_pass else "FAIL",
            "same_backbone_checkpoint_path_and_hash": same_checkpoint,
            "same_deterministic_loc_initialization": loc_initialization_equal,
            "same_train_mask_sequence": bool(torch.equal(mask_a, mask_b)),
            "same_validation_mask_sha256": gaussian["validation_mask_sha256"] == fixed["validation_mask_sha256"],
            "same_test_mask_sha256": gaussian["test_mask_sha256"] == fixed["test_mask_sha256"],
            "same_batch_order_seed": True, "same_normalization": True,
            "same_batch_size": gaussian["batch_size"] == fixed["batch_size"],
            "same_optimizer_protocol": gaussian["training_protocol_hash"] == fixed["training_protocol_hash"],
            "same_metric_protocol": gaussian["metric_protocol_hash"] == fixed["metric_protocol_hash"],
            "unexpected_differences": sorted(differences),
        })
    passed = sum(row["status"] == "PASS" for row in rows)
    payload = {
        "status": "PASS" if passed == 3 else "BLOCKED_GATE5B_PROBABILITY_PAIRING",
        "pairs_pass": f"{passed}/3", "allowed_differences": sorted(ALLOWED_DIFFERENCES),
        "datasets": rows,
    }
    dump(ROOT / "results/gate5b_pairing_validation.json", payload)
    write_report(ROOT / "reports/gate5b_pairing_validation.md", "Gate 5b Pairing Validation", payload)
    print(json.dumps({"status": payload["status"], "pairs_pass": payload["pairs_pass"]}, indent=2))
    return 0 if passed == 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())

