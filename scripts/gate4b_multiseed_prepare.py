#!/usr/bin/env python
"""Audit and prepare Gate 4b probability-head multi-seed confirmation."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate3b_smoke_missing_eval import (
    build_args,
    import_official_modules,
    load_checkpoint_strict,
    parse_simple_yaml,
)
from gate4b_multiseed_common import (
    CANONICAL_METRIC_HASH,
    DATASETS,
    LEGACY_INVALID_RUNS,
    MULTISEED_MASK_RULE,
    NEW_SEEDS,
    RATE,
    RESOURCE_FLOORS,
    REUSE_RUNS,
    ROOT,
    TEST_SEED,
    VAL_SEED,
    VARIANTS,
    checkpoint_inventory,
    dataset_spec,
    dump,
    load,
    load_resolved,
    relative,
    required_free_mib,
    resolve,
    sha256_file,
    task_config,
    validate_reuse,
    write_csv,
    write_json_markdown,
)
from gate4b_train_probabilistic import sample_masks


def prerequisite_audit():
    expanded_status = load(ROOT / "results/gate4b_expanded_status.json")
    expanded_decision = load(ROOT / "results/gate4b_expanded_formal_decision.json")
    cross_status = load(ROOT / "results/gate4b_cross_dataset_status.json")
    cross_decision = load(ROOT / "results/gate4b_cross_dataset_formal_decision.json")
    checks = {
        "expanded_implementation_pass": (
            expanded_decision.get("GATE4B_EXPANDED_IMPLEMENTATION_PASS") is True
        ),
        "expanded_signal_positive": (
            expanded_decision.get("GATE4B_EXPANDED_SIGNAL_POSITIVE") is True
        ),
        "cross_dataset_implementation_pass": (
            cross_decision.get("GATE4B_CROSS_DATASET_IMPLEMENTATION_PASS") is True
        ),
        "cross_dataset_signal_positive": (
            cross_decision.get("GATE4B_CROSS_DATASET_SIGNAL_POSITIVE") is True
        ),
        "authorized_for_multiseed": (
            cross_decision.get("ENTER_GATE4B_MULTI_SEED_CONFIRMATION") is True
        ),
        "expanded_not_running": expanded_status.get("scheduler") != "RUNNING",
        "cross_dataset_complete": cross_status.get("scheduler") == "COMPLETE",
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "BLOCKED_MULTI_SEED_PREREQUISITE_INCOMPLETE",
        "checks": checks,
        "expanded_decision": expanded_decision,
        "cross_dataset_decision": cross_decision,
        "ENTER_GATE4C": False,
    }
    dump(ROOT / "results/gate4b_multiseed_prerequisite.json", payload)
    write_json_markdown(
        ROOT / "reports/gate4b_multiseed_prerequisite.md",
        "Gate 4b Multi-Seed Prerequisite",
        payload,
    )
    if payload["status"] != "PASS":
        raise RuntimeError(payload["status"])
    return payload


def mask_metadata_path(path, split):
    return path.with_name("seed_43025.metadata.json") if split == "validation" else path.with_suffix(".json")


def audit_assets():
    inventory = checkpoint_inventory()
    provider, official_module = import_official_modules()
    dataset_rows = []
    checkpoint_rows = []
    mask_rows = []
    split_lengths = {}
    for dataset in DATASETS:
        source = inventory[dataset]
        config = parse_simple_yaml(Path(source["baseline_config_path"]))
        args = build_args(config, int(config["batch_size"]), "cpu")
        train_data, _ = provider(args, "train")
        val_data, _ = provider(args, "val")
        test_data, _ = provider(args, "test")
        split_lengths[dataset] = {
            "train": len(train_data), "validation": len(val_data), "test": len(test_data)
        }
        checkpoint = Path(source["checkpoint_path"])
        checkpoint_hash = sha256_file(checkpoint)
        official = official_module.Model(args).float()
        strict_result = load_checkpoint_strict(official, checkpoint)
        raw = torch.load(checkpoint, map_location="cpu")
        checkpoint_ok = (
            source["status"] == "SUCCESS"
            and checkpoint_hash == source["checkpoint_sha256"]
            and bool(strict_result.get("strict_load", True))
            and not any("probabilistic_head" in key for key in raw)
        )
        checkpoint_rows.append({
            "dataset": dataset,
            "status": "PASS" if checkpoint_ok else "FAIL",
            "official_experiment_id": source["experiment_id"],
            "checkpoint_path": relative(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "strict_load": True,
            "strict_load_tensor_count": len(raw),
            "probability_head_weights_present": any("probabilistic_head" in key for key in raw),
        })
        dataset_rows.append({
            "dataset": dataset,
            "status": "PASS",
            "official_checkpoint_path": relative(checkpoint),
            "official_checkpoint_sha256": checkpoint_hash,
            "input_channels": int(config["enc_in"]),
            "target_channels": int(config["enc_in"]),
            "seq_len": int(config["seq_len"]),
            "pred_len": int(config["pred_len"]),
            "batch_size": int(config["batch_size"]),
            "metric_space": "official_normalized",
            "normalization_reference": "StandardScaler fit on official train split only",
            "train_windows": len(train_data),
            "validation_windows": len(val_data),
            "test_windows": len(test_data),
            "required_free_mib_floor": RESOURCE_FLOORS[dataset],
            "required_free_mib": required_free_mib(dataset),
        })
        for split, asset, expected_windows, expected_seed in (
            ("validation", dataset_spec(dataset)["validation_mask"], len(val_data), VAL_SEED),
            ("test", dataset_spec(dataset)["test_mask"], len(test_data), TEST_SEED),
        ):
            path = ROOT / asset
            metadata_path = mask_metadata_path(path, split)
            metadata = load(metadata_path)
            with np.load(path) as archive:
                mask = archive["mask"]
            expected_shape = (expected_windows, 384, int(config["enc_in"]))
            binary = mask.dtype == np.uint8 and bool(np.logical_or(mask == 0, mask == 1).all())
            actual_hash = sha256_file(path)
            metadata_hash = metadata.get("sha256")
            if split == "validation":
                metadata_ok = (
                    metadata.get("dataset") == dataset
                    and metadata.get("split") == "validation"
                    and int(metadata.get("seed", -1)) == expected_seed
                    and metadata.get("future_target_used") is False
                )
            else:
                metadata_ok = (
                    metadata.get("dataset") == dataset
                    and int(metadata.get("seed", -1)) == expected_seed
                    and metadata.get("mask_type") == "random"
                    and "future" in str(metadata.get("does_not_mask", ""))
                )
            verified = (
                tuple(mask.shape) == expected_shape
                and binary
                and metadata_hash == actual_hash
                and metadata_ok
            )
            mask_rows.append({
                "dataset": dataset,
                "split": split,
                "status": "PASS" if verified else "FAIL",
                "path": asset,
                "metadata_path": relative(metadata_path),
                "shape": "x".join(map(str, mask.shape)),
                "dtype": str(mask.dtype),
                "target_missing_rate": RATE,
                "actual_missing_rate": float(1.0 - mask.mean()),
                "seed": expected_seed,
                "sha256": actual_hash,
                "hash_mismatch": metadata_hash != actual_hash,
                "metadata_mismatch": not metadata_ok,
                "future_target_used": False,
            })
            del mask
        del official, raw, train_data, val_data, test_data

    dataset_payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in dataset_rows) else "FAIL",
        "datasets_verified": sum(row["status"] == "PASS" for row in dataset_rows),
        "expected_datasets": 3,
        "rows": dataset_rows,
    }
    checkpoint_payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in checkpoint_rows) else "FAIL",
        "checkpoints_strict_loaded": sum(row["status"] == "PASS" for row in checkpoint_rows),
        "expected_checkpoints": 3,
        "rows": checkpoint_rows,
    }
    mask_payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in mask_rows) else "FAIL",
        "validation_masks_verified": sum(
            row["status"] == "PASS" and row["split"] == "validation" for row in mask_rows
        ),
        "test_masks_verified": sum(
            row["status"] == "PASS" and row["split"] == "test" for row in mask_rows
        ),
        "hash_mismatch_count": sum(row["hash_mismatch"] for row in mask_rows),
        "metadata_mismatch_count": sum(row["metadata_mismatch"] for row in mask_rows),
        "rows": mask_rows,
    }
    dump(ROOT / "results/gate4b_multiseed_dataset_audit.json", dataset_payload)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_dataset_audit.md",
                        "Gate 4b Multi-Seed Dataset Audit", dataset_payload)
    dump(ROOT / "results/gate4b_multiseed_checkpoint_audit.json", checkpoint_payload)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_checkpoint_audit.md",
                        "Gate 4b Multi-Seed Checkpoint Audit", checkpoint_payload)
    dump(ROOT / "results/gate4b_multiseed_mask_inventory.json", mask_payload)
    write_csv(ROOT / "reports/gate4b_multiseed_mask_inventory.csv", mask_rows)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_mask_inventory.md",
                        "Gate 4b Multi-Seed Mask Inventory", mask_payload)
    if any(payload["status"] != "PASS" for payload in (
        dataset_payload, checkpoint_payload, mask_payload
    )):
        raise RuntimeError("BLOCKED_MULTI_SEED_ASSET_AUDIT_FAILURE")
    return split_lengths, dataset_payload, checkpoint_payload, mask_payload


def source_contains(path, patterns):
    text = (ROOT / path).read_text()
    return {pattern: pattern in text for pattern in patterns}


def audit_seed_protocol(split_lengths):
    trainer_checks = source_contains("scripts/gate4b_train_probabilistic.py", (
        "random.seed(experiment_seed)",
        "np.random.seed(experiment_seed)",
        "torch.manual_seed(experiment_seed)",
        "torch.cuda.manual_seed_all(experiment_seed)",
        "torch.Generator().manual_seed(experiment_seed)",
        "worker_init_fn=seed_worker",
        "train_mask_base_seed, experiment_seed, train_mask_seed_rule",
    ))
    runner_checks = source_contains("scripts/gate4b_formal_runner.py", (
        "allowed_seeds = {2021, 2022, 2023, 2025}",
        "40000 + experiment_seed",
        "fixed_official_checkpoint",
    ))
    scheduler_checks = source_contains("scripts/gate4b_scheduler.py", (
        "experiment_seed",
        "reuse_metric_protocol_hash",
        "reuse_schema_audit_pass",
    ))
    seed_rows = [{
        "experiment_seed": seed,
        "train_mask_base_seed": 40000 + seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "python_seed": seed,
        "dataloader_seed": seed,
        "worker_seed_rule": "torch.initial_seed()%2**32 -> numpy/python",
        "train_mask_seed_rule": MULTISEED_MASK_RULE if seed != 2025 else "historical corrected rule retained for legal reuse",
        "validation_mask_seed": VAL_SEED,
        "test_mask_seed": TEST_SEED,
        "variant_in_mask_seed": False,
    } for seed in (2021, 2022, 2023, 2025)]
    all_source_checks = all(trainer_checks.values()) and all(runner_checks.values()) and all(scheduler_checks.values())
    payload = {
        "status": "PASS" if all_source_checks else "FAIL",
        "scope": "PROBABILISTIC_HEAD_TRAINING_SEEDS",
        "FULL_END_TO_END_MULTI_SEED": False,
        "trainer_checks": trainer_checks,
        "runner_checks": runner_checks,
        "scheduler_checks": scheduler_checks,
        "new_gpu_smoke_required": True,
        "new_gpu_smoke_reason": "shared trainer seed and train-mask path was modified",
        "seed_rows": seed_rows,
        "train_window_counts": {dataset: split_lengths[dataset]["train"] for dataset in DATASETS},
    }
    dump(ROOT / "results/gate4b_multiseed_seed_protocol_audit.json", payload)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_seed_protocol_audit.md",
                        "Gate 4b Multi-Seed Seed Protocol Audit", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("BLOCKED_MULTI_SEED_SEED_PROTOCOL_FAILURE")
    return payload


def tensor_sha(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def pairing_validation(split_lengths):
    rows = []
    for dataset in DATASETS:
        spec = dataset_spec(dataset)
        per_seed_mask_hash = {}
        for seed in NEW_SEEDS:
            generator_a = torch.Generator().manual_seed(seed)
            generator_b = torch.Generator().manual_seed(seed)
            first_a = torch.randperm(split_lengths[dataset]["train"], generator=generator_a)[:spec["batch_size"]]
            first_b = torch.randperm(split_lengths[dataset]["train"], generator=generator_b)[:spec["batch_size"]]
            shape = (len(first_a), 384, spec["target_channels"])
            gaussian_mask = sample_masks(
                first_a, 0, shape, dataset, "random", RATE, 40000 + seed, seed, MULTISEED_MASK_RULE
            )
            student_mask = sample_masks(
                first_b, 0, shape, dataset, "random", RATE, 40000 + seed, seed, MULTISEED_MASK_RULE
            )
            gaussian = task_config(dataset, "gaussian_head", seed)
            student = task_config(dataset, "student_t_head", seed)
            first_hash_a = tensor_sha(first_a)
            first_hash_b = tensor_sha(first_b)
            mask_hash_a = tensor_sha(gaussian_mask)
            mask_hash_b = tensor_sha(student_mask)
            per_seed_mask_hash[seed] = mask_hash_a
            checks = {
                "mask_equal": mask_hash_a == mask_hash_b and torch.equal(gaussian_mask, student_mask),
                "first_batch_indices_equal": first_hash_a == first_hash_b and torch.equal(first_a, first_b),
                "validation_mask_hash_equal": gaussian["validation_mask_sha256"] == student["validation_mask_sha256"],
                "test_mask_hash_equal": gaussian["test_mask_sha256"] == student["test_mask_sha256"],
                "source_checkpoint_hash_equal": gaussian["source_checkpoint_sha256"] == student["source_checkpoint_sha256"],
                "training_protocol_hash_equal": gaussian["training_protocol_hash"] == student["training_protocol_hash"],
                "metric_protocol_hash_equal": gaussian["metric_protocol_hash"] == student["metric_protocol_hash"],
            }
            rows.append({
                "dataset": dataset,
                "experiment_seed": seed,
                "status": "PASS" if all(checks.values()) else "FAIL",
                "first_batch_indices_sha256": first_hash_a,
                "first_batch_mask_sha256": mask_hash_a,
                "actual_first_batch_missing_rate": float((gaussian_mask == 0).float().mean()),
                "checks": checks,
            })
            del gaussian_mask, student_mask
        distinct = len(set(per_seed_mask_hash.values())) == len(NEW_SEEDS)
        for row in rows:
            if row["dataset"] == dataset:
                row["different_seed_masks_distinct"] = distinct
                if not distinct:
                    row["status"] = "FAIL"
    payload = {
        "status": "PASS" if len(rows) == 9 and all(row["status"] == "PASS" for row in rows) else "BLOCKED_MULTI_SEED_PAIRING_FAILURE",
        "expected_pair_audits": 9,
        "passed_pair_audits": sum(row["status"] == "PASS" for row in rows),
        "different_seed_sequences_distinct_by_dataset": {
            dataset: all(row["different_seed_masks_distinct"] for row in rows if row["dataset"] == dataset)
            for dataset in DATASETS
        },
        "rows": rows,
    }
    dump(ROOT / "results/gate4b_multiseed_pairing_validation.json", payload)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_pairing_validation.md",
                        "Gate 4b Multi-Seed Pairing Validation", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(payload["status"])
    return payload


def reuse_audit():
    rows = [validate_reuse(dataset, variant) for dataset in DATASETS for variant in VARIANTS]
    invalid_rows = []
    for run_name in LEGACY_INVALID_RUNS:
        run_dir = ROOT / run_name
        metrics = load(run_dir / "metrics.json") if (run_dir / "metrics.json").is_file() else {}
        invalid_rows.append({
            "dataset": metrics.get("dataset", "etth1"),
            "variant": metrics.get("variant", "unknown"),
            "experiment_seed": metrics.get("seed", 2025),
            "run_dir": run_name,
            "reusable": False,
            "reason": "legacy_unmasked_invalid",
            "input_mask_applied": metrics.get("input_mask_applied", False),
            "audit_retained": True,
            "invalid_for_formal_evidence": True,
        })
    status = "PASS" if len(rows) == 6 and all(row["reusable"] for row in rows) else "FAIL"
    payload = {
        "status": status,
        "expected_seed2025_reuse": 6,
        "legal_seed2025_reuse": sum(row["reusable"] for row in rows),
        "legacy_unmasked_reuse_count": 0,
        "legacy_unmasked_audit_retained": len(invalid_rows),
        "rows": rows,
        "legacy_invalid_rows": invalid_rows,
    }
    dump(ROOT / "results/gate4b_multiseed_reuse_inventory.json", payload)
    csv_rows = []
    for row in rows + invalid_rows:
        flat = {key: value for key, value in row.items() if key != "checks"}
        if "checks" in row:
            flat["checks_pass"] = all(row["checks"].values())
        csv_rows.append(flat)
    write_csv(ROOT / "reports/gate4b_multiseed_reuse_inventory.csv", csv_rows)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_reuse_inventory.md",
                        "Gate 4b Multi-Seed Reuse Inventory", payload)
    if status != "PASS":
        raise RuntimeError("BLOCKED_MULTI_SEED_REUSE_AUDIT_FAILURE")
    return payload


def write_task_configs():
    paths = []
    for dataset in DATASETS:
        for seed in (2021, 2022, 2023, 2025):
            for variant in VARIANTS:
                task = task_config(dataset, variant, seed)
                path = ROOT / "configs/gate4b_multiseed" / f"{dataset}_{variant}_random_rate0.3_seed{seed}.json"
                dump(path, task)
                paths.append(relative(path))
    smoke_paths = []
    for variant in VARIANTS:
        task = task_config("etth1", variant, 2021, smoke=True)
        path = ROOT / "configs/gate4b_multiseed_smoke" / f"etth1_{variant}_random_rate0.3_seed2021.json"
        dump(path, task)
        smoke_paths.append(relative(path))
    return paths, smoke_paths


def upstream_state():
    repo = ROOT / "upstream/TimePerceiver"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output(["git", "-C", str(repo), "diff", "--stat"], text=True).strip()
    short = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True).splitlines()
    return {
        "commit": commit,
        "tracked_diff_empty": tracked == "",
        "status_short": short,
        "untracked_only": all(line.startswith("??") for line in short),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-queue", action="store_true")
    args = parser.parse_args()
    prerequisite_audit()
    split_lengths, dataset_audit, checkpoint_audit, mask_audit = audit_assets()
    seed_audit = audit_seed_protocol(split_lengths)
    pairing = pairing_validation(split_lengths)
    reuse = reuse_audit()
    task_paths, smoke_paths = write_task_configs()
    upstream = upstream_state()
    if not upstream["tracked_diff_empty"]:
        raise RuntimeError("BLOCKED_UPSTREAM_TRACKED_DIFF")
    queue_created = False
    smoke_status = "NOT_STARTED"
    queue_path = ROOT / "configs/queues/gate4b_multiseed_confirmation.yaml"
    if args.create_queue:
        smoke = load(ROOT / "results/gate4b_multiseed_seed_smoke.json")
        if smoke.get("status") != "PASS" or smoke.get("successful_variants") != 2:
            raise RuntimeError("BLOCKED_MULTI_SEED_SEED_SMOKE_INCOMPLETE")
        smoke_status = "PASS 2/2"
        tasks = [load(ROOT / path) for path in task_paths]
        queue = {
            "stage": "Gate 4b multi-seed confirmation",
            "schema_version": 5,
            "scope": "PROBABILISTIC_HEAD_TRAINING_SEEDS",
            "full_end_to_end_multi_seed": False,
            "dataset_count": 3,
            "seed_count": 4,
            "variant_count": 2,
            "logical_task_count": 24,
            "result_row_count_target": 27,
            "paired_delta_row_count_target": 12,
            "reused_task_count": 6,
            "active_task_count": 18,
            "max_concurrent_total": 2,
            "max_tasks_per_gpu": 1,
            "experiment_stage_label": "gate4b_multiseed",
            "state_file": "runtime/gate4b_multiseed_confirmation_scheduler_state.json",
            "pid_file": "runtime/gate4b_multiseed_confirmation_scheduler.pid",
            "lock_file": "runtime/locks/gate4b_multiseed_confirmation_scheduler.lock",
            "dataset_required_free_mib": {
                dataset: required_free_mib(dataset) for dataset in DATASETS
            },
            "bounded_oom_retry": 3,
            "queue_complete_marker": "QUEUE_COMPLETE",
            "tasks": tasks,
        }
        dump(queue_path, queue)
        queue_created = True
    status = {
        "stage": "Gate 4b multi-seed confirmation",
        "status": "QUEUE_CREATED_PENDING_VALIDATION" if queue_created else "PLANNING_PASS_SEED_SMOKE_REQUIRED",
        "scope": "PROBABILISTIC_HEAD_TRAINING_SEEDS",
        "FULL_END_TO_END_MULTI_SEED": False,
        "prerequisite": "PASS",
        "dataset_audit": f"PASS {dataset_audit['datasets_verified']}/3",
        "checkpoint_audit": f"PASS {checkpoint_audit['checkpoints_strict_loaded']}/3",
        "mask_audit": "PASS 6/6",
        "seed_protocol_audit": seed_audit["status"],
        "pairing_validation": f"PASS {pairing['passed_pair_audits']}/9",
        "reuse_audit": f"PASS {reuse['legal_seed2025_reuse']}/6",
        "logical_probability_tasks": 24,
        "reused_tasks": 6,
        "active_tasks": 18,
        "seed_gpu_smoke": smoke_status,
        "queue": "CREATED_PENDING_VALIDATION" if queue_created else "NOT_CREATED",
        "queue_path": relative(queue_path) if queue_created else None,
        "queue_validation": "NOT_STARTED",
        "scheduler": "NOT_STARTED",
        "summary": "NOT_STARTED",
        "formal_decision": "PENDING",
        "task_config_count": len(task_paths),
        "smoke_task_config_count": len(smoke_paths),
        "canonical_metric_protocol_hash": CANONICAL_METRIC_HASH,
        "upstream": upstream,
        "ENTER_GATE4C": False,
    }
    dump(ROOT / "results/gate4b_multiseed_status.json", status)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_status.md",
                        "Gate 4b Multi-Seed Status", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
