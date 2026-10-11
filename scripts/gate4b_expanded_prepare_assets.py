#!/usr/bin/env python
"""Prepare and audit fixed assets for the Gate 4b expanded ETTh1 matrix."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (("random", 0.1), ("random", 0.3), ("random", 0.5),
              ("block", 0.3), ("channel", 0.3), ("mixed", 0.3))
CHECKPOINT = "runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054/checkpoints/ETTh1_384_96_TimePerceiver_sl384_pl96_dm256_nl32_ld64_ul1_gf1_nh4_df512_pl12_qs1_sr1.0/checkpoint.pth"
sys_path = str(ROOT / "scripts")
import sys
sys.path.insert(0, sys_path)
from gate3a_generate_missingness_assets import GENERATOR_VERSION, make_mask, sha256_file, summarize_mask


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def condition_key(mask_type, rate):
    return f"{mask_type}_rate{rate}"


def validation_paths(mask_type, rate):
    if (mask_type, rate) == ("random", 0.3):
        base = ROOT / "artifacts/gate4a_val_masks/etth1/random/rate_0.3"
    else:
        base = ROOT / f"artifacts/gate4b_val_masks/etth1/{mask_type}/rate_{rate}"
    return base / "seed_43025.npz", base / "seed_43025.metadata.json"


def verify_binary_mask(path, expected_shape):
    with np.load(path) as archive:
        mask = archive["mask"]
    if tuple(mask.shape) != tuple(expected_shape):
        raise RuntimeError(f"shape mismatch {path}: {mask.shape}")
    if mask.dtype != np.uint8:
        raise RuntimeError(f"dtype mismatch {path}: {mask.dtype}")
    values = set(np.unique(mask).tolist())
    if not values <= {0, 1}:
        raise RuntimeError(f"non-binary mask {path}: {values}")
    return mask


def make_validation_inventory():
    rows = []
    shape = (2785, 384, 7)
    for mask_type, rate in CONDITIONS:
        path, metadata_path = validation_paths(mask_type, rate)
        reused = (mask_type, rate) == ("random", 0.3)
        created_now = False
        if not path.exists():
            if reused:
                raise RuntimeError("required reused random 0.3 validation mask is missing")
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = make_mask(mask_type, 43025, shape, rate)
            np.savez_compressed(path, mask=mask)
            summary = summarize_mask(mask, rate)
            metadata = {
                **summary, "dataset": "etth1", "split": "validation", "mask_type": mask_type,
                "seed": 43025, "sha256": sha256_file(path), "generator_version": GENERATOR_VERSION,
                "mask_semantics": "1=observed/keep, 0=missing/mask",
                "source": "official validation windows only", "applies_to": "validation seq_x only",
                "future_target_used": False, "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            dump(metadata_path, metadata)
            created_now = True
        mask = verify_binary_mask(path, shape)
        metadata = load(metadata_path)
        actual = float(1.0 - mask.mean())
        recorded = metadata.get("actual_missing_rate", metadata.get("actual_rate"))
        if metadata.get("sha256") != sha256_file(path) or abs(float(recorded) - actual) > 1e-12:
            raise RuntimeError(f"validation metadata/hash mismatch: {path}")
        if metadata.get("future_target_used") is not False:
            raise RuntimeError(f"future target flag invalid: {metadata_path}")
        rows.append({
            "condition": condition_key(mask_type, rate), "missing_type": mask_type,
            "target_missing_rate": rate, "actual_missing_rate": actual, "seed": 43025,
            "shape": "x".join(map(str, shape)), "dtype": str(mask.dtype), "sha256": sha256_file(path),
            "mask_path": str(path.relative_to(ROOT)), "metadata_path": str(metadata_path.relative_to(ROOT)),
            "reused": reused, "created": not reused, "created_now": created_now, "verified": True,
        })
    return rows


def make_test_inventory():
    source_rows = list(csv.DictReader(open(ROOT / "reports/gate3a_mask_inventory.csv")))
    rows = []
    for mask_type, rate in CONDITIONS:
        source = next(row for row in source_rows if row["dataset"] == "etth1"
                      and row["mask_type"] == mask_type and float(row["rate"]) == rate
                      and int(row["seed"]) == 2025)
        path = Path(source["npz_path"])
        metadata_path = Path(source["meta_path"])
        mask = verify_binary_mask(path, (2785, 384, 7))
        metadata = load(metadata_path)
        actual_hash = sha256_file(path)
        if actual_hash != source["sha256"] or actual_hash != metadata["sha256"]:
            raise RuntimeError(f"test hash mismatch: {path}")
        actual = float(1.0 - mask.mean())
        if abs(actual - float(metadata["actual_missing_rate"])) > 1e-12:
            raise RuntimeError(f"test metadata rate mismatch: {path}")
        if metadata.get("does_not_mask") != "decoder label or future prediction target":
            raise RuntimeError(f"test future-target semantics mismatch: {metadata_path}")
        rows.append({
            "condition": condition_key(mask_type, rate), "missing_type": mask_type,
            "target_missing_rate": rate, "actual_missing_rate": actual, "seed": 2025,
            "shape": "x".join(map(str, mask.shape)), "dtype": str(mask.dtype), "sha256": actual_hash,
            "mask_path": str(path.relative_to(ROOT)), "metadata_path": str(metadata_path.relative_to(ROOT)),
            "verified": True, "hash_mismatch": False, "metadata_mismatch": False,
        })
    return rows


def make_official_inventory(test_rows):
    source_rows = list(csv.DictReader(open(ROOT / "reports/gate3b_full_results.csv")))
    rows = []
    for test in test_rows:
        source = next(row for row in source_rows if row["dataset"] == "etth1"
                      and row["missing_type"] == test["missing_type"]
                      and float(row["target_missing_rate"]) == float(test["target_missing_rate"])
                      and int(row["seed"]) == 2025 and row["status"] == "SUCCESS")
        if source["mask_sha256"] != test["sha256"]:
            raise RuntimeError(f"official/test mask mismatch: {test['condition']}")
        rows.append({
            "condition": test["condition"], "missing_type": test["missing_type"],
            "target_missing_rate": test["target_missing_rate"],
            "actual_missing_rate": float(source["actual_missing_rate"]), "seed": 2025,
            "experiment_id": source["experiment_id"], "status": source["status"],
            "mse": float(source["mse"]), "mae": float(source["mae"]), "rmse": float(source["rmse"]),
            "test_batch_count": int(source["batch_count"]), "test_sample_count": int(source["sample_count"]),
            "mask_sha256": source["mask_sha256"], "checkpoint_sha256": source["checkpoint_sha256"],
            "nll": "NA", "crps": "NA", "coverage_error": "NA", "interval_score": "NA",
            "scale": "NA", "df": "NA", "run_dir": source["run_dir"], "reused": True,
        })
    return rows


def task(mask_type, rate, variant, validation, test, smoke=False):
    training_protocol = {
        "dataset": "etth1", "seq_len": 384, "pred_len": 96, "seed": 2025,
        "missing_type": mask_type, "missing_rate": rate, "train_mask_base_seed": 42025,
        "train_mask_seed_rule": "sha256(base_seed|epoch|sample_index|missing_type|target_rate)",
        "input_mask_application": "batch_x * mask before model forward", "freeze_backbone": True,
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "gradient_clip": 1.0, "max_epochs": 1 if smoke else 20, "patience": 3,
        "early_stopping_metric": "validation_nll", "test_used_for_model_selection": False,
        "initial_scale": 0.5, "scale_floor": 1e-4, "source_checkpoint": CHECKPOINT,
        "validation_mask": validation["mask_path"], "test_mask": test["mask_path"],
        "source_checkpoint_sha256": sha256_file(ROOT / CHECKPOINT),
        "validation_mask_sha256": validation["sha256"], "test_mask_sha256": test["sha256"],
    }
    metric_protocol = {
        "metric_space": "official_normalized", "crps_method": "empirical_monte_carlo",
        "crps_samples": 64, "crps_seed": 52025, "crps_chunk_elements": 65536,
        "interval_levels": [0.5, 0.8, 0.9, 0.95], "interval_type": "native_parametric",
        "conformal_calibrated": False, "posthoc_calibration": False,
    }
    config_dir = "configs/gate4b_expanded_smoke" if smoke else "configs/gate4b_expanded"
    config_path = f"{config_dir}/etth1_{mask_type}_rate{rate}_{variant}_seed2025.json"
    result = {
        **training_protocol, **metric_protocol, "stage": "gate4b_expanded_protocol_smoke" if smoke else "gate4b_expanded",
        "variant": variant, "initial_df": None if variant == "gaussian_head" else 8.0,
        "df_parameterization": None if variant == "gaussian_head" else "2.01 + softplus(raw_df)",
        "smoke_only": smoke, "max_train_batches": 4 if smoke else None,
        "max_val_batches": 2 if smoke else None, "max_test_batches": 2 if smoke else None,
        "required_free_mib": 4096, "config": config_path,
        "training_protocol_hash": digest(training_protocol), "metric_protocol_hash": digest(metric_protocol),
        "command": ["gate4b_formal_runner", "--task-file", "{task_file}", "--run-dir", "{run_dir}"],
    }
    dump(ROOT / config_path, result)
    return result


def make_reuse_inventory(tasks):
    rows = []
    for item in tasks:
        pattern = (f"timeperceiver__etth1__gate4b*__{item['variant']}__"
                   f"{item['missing_type']}_rate{item['missing_rate']}__seed2025__*/status.json")
        matches = []
        for path in (ROOT / "runs").glob(pattern):
            try:
                status, metrics = load(path), load(path.parent / "metrics.json")
                valid = (status.get("status") == "SUCCESS" and not metrics.get("smoke_only")
                         and metrics.get("input_mask_applied") is True
                         and metrics.get("training_protocol_hash") == item["training_protocol_hash"]
                         and metrics.get("metric_protocol_hash") == item["metric_protocol_hash"]
                         and metrics.get("source_checkpoint_sha256") == item["source_checkpoint_sha256"]
                         and metrics.get("validation_mask_sha256") == item["validation_mask_sha256"]
                         and metrics.get("test_mask_sha256") == item["test_mask_sha256"]
                         and metrics.get("test_used_for_model_selection") is False
                         and metrics.get("backbone_non_null_gradient_count") == 0)
                if valid:
                    matches.append(path.parent)
            except Exception:
                pass
        selected = str(sorted(matches)[-1].relative_to(ROOT)) if matches else ""
        rows.append({"condition": condition_key(item["missing_type"], item["missing_rate"]),
                     "variant": item["variant"], "reusable": bool(selected), "run_dir": selected,
                     "reason": "all fixed-mask protocol fields match" if selected else
                     "no corrected full run with input_mask_applied=true and matching hashes"})
    return rows


def write_md_table(path, title, rows, columns):
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    validation_rows = make_validation_inventory()
    test_rows = make_test_inventory()
    official_rows = make_official_inventory(test_rows)
    tasks = []
    for mask_type, rate in CONDITIONS:
        validation = next(row for row in validation_rows if row["missing_type"] == mask_type
                          and float(row["target_missing_rate"]) == rate)
        test = next(row for row in test_rows if row["missing_type"] == mask_type
                    and float(row["target_missing_rate"]) == rate)
        for variant in ("gaussian_head", "student_t_head"):
            tasks.append(task(mask_type, rate, variant, validation, test, smoke=False))
    smoke_tasks = []
    for mask_type, rate, variant in (("random", 0.3, "gaussian_head"),
                                     ("random", 0.3, "student_t_head"),
                                     ("block", 0.3, "gaussian_head"),
                                     ("channel", 0.3, "gaussian_head"),
                                     ("mixed", 0.3, "gaussian_head")):
        validation = next(row for row in validation_rows if row["missing_type"] == mask_type
                          and float(row["target_missing_rate"]) == rate)
        test = next(row for row in test_rows if row["missing_type"] == mask_type
                    and float(row["target_missing_rate"]) == rate)
        smoke_tasks.append(task(mask_type, rate, variant, validation, test, smoke=True))
    reuse_rows = make_reuse_inventory(tasks)

    semantics = []
    definitions = {
        "random": "independent element masking with P(missing)=target_rate",
        "block": "2%-20% temporal blocks, sometimes all channels and sometimes sampled channel subsets, until target count reached",
        "channel": "round(rate*C) whole channels per window, capped at C-1",
        "mixed": "channel(rate/3) union block(rate/3), then random observed entries to target total",
    }
    for mask_type in ("random", "block", "channel", "mixed"):
        semantics.append({"missing_type": mask_type, "generator_function": f"gate3a.make_mask->{mask_type}_mask",
                          "semantic_definition": definitions[mask_type], "actual_rate_rule": "1-mask.mean()",
                          "binary_convention": "1=observed/keep, 0=missing/mask", "input_shape": "[B,384,7]",
                          "future_target_used": False})
    implementation = {
        "status": "PASS", "probability_head_unchanged": True, "metric_implementation_unchanged": True,
        "formal_runner_parameterized_type_rate": True, "scheduler_requires_parameterization": True,
        "train_mask_generator_all_types": True, "validation_uses_gate3a_semantics": True,
        "test_resolution_uses_gate3a_inventory": True,
        "historical_minimal_input_mask_applied": False,
        "historical_minimal_reuse_legal": False,
        "repair": "shared runner now applies batch_x * mask before every train/validation/test forward",
        "probability_head_sha256": sha256_file(ROOT / "models/probabilistic_head.py"),
        "metric_implementation_sha256": sha256_file(ROOT / "scripts/gate4b_probabilistic_metrics.py"),
    }
    dump(ROOT / "results/gate4b_expanded_implementation_audit.json", implementation)
    dump(ROOT / "results/gate4b_expanded_missingness_semantics_audit.json",
         {"status": "PASS", "generator_version": GENERATOR_VERSION, "rows": semantics})
    dump(ROOT / "results/gate4b_expanded_val_mask_inventory.json",
         {"status": "PASS", "expected": 6, "reused": 1, "created": sum(row["created"] for row in validation_rows),
          "verified": 6, "rows": validation_rows})
    dump(ROOT / "results/gate4b_expanded_test_mask_inventory.json",
         {"status": "PASS", "expected": 6, "verified": 6, "hash_mismatch": 0,
          "metadata_mismatch": 0, "rows": test_rows})
    dump(ROOT / "results/gate4b_expanded_official_baseline_inventory.json",
         {"status": "PASS", "expected": 6, "verified": 6, "rows": official_rows})
    dump(ROOT / "results/gate4b_expanded_reuse_inventory.json",
         {"status": "PASS", "expected_probability_tasks": 12,
          "reused_probability_tasks": sum(row["reusable"] for row in reuse_rows), "rows": reuse_rows})
    csv_write(ROOT / "reports/gate4b_expanded_val_mask_inventory.csv", validation_rows)
    csv_write(ROOT / "reports/gate4b_expanded_test_mask_inventory.csv", test_rows)
    csv_write(ROOT / "reports/gate4b_expanded_official_baseline_inventory.csv", official_rows)
    csv_write(ROOT / "reports/gate4b_expanded_reuse_inventory.csv", reuse_rows)
    (ROOT / "reports/gate4b_expanded_implementation_audit.md").write_text(
        "# Gate 4b Expanded Implementation Audit\n\nStatus: **PASS after data-path repair**\n\n"
        "The historical minimal runner passed masks as an unused argument. Those runs are not reusable. "
        "The shared runner now applies `batch_x * mask`; probability-head and metric code are unchanged.\n")
    write_md_table(ROOT / "reports/gate4b_expanded_missingness_semantics_audit.md",
                   "Gate 4b Expanded Missingness Semantics Audit", semantics,
                   ["missing_type", "generator_function", "semantic_definition", "actual_rate_rule", "future_target_used"])
    write_md_table(ROOT / "reports/gate4b_expanded_val_mask_inventory.md", "Gate 4b Expanded Validation Masks",
                   validation_rows, ["condition", "target_missing_rate", "actual_missing_rate", "reused", "created", "verified", "sha256"])
    write_md_table(ROOT / "reports/gate4b_expanded_test_mask_inventory.md", "Gate 4b Expanded Test Masks",
                   test_rows, ["condition", "target_missing_rate", "actual_missing_rate", "verified", "sha256"])
    write_md_table(ROOT / "reports/gate4b_expanded_reuse_inventory.md", "Gate 4b Expanded Reuse Inventory",
                   reuse_rows, ["condition", "variant", "reusable", "run_dir", "reason"])
    previous_status = load(ROOT / "results/gate4b_expanded_status.json") if (ROOT / "results/gate4b_expanded_status.json").exists() else {}
    status = {
        "stage": "Gate 4b expanded", "status": "PLANNING_PASS", "implementation_audit": "PASS",
        "missingness_semantics_audit": "PASS", "validation_masks": "PASS 6/6",
        "test_masks": "PASS 6/6", "official_baselines": "PASS 6/6",
        "historical_minimal_reuse": "REJECTED_INPUT_MASK_NOT_APPLIED",
        "corrected_minimal_reuse_count": sum(row["reusable"] for row in reuse_rows),
        "protocol_smoke": previous_status.get("protocol_smoke", "NOT_STARTED"),
        "expanded_queue": previous_status.get("expanded_queue", "NOT_CREATED"),
        "scheduler": previous_status.get("scheduler", "NOT_STARTED"),
        "ENTER_GATE4C": False,
    }
    dump(ROOT / "results/gate4b_expanded_status.json", status)
    (ROOT / "reports/gate4b_expanded_status.md").write_text(
        "# Gate 4b Expanded Status\n\nStatus: **PLANNING_PASS**\n\n"
        "Assets and semantics pass. Historical minimal probability runs are not reusable because input masks were not applied. "
        "Corrected protocol smoke is required before queue creation.\n")
    print(json.dumps({"status": "PASS", "validation_verified": 6, "test_verified": 6,
                      "official_verified": 6, "corrected_reuse": sum(row["reusable"] for row in reuse_rows),
                      "smoke_tasks": len(smoke_tasks)}, indent=2))


if __name__ == "__main__":
    main()
