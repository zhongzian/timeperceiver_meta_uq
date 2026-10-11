#!/usr/bin/env python
"""Create Gate 4b smoke summaries and the two-task formal matrix."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKES = {
    "gaussian_head": ROOT / "runs/timeperceiver__etth1__gate4b_smoke__gaussian_head__random_rate0.3__seed2025__20260712_232200",
    "student_t_head": ROOT / "runs/timeperceiver__etth1__gate4b_smoke__student_t_head__random_rate0.3__seed2025__20260712_232400",
}
CHECKPOINT = "runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054/checkpoints/ETTh1_384_96_TimePerceiver_sl384_pl96_dm256_nl32_ld64_ul1_gf1_nh4_df512_pl12_qs1_sr1.0/checkpoint.pth"
VAL_MASK = "artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz"
TEST_MASK = "artifacts/masks/etth1/random/rate_0.3/seed_2025.npz"


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    rows = {}
    for variant, run in SMOKES.items():
        status, metrics = load(run / "status.json"), load(run / "metrics.json")
        if status.get("status") != "SUCCESS" or not metrics.get("smoke_only"):
            raise RuntimeError(f"invalid smoke result: {variant}")
        rows[variant] = {"run_dir": str(run.relative_to(ROOT)), **metrics}

    peak = max(float(row["peak_gpu_memory_mib"]) for row in rows.values())
    required = max(4096, math.ceil(peak * 1.5) + 512)
    smoke = {
        "status": "PASS",
        "GATE4B_TRAINING_SMOKE_PASS": True,
        "required_free_mib": required,
        "peak_formula": "max(4096, ceil(max_smoke_peak_mib * 1.5) + 512)",
        "variants": rows,
    }
    dump(ROOT / "results/gate4b_training_smoke.json", smoke)
    lines = [
        "# Gate 4b Training Smoke",
        "",
        "Status: **PASS**",
        "",
        "| Variant | Train batches | Val batches | Test batches | NLL finite | Gradients | Backbone gradients | Reload | Peak GPU MiB |",
        "|---|---:|---:|---:|---|---|---:|---|---:|",
    ]
    for variant, row in rows.items():
        lines.append(
            f"| {variant} | {row['train_batches']} | {row['val_batches']} | {row['test_batches']} | "
            f"{row['train_nll_finite'] and row['validation_nll_finite']} | {row['all_new_gradients_finite']} | "
            f"{row['backbone_non_null_gradient_count']} | {row['checkpoint_strict_reload']} | {row['peak_gpu_memory_mib']:.2f} |"
        )
    lines += ["", "Smoke metrics are pipeline checks only and are not formal results."]
    (ROOT / "reports/gate4b_training_smoke.md").write_text("\n".join(lines) + "\n")

    training_protocol = {
        "dataset": "etth1", "seq_len": 384, "pred_len": 96, "seed": 2025,
        "missing_type": "random", "missing_rate": 0.3, "train_mask_base_seed": 42025,
        "freeze_backbone": True, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "gradient_clip": 1.0, "max_epochs": 20,
        "patience": 3, "early_stopping_metric": "validation_nll",
        "test_used_for_model_selection": False, "initial_scale": 0.5, "scale_floor": 1e-4,
        "source_checkpoint": CHECKPOINT, "validation_mask": VAL_MASK, "test_mask": TEST_MASK,
    }
    metric_protocol = {
        "metric_space": "official_normalized", "crps_method": "empirical_monte_carlo",
        "crps_samples": 64, "crps_seed": 52025, "crps_chunk_elements": 65536,
        "interval_levels": [0.5, 0.8, 0.9, 0.95], "interval_type": "native_parametric",
        "conformal_calibrated": False, "posthoc_calibration": False,
    }
    train_hash, metric_hash = digest(training_protocol), digest(metric_protocol)
    tasks = []
    for variant in ("gaussian_head", "student_t_head"):
        config_rel = f"configs/gate4b/etth1_random_rate0.3_{variant}_seed2025.yaml"
        task = {
            **training_protocol, **metric_protocol, "variant": variant,
            "initial_df": None if variant == "gaussian_head" else 8.0,
            "df_parameterization": None if variant == "gaussian_head" else "2.01 + softplus(raw_df)",
            "training_protocol_hash": train_hash, "metric_protocol_hash": metric_hash,
            "required_free_mib": required, "config": config_rel,
            "max_train_batches": None, "max_val_batches": None, "max_test_batches": None,
            "command": ["gate4b_formal_runner", "--task-file", "{task_file}", "--run-dir", "{run_dir}"],
        }
        dump(ROOT / config_rel, task)
        tasks.append(task)
    queue = {
        "stage": "Gate 4b minimal probabilistic", "schema_version": 2,
        "logical_task_count": 2, "active_task_count": 2, "max_concurrent_total": 1,
        "required_free_mib": required, "scheduler_started": False, "tasks": tasks,
        "training_protocol_hash": train_hash, "metric_protocol_hash": metric_hash,
        "official_baseline_retrained": False, "gate4a_adapter": False,
        "missingness_film": False, "conformal": False, "calibration_fitting": False,
        "meta_learning": False, "test_time_adaptation": False, "gate4c": False,
    }
    dump(ROOT / "configs/queues/gate4b_minimal_probabilistic.yaml", queue)
    plan = load(ROOT / "results/gate4b_plan.json")
    plan.update({"training_smoke": "PASS", "formal_queue": "CREATED_PENDING_VALIDATION", "required_free_mib": required,
                 "training_protocol_hash": train_hash, "metric_protocol_hash": metric_hash})
    dump(ROOT / "results/gate4b_plan.json", plan)
    status_path = ROOT / "results/gate4b_status.json"
    status = load(status_path)
    status.update({"status": "PLANNING_PASS", "training_smoke": "PASS", "GATE4B_TRAINING_SMOKE_PASS": True,
                   "formal_queue": "CREATED_PENDING_VALIDATION", "scheduler": "NOT_STARTED",
                   "required_free_mib": required})
    dump(status_path, status)
    (ROOT / "reports/gate4b_status.md").write_text(
        "# Gate 4b Status\n\nTraining smoke: **PASS**\n\nFormal queue created and pending validation. Scheduler not started.\n"
    )
    print(json.dumps({"status": "PASS", "required_free_mib": required,
                      "training_protocol_hash": train_hash, "metric_protocol_hash": metric_hash}, indent=2))


if __name__ == "__main__":
    main()
