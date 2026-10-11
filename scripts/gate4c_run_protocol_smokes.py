#!/usr/bin/env python
"""Run and summarize the two Gate 4c GPU protocol smokes."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / f"logs/gate4c_protocol_smoke_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    with log_path.open("w") as log:
        for index, variant in enumerate(("gaussian_head", "student_t_head")):
            if index:
                time.sleep(1)
            run_id = (
                f"timeperceiver__etth1__gate4c_smoke__{variant}__"
                f"random_rate0.3__seed2025__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            run = ROOT / "runs" / run_id
            command = [
                sys.executable, "-u", str(ROOT / "scripts/gate4c_train_and_calibrate.py"),
                "--variant", variant, "--smoke", "--run-dir", str(run), "--device", "cuda",
            ]
            log.write("COMMAND " + " ".join(command) + "\n")
            log.flush()
            return_code = subprocess.call(command, cwd=str(ROOT), env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            metrics = load(run / "metrics.json") if (run / "metrics.json").exists() else {"error": "metrics missing"}
            status = load(run / "status.json") if (run / "status.json").exists() else {"status": "FAILED"}
            rows[variant] = {
                "status": status.get("status"),
                "return_code": return_code,
                "run_dir": str(run.relative_to(ROOT)),
                "train_batches": metrics.get("train_batches"),
                "selection_batches": metrics.get("selection_batches"),
                "calibration_batches": metrics.get("calibration_batches"),
                "test_batches": metrics.get("test_batches"),
                "train_nll": metrics.get("train_nll"),
                "selection_nll": metrics.get("selection_nll"),
                "q_hat_finite_nonnegative": metrics.get("q_hat_finite_nonnegative"),
                "conditioned_bins_valid": metrics.get("conditioned_bins_valid"),
                "point_prediction_unchanged": metrics.get("point_prediction_mutation") is False,
                "checkpoint_strict_reload": metrics.get("checkpoint_strict_reload"),
                "interval_nesting": metrics.get("interval_nesting"),
                "backbone_non_null_gradient_count": metrics.get("backbone_non_null_gradient_count"),
                "optimizer_backbone_parameter_count": metrics.get("optimizer_backbone_parameter_count"),
                "calibration_loader_call_count_before_best_checkpoint": metrics.get("calibration_loader_call_count_before_best_checkpoint"),
                "test_loader_call_count_before_qhat": metrics.get("test_loader_call_count_before_qhat"),
                "batch_y_modified": metrics.get("batch_y_modified"),
                "test_mask_used_during_training": metrics.get("test_mask_used_during_training"),
                "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
                "error": metrics.get("error", ""),
            }
    passed = len(rows) == 2 and all(
        row["status"] == "SUCCESS"
        and row["return_code"] == 0
        and row["q_hat_finite_nonnegative"] is True
        and row["conditioned_bins_valid"] is True
        and row["point_prediction_unchanged"] is True
        and row["checkpoint_strict_reload"] is True
        and row["backbone_non_null_gradient_count"] == 0
        and row["optimizer_backbone_parameter_count"] == 0
        and row["calibration_loader_call_count_before_best_checkpoint"] == 0
        and row["test_loader_call_count_before_qhat"] == 0
        and row["batch_y_modified"] is False
        and row["test_mask_used_during_training"] is False
        for row in rows.values()
    )
    peaks = [float(row["peak_gpu_memory_mib"]) for row in rows.values() if row["peak_gpu_memory_mib"] is not None]
    required = max(4096, int(math.ceil(max(peaks) * 1.5)) + 512) if peaks else None
    output = {
        "stage": "Gate 4c GPU protocol smoke",
        "status": "PASS" if passed else "BLOCKED_GATE4C_PROTOCOL_SMOKE_FAILURE",
        "GATE4C_PROTOCOL_SMOKE_PASS": passed,
        "variants": rows,
        "smoke_log": str(log_path.relative_to(ROOT)),
        "peak_formula": "max(4096, ceil(max_gate4c_smoke_peak_mib * 1.5) + 512)",
        "required_free_mib": required,
    }
    (ROOT / "results/gate4c_training_smoke.json").write_text(json.dumps(output, indent=2) + "\n")
    md = [
        "# Gate 4c GPU Protocol Smoke", "", f"Status: **{output['status']}**", "",
        "| Head | Status | Train NLL | Selection NLL | q-hat | Bins | Point unchanged | Reload | Peak MiB |",
        "|---|---|---:|---:|---|---|---|---|---:|",
    ]
    for variant, row in rows.items():
        md.append(
            f"| {variant} | {row['status']} | {row['train_nll']} | {row['selection_nll']} | "
            f"{row['q_hat_finite_nonnegative']} | {row['conditioned_bins_valid']} | "
            f"{row['point_prediction_unchanged']} | {row['checkpoint_strict_reload']} | {row['peak_gpu_memory_mib']} |"
        )
    md += ["", f"Formal queue required free memory: `{required}` MiB."]
    (ROOT / "reports/gate4c_training_smoke.md").write_text("\n".join(md) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if passed else 2
if __name__ == "__main__":
    raise SystemExit(main())
