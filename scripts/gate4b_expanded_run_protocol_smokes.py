#!/usr/bin/env python
"""Run the five corrected Gate 4b expanded protocol smokes sequentially."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
TASKS = (
    ("random", 0.3, "gaussian_head"),
    ("random", 0.3, "student_t_head"),
    ("block", 0.3, "gaussian_head"),
    ("channel", 0.3, "gaussian_head"),
    ("mixed", 0.3, "gaussian_head"),
)


def config_path(mask_type, rate, variant):
    return ROOT / f"configs/gate4b_expanded_smoke/etth1_{mask_type}_rate{rate}_{variant}_seed2025.json"


def main():
    rows = []
    for mask_type, rate, variant in TASKS:
        config = config_path(mask_type, rate, variant)
        validation = subprocess.run(
            [PYTHON, str(ROOT / "scripts/gate4b_formal_runner.py"), "--task-file", str(config),
             "--validate-only", "--allow-smoke"], cwd=ROOT, text=True, capture_output=True
        )
        if validation.returncode:
            rows.append({"missing_type": mask_type, "rate": rate, "variant": variant,
                         "status": "CONFIG_ERROR", "error": validation.stdout + validation.stderr})
            break
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = (f"timeperceiver__etth1__gate4b_expanded__smoke__{variant}__"
                         f"{mask_type}_rate{rate}__seed2025__{timestamp}")
        run_dir = ROOT / "runs" / experiment_id
        command = [PYTHON, "-u", str(ROOT / "scripts/gate4b_formal_runner.py"),
                   "--task-file", str(config), "--run-dir", str(run_dir), "--allow-smoke"]
        result = subprocess.run(command, cwd=ROOT)
        metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
        passed = (result.returncode == 0 and metrics.get("train_mask_deterministic") is True
                  and metrics.get("input_mask_applied") is True and metrics.get("train_nll_finite") is True
                  and metrics.get("validation_nll_finite") is True and metrics.get("test_metrics_finite") is True
                  and metrics.get("backbone_non_null_gradient_count") == 0
                  and metrics.get("checkpoint_strict_reload") is True
                  and metrics.get("batch_y_modified") is False
                  and metrics.get("test_mask_used_during_training") is False)
        rows.append({
            "missing_type": mask_type, "rate": rate, "variant": variant,
            "status": "PASS" if passed else "FAILED", "experiment_id": experiment_id,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "deterministic": metrics.get("train_mask_deterministic"),
            "input_mask_applied": metrics.get("input_mask_applied"),
            "actual_train_missing_rate": metrics.get("actual_train_missing_rate"),
            "finite_nll": bool(metrics.get("train_nll_finite") and metrics.get("validation_nll_finite")),
            "backbone_gradients": metrics.get("backbone_non_null_gradient_count"),
            "reload": metrics.get("checkpoint_strict_reload"), "exit_code": result.returncode,
            "error": metrics.get("error", ""),
        })
        if not passed:
            break
    overall = len(rows) == len(TASKS) and all(row["status"] == "PASS" for row in rows)
    output = {"status": "PASS" if overall else "BLOCKED_EXPANDED_PROTOCOL_SMOKE_FAILURE",
              "expected": len(TASKS), "passed": sum(row["status"] == "PASS" for row in rows), "rows": rows}
    (ROOT / "results/gate4b_expanded_protocol_smoke.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["# Gate 4b Expanded Protocol Smoke", "", f"Status: **{output['status']}**", "",
             "| Missing type | Variant | Status | Deterministic | Input masked | Finite NLL | Backbone gradients | Reload |",
             "|---|---|---|---|---|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['missing_type']} | {row['variant']} | {row['status']} | {row.get('deterministic')} | "
                     f"{row.get('input_mask_applied')} | {row.get('finite_nll')} | {row.get('backbone_gradients')} | {row.get('reload')} |")
    (ROOT / "reports/gate4b_expanded_protocol_smoke.md").write_text("\n".join(lines) + "\n")
    status_path = ROOT / "results/gate4b_expanded_status.json"
    status = json.loads(status_path.read_text())
    status["protocol_smoke"] = output["status"]
    if not overall:
        status["status"] = "BLOCKED_EXPANDED_PROTOCOL_SMOKE_FAILURE"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
