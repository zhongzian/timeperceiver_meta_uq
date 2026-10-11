#!/usr/bin/env python
"""Decide the corrected fixed-mask random-0.3 Gate 4b entry gate."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def latest(variant):
    pattern = f"timeperceiver__etth1__gate4b_corrected_minimal__{variant}__random_rate0.3__seed2025__*/status.json"
    runs = []
    for path in (ROOT / "runs").glob(pattern):
        status, metrics = load(path), load(path.parent / "metrics.json")
        if status.get("status") == "SUCCESS" and metrics.get("input_mask_applied") is True:
            runs.append((path.parent, metrics))
    if not runs:
        raise RuntimeError(f"no corrected run for {variant}")
    return sorted(runs)[-1]


def main():
    gaussian_run, gaussian = latest("gaussian_head")
    student_run, student = latest("student_t_head")
    nll = student["test_mean_nll"] < gaussian["test_mean_nll"]
    crps = student["test_crps"] <= gaussian["test_crps"]
    point = student["test_mse"] <= gaussian["test_mse"] * 1.01
    signal = nll and crps and point
    output = {
        "status": "STUDENT_T_MINIMAL_SIGNAL_POSITIVE" if signal else "STUDENT_T_SIGNAL_NEGATIVE",
        "corrected_fixed_mask_protocol": True, "historical_unmasked_results_valid": False,
        "gaussian_run": str(gaussian_run.relative_to(ROOT)), "student_t_run": str(student_run.relative_to(ROOT)),
        "gaussian": {key: gaussian[key] for key in ("best_epoch", "best_validation_nll", "test_mean_nll", "test_crps", "test_mse")},
        "student_t": {key: student[key] for key in ("best_epoch", "best_validation_nll", "test_mean_nll", "test_crps", "test_mse")},
        "GATE4B_STUDENT_T_NLL_BEATS_GAUSSIAN": nll,
        "GATE4B_STUDENT_T_CRPS_NONINFERIOR": crps,
        "GATE4B_STUDENT_T_POINT_NONINFERIOR": point,
        "GATE4B_STUDENT_T_SIGNAL_POSITIVE": signal,
        "ENTER_GATE4B_EXPANDED_MATRIX": signal,
    }
    (ROOT / "results/gate4b_corrected_minimal_decision.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "reports/gate4b_corrected_minimal_decision.md").write_text(
        f"# Gate 4b Corrected Minimal Decision\n\nStatus: **{output['status']}**\n\n"
        "The shared runner applies `batch_x * mask`. Historical unmasked minimal runs are not evidence.\n\n"
        f"- Gaussian NLL/CRPS/MSE: `{gaussian['test_mean_nll']:.6f}` / `{gaussian['test_crps']:.6f}` / `{gaussian['test_mse']:.6f}`\n"
        f"- Student-t NLL/CRPS/MSE: `{student['test_mean_nll']:.6f}` / `{student['test_crps']:.6f}` / `{student['test_mse']:.6f}`\n"
        f"- Enter expanded matrix: `{str(signal).lower()}`\n")
    status_path = ROOT / "results/gate4b_status.json"
    status = load(status_path)
    status.update({"status": output["status"], "corrected_fixed_mask_protocol": True,
                   "historical_unmasked_minimal_valid": False,
                   "GATE4B_STUDENT_T_SIGNAL_POSITIVE": signal, "ENTER_GATE4B_EXPANDED_MATRIX": signal})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if signal else 2


if __name__ == "__main__":
    raise SystemExit(main())
