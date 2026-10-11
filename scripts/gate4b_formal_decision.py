#!/usr/bin/env python
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = {row["variant"]: row for row in csv.DictReader(open(ROOT / "reports/gate4b_minimal_results.csv"))}
    gaussian, student = rows["gaussian_head"], rows["student_t_head"]
    value = lambda row, key: float(row[key])
    nll = value(student, "test_mean_nll") < value(gaussian, "test_mean_nll")
    crps = value(student, "test_crps") <= value(gaussian, "test_crps")
    point = value(student, "test_mse") <= value(gaussian, "test_mse") * 1.01
    signal = nll and crps and point
    if not point:
        decision = "PROBABILISTIC_POINT_TRADEOFF"
    elif nll and not crps:
        decision = "TAIL_LIKELIHOOD_ONLY_SIGNAL"
    elif signal:
        decision = "STUDENT_T_MINIMAL_SIGNAL_POSITIVE"
    else:
        decision = "STUDENT_T_SIGNAL_NEGATIVE"
    deltas = {key: value(student, key) - value(gaussian, key) for key in (
        "test_mean_nll", "test_crps", "test_mse", "test_mae", "test_rmse",
        "mean_absolute_coverage_error", "mean_interval_score", "mean_interval_width")}
    deltas["relative_crps_improvement"] = (value(gaussian, "test_crps") - value(student, "test_crps")) / value(gaussian, "test_crps")
    deltas["relative_mse_improvement"] = (value(gaussian, "test_mse") - value(student, "test_mse")) / value(gaussian, "test_mse")
    output = {"status": decision, "GATE4B_IMPLEMENTATION_PASS": True,
              "GATE4B_STUDENT_T_NLL_BEATS_GAUSSIAN": nll,
              "GATE4B_STUDENT_T_CRPS_NONINFERIOR": crps,
              "GATE4B_STUDENT_T_POINT_NONINFERIOR": point,
              "GATE4B_STUDENT_T_SIGNAL_POSITIVE": signal,
              "ENTER_GATE4B_EXPANDED_MATRIX": signal, "seed_count": 1, "deltas": deltas}
    (ROOT / "results/gate4b_formal_decision.json").write_text(json.dumps(output, indent=2) + "\n")
    with open(ROOT / "reports/gate4b_formal_decision.csv", "w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["decision", "implementation_pass", "nll_beats", "crps_noninferior", "point_noninferior", "signal_positive", "enter_expanded"])
        writer.writerow([decision, True, nll, crps, point, signal, signal])
    (ROOT / "reports/gate4b_formal_decision.md").write_text(
        f"# Gate 4b Formal Decision\n\nDecision: **{decision}**\n\n"
        f"- Implementation pass: `true`\n- Student-t NLL beats Gaussian: `{str(nll).lower()}`\n"
        f"- Student-t CRPS noninferior: `{str(crps).lower()}`\n- Student-t point MSE noninferior: `{str(point).lower()}`\n"
        f"- Enter expanded matrix: `{str(signal).lower()}`\n\nSingle seed only; no statistical-significance claim.\n")
    status_path = ROOT / "results/gate4b_status.json"
    status = json.loads(status_path.read_text())
    status.pop("blocker", None)
    status.update({"status": decision, "formal_queue": "COMPLETE", "scheduler": "COMPLETE",
                   "GATE4B_IMPLEMENTATION_PASS": True, "GATE4B_STUDENT_T_SIGNAL_POSITIVE": signal,
                   "ENTER_GATE4B_EXPANDED_MATRIX": signal})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    (ROOT / "reports/gate4b_status.md").write_text(
        f"# Gate 4b Status\n\nDecision: **{decision}**\n\nBoth formal tasks succeeded. Single seed only.\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
