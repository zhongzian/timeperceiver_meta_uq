#!/usr/bin/env python
"""Apply the predeclared Gate 4b expanded signal gates."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader(open(ROOT / "reports/gate4b_expanded_results.csv")))
    deltas = list(csv.DictReader(open(ROOT / "reports/gate4b_expanded_condition_deltas.csv")))
    probability = [row for row in rows if row["variant"] != "official"]
    implementation = (len(rows) == 18 and len(probability) == 12
                      and all(row["status"] == "SUCCESS" for row in probability)
                      and all(math.isfinite(float(row[key])) for row in probability
                              for key in ("test_mean_nll", "test_crps", "test_mse", "scale_min"))
                      and all(float(row["scale_min"]) > 0 for row in probability)
                      and all(float(row["min_df"]) > 2.01 for row in probability if row["variant"] == "student_t_head"))
    nll_count = sum(row["student_t_nll_win"] == "True" for row in deltas)
    crps_count = sum(row["student_t_crps_noninferior"] == "True" for row in deltas)
    point_count = sum(row["student_t_point_noninferior"] == "True" for row in deltas)
    signal = implementation and nll_count >= 4 and crps_count >= 4 and point_count >= 5
    if not implementation:
        decision = "EXPANDED_IMPLEMENTATION_INCOMPLETE"
    elif point_count < 5:
        decision = "EXPANDED_PROBABILISTIC_POINT_TRADEOFF"
    elif signal:
        decision = "STUDENT_T_EXPANDED_SIGNAL_POSITIVE"
    elif nll_count >= 4 and crps_count < 4:
        decision = "EXPANDED_TAIL_LIKELIHOOD_ONLY"
    else:
        decision = "STUDENT_T_CONDITION_SPECIFIC_SIGNAL"
    output = {
        "status": decision, "GATE4B_EXPANDED_IMPLEMENTATION_PASS": implementation,
        "GATE4B_EXPANDED_NLL_WIN_COUNT": nll_count,
        "GATE4B_EXPANDED_CRPS_NONINFERIOR_COUNT": crps_count,
        "GATE4B_EXPANDED_POINT_NONINFERIOR_COUNT": point_count,
        "GATE4B_EXPANDED_SIGNAL_POSITIVE": signal,
        "ENTER_GATE4B_CROSS_DATASET_MATRIX": implementation and signal,
        "ENTER_GATE4C": False, "seed_count": 1,
    }
    (ROOT / "results/gate4b_expanded_formal_decision.json").write_text(json.dumps(output, indent=2) + "\n")
    with open(ROOT / "reports/gate4b_expanded_formal_decision.csv", "w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(output); writer.writerow(output.values())
    (ROOT / "reports/gate4b_expanded_formal_decision.md").write_text(
        f"# Gate 4b Expanded Formal Decision\n\nDecision: **{decision}**\n\n"
        f"- NLL wins: `{nll_count}/6`\n- CRPS noninferior: `{crps_count}/6`\n"
        f"- Point noninferior: `{point_count}/6`\n- Enter cross-dataset: `{str(output['ENTER_GATE4B_CROSS_DATASET_MATRIX']).lower()}`\n"
        "- Enter Gate 4c: `false`\n\nSingle seed only; no significance claim.\n")
    status_path = ROOT / "results/gate4b_expanded_status.json"
    status = json.loads(status_path.read_text())
    status.update(output); status.update({"expanded_queue": "COMPLETE", "scheduler": "COMPLETE"})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    (ROOT / "reports/gate4b_expanded_status.md").write_text(
        f"# Gate 4b Expanded Status\n\nDecision: **{decision}**\n\nAll 12 probability tasks completed or were legally reused.\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
