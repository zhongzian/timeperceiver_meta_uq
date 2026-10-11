#!/usr/bin/env python3
"""Apply the predeclared Gate 5a formal decision rules."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_common import dump, load, write_csv, write_json_markdown


def counts(rows, comparison):
    selected = [row for row in rows if row["comparison"] == comparison]
    truth = lambda value: str(value).lower() == "true"
    return {
        "cells": len(selected),
        "nll_wins": sum(truth(row["nll_win"]) for row in selected),
        "crps_noninferior": sum(truth(row["crps_noninferior"]) for row in selected),
        "point_noninferior": sum(truth(row["point_noninferior"]) for row in selected),
    }


def signal(value):
    return value["cells"] == 3 and value["nll_wins"] >= 2 and value["crps_noninferior"] >= 2 and value["point_noninferior"] == 3


def main():
    status = load(ROOT / "results/gate5a_status.json")
    with (ROOT / "reports/gate5a_pairwise_deltas.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    laplace = counts(rows, "Laplace vs Gaussian")
    fixed = counts(rows, "Fixed-t vs Gaussian")
    learned = counts(rows, "Learnable-t vs Fixed-t")
    learned_laplace = counts(rows, "Learnable-t vs Laplace")
    implementation = status.get("GATE5A_IMPLEMENTATION_PASS") is True
    laplace_positive = signal(laplace)
    fixed_positive = signal(fixed)
    learned_positive = signal(learned)
    overall = implementation and (laplace_positive or fixed_positive) and learned_positive
    if overall:
        decision = "ADAPTIVE_STUDENT_T_DISTRIBUTION_ABLATION_POSITIVE"
        ready = True
    elif implementation and fixed_positive and not learned_positive:
        decision = "FIXED_HEAVY_TAIL_SUFFICIENT"
        ready = True
    elif implementation and not laplace_positive and not fixed_positive and learned_positive:
        decision = "LEARNABLE_STUDENT_T_SPECIFIC_SIGNAL"
        ready = True
    else:
        decision = "DISTRIBUTION_HEAD_SIGNAL_INCONCLUSIVE"
        ready = False
    payload = {
        "status": "PASS" if implementation else "FAILED",
        "GATE5A_IMPLEMENTATION_PASS": implementation,
        "GATE5A_LAPLACE_NLL_WIN_COUNT": laplace["nll_wins"],
        "GATE5A_LAPLACE_CRPS_NONINFERIOR_COUNT": laplace["crps_noninferior"],
        "GATE5A_LAPLACE_POINT_NONINFERIOR_COUNT": laplace["point_noninferior"],
        "GATE5A_LAPLACE_SIGNAL_POSITIVE": laplace_positive,
        "GATE5A_FIXED_T_NLL_WIN_COUNT": fixed["nll_wins"],
        "GATE5A_FIXED_T_CRPS_NONINFERIOR_COUNT": fixed["crps_noninferior"],
        "GATE5A_FIXED_T_POINT_NONINFERIOR_COUNT": fixed["point_noninferior"],
        "GATE5A_FIXED_T_SIGNAL_POSITIVE": fixed_positive,
        "GATE5A_LEARNED_DF_NLL_WIN_COUNT": learned["nll_wins"],
        "GATE5A_LEARNED_DF_CRPS_NONINFERIOR_COUNT": learned["crps_noninferior"],
        "GATE5A_LEARNED_DF_POINT_NONINFERIOR_COUNT": learned["point_noninferior"],
        "GATE5A_LEARNED_DF_SIGNAL_POSITIVE": learned_positive,
        "GATE5A_DISTRIBUTION_ABLATION_SUPPORTS_MAIN_CLAIM": overall,
        "READY_FOR_SECOND_BACKBONE_PLANNING": ready,
        "decision": decision,
        "learnable_t_vs_laplace": learned_laplace,
        "statistical_significance_claim": False,
    }
    dump(ROOT / "results/gate5a_formal_decision.json", payload)
    write_csv(ROOT / "reports/gate5a_formal_decision.csv", [payload])
    write_json_markdown(ROOT / "reports/gate5a_formal_decision.md", "Gate 5a Formal Decision", payload)
    print(json.dumps(payload, indent=2))
    return 0 if implementation else 2


if __name__ == "__main__":
    raise SystemExit(main())
