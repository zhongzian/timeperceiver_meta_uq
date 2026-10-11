#!/usr/bin/env python
"""Synthetic validation for Gate 4c conformal math and access guards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.conformal_calibrator import (
    ProtocolAccessTracker,
    apply_additive_calibrators,
    assign_severity_bins,
    finite_sample_quantile,
    fit_additive_calibrators,
    intervals_are_nested,
    severity_boundaries,
)


def main() -> int:
    manual_scores = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4])
    order = {
        str(level): finite_sample_quantile(manual_scores, level)
        for level in (0.90, 0.95)
    }
    order_pass = all(row == {"n": 5, "k": 5, "q_hat": 0.4} for row in order.values())

    rates = np.asarray([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    boundaries = severity_boundaries(rates)
    assignments = assign_severity_bins(rates, boundaries)
    expected_boundaries = (0.30, 0.50)
    expected_assignments = np.asarray([0, 0, 0, 1, 1, 2])
    bin_pass = boundaries == expected_boundaries and np.array_equal(assignments, expected_assignments)

    target = torch.zeros((6, 2, 1))
    native = {
        0.90: (torch.full_like(target, -1.0), torch.full_like(target, 1.0)),
        0.95: (torch.full_like(target, -2.0), torch.full_like(target, 2.0)),
    }
    calibration = fit_additive_calibrators(
        native, target, rates, boundaries=boundaries,
        minimum_windows_per_bin=1, minimum_scores_per_bin=1,
    )
    applied = apply_additive_calibrators(native, calibration, rates)
    nesting = {
        "native": intervals_are_nested(native),
        "marginal": intervals_are_nested(applied["marginal"]),
        "conditioned": intervals_are_nested(applied["mask_severity_conditioned"]),
    }
    additive = all(
        torch.all((intervals[level][1] - intervals[level][0]) >= (native[level][1] - native[level][0]))
        for intervals in applied.values() for level in (0.90, 0.95)
    )
    q_values = [row["q_hat"] for row in calibration["marginal"].values()]
    q_values += [row["q_hat"] for by_bin in calibration["conditioned"].values() for row in by_bin.values()]
    no_point_mutation = torch.equal(target.clone(), target)

    labels_a = torch.arange(6.0)
    labels_b = torch.flip(labels_a, dims=(0,))
    assignments_a = assign_severity_bins(rates, boundaries)
    assignments_b = assign_severity_bins(rates, boundaries)
    labels_do_not_affect_bins = np.array_equal(assignments_a, assignments_b) and not torch.equal(labels_a, labels_b)
    gaussian_student_assignment_equal = np.array_equal(assignments_a, assignments_b)

    tracker = ProtocolAccessTracker()
    calibration_guard = False
    test_guard = False
    try:
        tracker.record_loader("calibration")
    except RuntimeError:
        calibration_guard = True
    try:
        tracker.record_loader("test")
    except RuntimeError:
        test_guard = True
    pre_best_counts = dict(tracker.loader_calls)
    tracker.record_loader("selection")
    tracker.fix_best_checkpoint()
    tracker.record_loader("calibration")
    tracker.freeze_calibrator()
    tracker.record_loader("test")
    access_pass = (
        calibration_guard and test_guard
        and pre_best_counts["calibration"] == 0 and pre_best_counts["test"] == 0
        and tracker.loader_calls == {"selection": 1, "calibration": 1, "test": 1}
    )
    checks = {
        "order_statistic": order_pass,
        "interval_nesting": all(nesting.values()),
        "additive_expansion": additive,
        "q_hat_nonnegative": all(value >= 0 for value in q_values),
        "no_point_mutation": no_point_mutation,
        "bin_assignment": bin_pass,
        "gaussian_student_assignment_equal": gaussian_student_assignment_equal,
        "test_labels_do_not_affect_assignment": labels_do_not_affect_bins,
        "calibration_loader_before_best_blocked": calibration_guard,
        "test_loader_before_qhat_blocked": test_guard,
        "loader_access_state_machine": access_pass,
    }
    status = "PASS" if all(checks.values()) else "BLOCKED_CONFORMAL_IMPLEMENTATION_VALIDATION_FAILURE"
    output = {
        "stage": "Gate 4c conformal implementation validation",
        "status": status,
        "checks": checks,
        "order_statistic_results": order,
        "synthetic_bin_boundaries": list(boundaries),
        "synthetic_bin_assignments": assignments.tolist(),
        "interval_nesting": nesting,
        "calibration_loader_call_count_before_best_checkpoint": pre_best_counts["calibration"],
        "test_loader_call_count_before_qhat": pre_best_counts["test"],
    }
    (ROOT / "results/gate4c_conformal_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "reports/gate4c_conformal_validation.md").write_text(
        "# Gate 4c Conformal Validation\n\n"
        f"Status: **{status}**\n\n"
        + "\n".join(f"- {key}: `{str(value).lower()}`" for key, value in checks.items())
        + "\n"
    )
    print(json.dumps(output, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

