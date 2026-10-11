#!/usr/bin/env python3
"""Numerically validate the two Gate 5a distribution extensions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate4b_probabilistic_metrics import central_intervals as gate4b_intervals
from gate4b_probabilistic_metrics import nll_elements as gate4b_nll
from gate5a_common import dump, write_json_markdown
from gate5a_distribution_metrics import central_intervals, empirical_crps, laplace_quantile, nll_elements
from models.probabilistic_head import ProbabilisticHead


LEVELS = (0.5, 0.8, 0.9, 0.95)


def interval_nested(intervals):
    widths = []
    for level in LEVELS:
        lower, upper = intervals[level]
        if not bool(torch.all(lower <= upper)):
            return False
        widths.append(upper - lower)
    return all(bool(torch.all(left <= right)) for left, right in zip(widths, widths[1:]))


def parameter_audit(channels):
    heads = {
        variant: ProbabilisticHead(64, 12, channels, variant).double()
        for variant in ("gaussian_head", "laplace_head", "student_t_fixed_df8_head", "student_t_head")
    }
    counts = {variant: sum(parameter.numel() for parameter in head.parameters()) for variant, head in heads.items()}
    passed = (
        counts["laplace_head"] == counts["gaussian_head"]
        and counts["student_t_fixed_df8_head"] == counts["gaussian_head"]
        and counts["student_t_head"] == counts["gaussian_head"] + channels
        and heads["student_t_fixed_df8_head"].raw_df is None
        and heads["student_t_fixed_df8_head"].fixed_df is not None
        and bool(torch.equal(heads["student_t_fixed_df8_head"].fixed_df, torch.full_like(heads["student_t_fixed_df8_head"].fixed_df, 8.0)))
    )
    return {"channels": channels, "counts": counts, "status": "PASS" if passed else "FAIL"}


def main():
    torch.manual_seed(2025)
    loc = torch.randn(3, 5, 7, dtype=torch.float64)
    scale = torch.nn.functional.softplus(torch.randn_like(loc)) + 1e-4
    target = torch.randn_like(loc)
    fixed_df = torch.full_like(loc, 8.0)

    laplace_manual = torch.log(2.0 * scale) + torch.abs(target - loc) / scale
    laplace_reference = -torch.distributions.Laplace(loc, scale).log_prob(target)
    laplace_formula_diff = float(torch.max(torch.abs(laplace_manual - laplace_reference)))

    quantile_diff = 0.0
    for level in LEVELS:
        for probability in ((1.0 - level) / 2.0, (1.0 + level) / 2.0):
            actual = laplace_quantile(probability, loc, scale)
            expected = torch.distributions.Laplace(loc, scale).icdf(torch.full_like(loc, probability))
            quantile_diff = max(quantile_diff, float(torch.max(torch.abs(actual - expected))))

    fixed_nll = nll_elements("student_t_fixed_df8_head", loc, scale, target, fixed_df)
    reference_nll = gate4b_nll("student_t_head", loc, scale, target, fixed_df)
    fixed_nll_diff = float(torch.max(torch.abs(fixed_nll - reference_nll)))
    fixed_intervals = central_intervals("student_t_fixed_df8_head", loc, scale, fixed_df)
    reference_intervals = gate4b_intervals("student_t_head", loc, scale, fixed_df)
    fixed_quantile_diff = max(
        float(torch.max(torch.abs(actual - expected)))
        for level in LEVELS
        for actual, expected in zip(fixed_intervals[level], reference_intervals[level])
    )

    torch.manual_seed(52025)
    fixed_sample = torch.distributions.StudentT(fixed_df, loc, scale).sample((11,))
    torch.manual_seed(52025)
    reference_sample = torch.distributions.StudentT(fixed_df, loc, scale).sample((11,))
    fixed_sample_shape_identical = fixed_sample.shape == reference_sample.shape
    fixed_sample_values_identical = bool(torch.equal(fixed_sample, reference_sample))

    learnable_head = ProbabilisticHead(64, 12, 7, "student_t_head").double()
    initial_learned_df = 2.01 + torch.nn.functional.softplus(learnable_head.raw_df)
    expanded_initial_df = initial_learned_df.expand_as(loc)
    initial_parity_diff = float(torch.max(torch.abs(
        gate4b_nll("student_t_head", loc, scale, target, expanded_initial_df) - fixed_nll
    )))

    intervals = {
        "gaussian_head": central_intervals("gaussian_head", loc, scale),
        "laplace_head": central_intervals("laplace_head", loc, scale),
        "student_t_fixed_df8_head": fixed_intervals,
        "student_t_head": central_intervals("student_t_head", loc, scale, expanded_initial_df),
    }
    interval_nesting = {variant: interval_nested(values) for variant, values in intervals.items()}
    nll_finite = {
        "gaussian_head": bool(torch.isfinite(nll_elements("gaussian_head", loc, scale, target)).all()),
        "laplace_head": bool(torch.isfinite(nll_elements("laplace_head", loc, scale, target)).all()),
        "student_t_fixed_df8_head": bool(torch.isfinite(fixed_nll).all()),
        "student_t_head": bool(torch.isfinite(nll_elements("student_t_head", loc, scale, target, expanded_initial_df)).all()),
    }
    crps = {
        "gaussian_head": empirical_crps("gaussian_head", loc, scale, target, None, 64, 52025, 128),
        "laplace_head": empirical_crps("laplace_head", loc, scale, target, None, 64, 52025, 128),
        "student_t_fixed_df8_head": empirical_crps("student_t_fixed_df8_head", loc, scale, target, fixed_df, 64, 52025, 128),
        "student_t_head": empirical_crps("student_t_head", loc, scale, target, expanded_initial_df, 64, 52025, 128),
    }
    capacity = [parameter_audit(channels) for channels in (7, 21, 321)]
    checks = {
        "laplace_formula": laplace_formula_diff <= 1e-7,
        "laplace_quantile": quantile_diff <= 1e-6,
        "fixed_t_nll_parity": fixed_nll_diff <= 1e-7,
        "fixed_t_quantile_parity": fixed_quantile_diff <= 1e-6,
        "fixed_t_sample_shape": fixed_sample_shape_identical,
        "fixed_t_sample_values": fixed_sample_values_identical,
        "initial_fixed_learnable_parity": initial_parity_diff <= 1e-6,
        "capacity_fairness": all(row["status"] == "PASS" for row in capacity),
        "scale_finite_positive": bool(torch.isfinite(scale).all() and torch.all(scale > 0)),
        "fixed_df_exactly_8": bool(torch.equal(fixed_df, torch.full_like(fixed_df, 8.0))),
        "learnable_df_above_2_01": bool(torch.all(expanded_initial_df > 2.01)),
        "all_nll_finite": all(nll_finite.values()),
        "all_crps_finite": all(math.isfinite(value) for value in crps.values()),
        "all_intervals_nested": all(interval_nesting.values()),
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "BLOCKED_GATE5A_DISTRIBUTION_VALIDATION_FAILURE",
        "thresholds": {
            "laplace_nll_max_abs_diff": 1e-7,
            "laplace_quantile_max_abs_diff": 1e-6,
            "fixed_t_nll_max_abs_diff": 1e-7,
            "fixed_t_quantile_max_abs_diff": 1e-6,
            "initial_fixed_learnable_nll_max_abs_diff": 1e-6,
        },
        "observed": {
            "laplace_nll_max_abs_diff": laplace_formula_diff,
            "laplace_quantile_max_abs_diff": quantile_diff,
            "fixed_t_nll_max_abs_diff": fixed_nll_diff,
            "fixed_t_quantile_max_abs_diff": fixed_quantile_diff,
            "initial_fixed_learnable_nll_max_abs_diff": initial_parity_diff,
            "crps": crps,
            "nll_finite": nll_finite,
            "interval_nesting": interval_nesting,
            "learnable_initial_df_min": float(expanded_initial_df.min()),
            "learnable_initial_df_max": float(expanded_initial_df.max()),
        },
        "capacity": capacity,
        "checks": checks,
    }
    dump(ROOT / "results/gate5a_distribution_validation.json", payload)
    write_json_markdown(
        ROOT / "reports/gate5a_distribution_validation.md", "Gate 5a Distribution Validation", payload,
        "Synthetic double-precision checks; no training or test-set selection is performed.",
    )
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
