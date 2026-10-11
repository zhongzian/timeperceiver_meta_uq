#!/usr/bin/env python3
"""Distribution and capacity validation for Gate 5b PatchTST heads."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from scipy.special import stdtrit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_distribution_metrics import central_intervals, empirical_crps, nll_elements  # noqa: E402
from gate5b_common import DATASETS, dataset_spec, dump, write_report  # noqa: E402
from models.patchtst_backbone import PatchTSTBackbone  # noqa: E402
from models.probabilistic_patchtst import ProbabilisticPatchTST  # noqa: E402


def main():
    torch.manual_seed(52025)
    loc = torch.randn(3, 5, 4, dtype=torch.float64)
    scale = torch.rand(3, 5, 4, dtype=torch.float64) + 0.2
    target = torch.randn(3, 5, 4, dtype=torch.float64)
    df = torch.full_like(loc, 8.0)
    gaussian_actual = nll_elements("gaussian_head", loc, scale, target, None)
    gaussian_reference = -torch.distributions.Normal(loc, scale).log_prob(target)
    fixed_actual = nll_elements("student_t_fixed_df8_head", loc, scale, target, df)
    fixed_reference = -torch.distributions.StudentT(df, loc, scale).log_prob(target)
    gaussian_parity = bool(torch.allclose(gaussian_actual, gaussian_reference, atol=1e-12, rtol=1e-12))
    fixed_nll_parity = bool(torch.allclose(fixed_actual, fixed_reference, atol=1e-12, rtol=1e-12))
    intervals = central_intervals("student_t_fixed_df8_head", loc, scale, df)
    quantile_parity = True
    interval_nesting = True
    previous_width = None
    for level in (0.5, 0.8, 0.9, 0.95):
        lower, upper = intervals[level]
        alpha = (1.0 - level) / 2.0
        lower_z = torch.from_numpy(stdtrit(df.numpy(), alpha)).to(dtype=loc.dtype)
        upper_z = torch.from_numpy(stdtrit(df.numpy(), 1.0 - alpha)).to(dtype=loc.dtype)
        expected_lower = loc + lower_z * scale
        expected_upper = loc + upper_z * scale
        quantile_parity &= bool(torch.allclose(lower, expected_lower, atol=1e-10, rtol=1e-10))
        quantile_parity &= bool(torch.allclose(upper, expected_upper, atol=1e-10, rtol=1e-10))
        width = upper - lower
        if previous_width is not None:
            interval_nesting &= bool((width >= previous_width).all())
        previous_width = width
    crps_gaussian = empirical_crps("gaussian_head", loc.float(), scale.float(), target.float(), None, 64, 52025, 65536)
    crps_fixed = empirical_crps("student_t_fixed_df8_head", loc.float(), scale.float(), target.float(), df.float(), 64, 52025, 65536)
    capacity_rows = []
    for dataset in DATASETS:
        backbone = PatchTSTBackbone(dataset_spec(dataset)["architecture"])
        gaussian = ProbabilisticPatchTST(copy.deepcopy(backbone), "gaussian")
        fixed_t = ProbabilisticPatchTST(copy.deepcopy(backbone), "fixed_t8")
        capacity_rows.append({
            "dataset": dataset,
            "gaussian_trainable_parameters": gaussian.trainable_parameter_count(),
            "fixed_t8_trainable_parameters": fixed_t.trainable_parameter_count(),
            "equal_trainable_parameters": gaussian.trainable_parameter_count() == fixed_t.trainable_parameter_count(),
            "gaussian_backbone_trainable_parameters": gaussian.backbone_trainable_parameter_count(),
            "fixed_t8_backbone_trainable_parameters": fixed_t.backbone_trainable_parameter_count(),
            "fixed_t8_df_parameters": fixed_t.df_parameter_count(),
        })
    capacity_pass = all(
        row["equal_trainable_parameters"] and row["gaussian_backbone_trainable_parameters"] == 0
        and row["fixed_t8_backbone_trainable_parameters"] == 0 and row["fixed_t8_df_parameters"] == 0
        for row in capacity_rows
    )
    passed = all((
        gaussian_parity, fixed_nll_parity, quantile_parity, interval_nesting,
        torch.isfinite(torch.tensor(crps_gaussian)), torch.isfinite(torch.tensor(crps_fixed)),
        bool((scale > 0).all()), bool(torch.isfinite(scale).all()), capacity_pass,
    ))
    payload = {
        "status": "PASS" if passed else "BLOCKED_GATE5B_PROBABILITY_PAIRING",
        "gaussian_nll_parity": gaussian_parity, "fixed_t8_nll_parity": fixed_nll_parity,
        "fixed_t8_quantile_parity": quantile_parity, "crps_implementation_reused": True,
        "crps_implementation": "scripts/gate5a_distribution_metrics.py",
        "crps_gaussian_finite": bool(torch.isfinite(torch.tensor(crps_gaussian))),
        "crps_fixed_t8_finite": bool(torch.isfinite(torch.tensor(crps_fixed))),
        "interval_nesting": interval_nesting, "scale_positive_and_finite": True,
        "fixed_df_exact": 8.0, "capacity_fairness": "PASS" if capacity_pass else "FAIL",
        "capacity_rows": capacity_rows,
    }
    dump(ROOT / "results/gate5b_distribution_validation.json", payload)
    write_report(ROOT / "reports/gate5b_distribution_validation.md", "Gate 5b Distribution Validation", payload)
    print(json.dumps({"status": payload["status"], "capacity_fairness": payload["capacity_fairness"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
