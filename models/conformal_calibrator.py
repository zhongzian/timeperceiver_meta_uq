"""Leakage-safe additive split-conformal utilities for Gate 4c."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch


LEVELS = (0.90, 0.95)
STAGE_ORDER = (
    "TRAIN",
    "MODEL_SELECTION",
    "LOAD_BEST",
    "FREEZE_MODEL",
    "CALIBRATION_FIT",
    "FREEZE_CALIBRATOR",
    "TEST_EVALUATION",
)


class ProtocolAccessTracker:
    """Fail closed if calibration/test data is touched before its protocol boundary."""

    def __init__(self) -> None:
        self.best_checkpoint_fixed = False
        self.calibrator_frozen = False
        self.loader_calls = {"selection": 0, "calibration": 0, "test": 0}

    def fix_best_checkpoint(self) -> None:
        self.best_checkpoint_fixed = True

    def freeze_calibrator(self) -> None:
        if not self.best_checkpoint_fixed:
            raise RuntimeError("calibrator cannot be frozen before best checkpoint is fixed")
        self.calibrator_frozen = True

    def record_loader(self, name: str) -> None:
        if name not in self.loader_calls:
            raise ValueError(f"unknown loader: {name}")
        if name == "calibration" and not self.best_checkpoint_fixed:
            raise RuntimeError("calibration loader accessed before best checkpoint was fixed")
        if name == "test" and not self.calibrator_frozen:
            raise RuntimeError("test loader accessed before conformal quantiles were frozen")
        self.loader_calls[name] += 1


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> Dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("conformal scores must be nonempty, finite, and nonnegative")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"invalid coverage: {coverage}")
    n = int(values.size)
    k = min(n, int(math.ceil((n + 1) * coverage)))
    q_hat = float(np.partition(values, k - 1)[k - 1])
    return {"n": n, "k": k, "q_hat": q_hat}


def higher_quantile(values: np.ndarray, quantile: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile input must be nonempty and finite")
    index = min(array.size - 1, int(math.ceil(quantile * (array.size - 1))))
    return float(array[index])


def severity_boundaries(window_missing_rates: np.ndarray) -> Tuple[float, float]:
    rates = np.asarray(window_missing_rates, dtype=np.float64).reshape(-1)
    b1 = higher_quantile(rates, 1.0 / 3.0)
    b2 = higher_quantile(rates, 2.0 / 3.0)
    if not b1 < b2:
        raise ValueError(f"degenerate severity boundaries: b1={b1}, b2={b2}")
    return b1, b2


def assign_severity_bins(window_missing_rates: np.ndarray, boundaries: Tuple[float, float]) -> np.ndarray:
    rates = np.asarray(window_missing_rates, dtype=np.float64).reshape(-1)
    b1, b2 = boundaries
    bins = np.where(rates <= b1, 0, np.where(rates <= b2, 1, 2)).astype(np.int64)
    return bins


def additive_nonconformity(lower: torch.Tensor, upper: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    scores = torch.maximum(torch.maximum(lower - target, target - upper), torch.zeros_like(target))
    return scores.detach().cpu().numpy().astype(np.float64, copy=False)


def fit_additive_calibrators(
    native_intervals: Mapping[float, Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    window_missing_rates: np.ndarray,
    boundaries: Tuple[float, float] | None = None,
    levels: Iterable[float] = LEVELS,
    minimum_windows_per_bin: int = 50,
    minimum_scores_per_bin: int = 5000,
) -> Dict[str, object]:
    rates = np.asarray(window_missing_rates, dtype=np.float64).reshape(-1)
    if target.shape[0] != rates.size:
        raise ValueError("target window count does not match missing-rate count")
    boundaries = boundaries or severity_boundaries(rates)
    assignments = assign_severity_bins(rates, boundaries)
    window_counts = [int(np.sum(assignments == index)) for index in range(3)]
    if any(count < minimum_windows_per_bin for count in window_counts):
        raise ValueError(f"insufficient calibration windows per bin: {window_counts}")

    marginal: Dict[str, Dict[str, float | int]] = {}
    conditioned: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    score_counts = [0, 0, 0]
    for level in levels:
        lower, upper = native_intervals[float(level)]
        scores = additive_nonconformity(lower, upper, target)
        tag = str(int(round(float(level) * 100)))
        marginal[tag] = finite_sample_quantile(scores, float(level))
        conditioned[tag] = {}
        for bin_index in range(3):
            selected = scores[assignments == bin_index]
            score_counts[bin_index] = int(selected.size)
            if selected.size < minimum_scores_per_bin:
                raise ValueError(f"insufficient calibration scalar scores in bin {bin_index}: {selected.size}")
            conditioned[tag][str(bin_index)] = finite_sample_quantile(selected, float(level))

    return {
        "levels": [float(level) for level in levels],
        "score_type": "additive_cqr_nonconformity",
        "score_pooling": "all_scalar_forecast_elements",
        "finite_sample_order_statistic": True,
        "condition_source": "historical_input_mask_only",
        "condition_bin_count": 3,
        "bin_boundaries": [float(boundaries[0]), float(boundaries[1])],
        "calibration_windows_per_bin": window_counts,
        "calibration_scores_per_bin": score_counts,
        "marginal": marginal,
        "conditioned": conditioned,
        "assignments": assignments,
    }


def apply_additive_calibrators(
    native_intervals: Mapping[float, Tuple[torch.Tensor, torch.Tensor]],
    calibration: Mapping[str, object],
    window_missing_rates: np.ndarray,
) -> Dict[str, Dict[float, Tuple[torch.Tensor, torch.Tensor]]]:
    boundaries_raw = calibration["bin_boundaries"]
    boundaries = (float(boundaries_raw[0]), float(boundaries_raw[1]))
    assignments = assign_severity_bins(window_missing_rates, boundaries)
    marginal: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
    conditioned: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
    for level, (lower, upper) in native_intervals.items():
        tag = str(int(round(float(level) * 100)))
        q_marginal = float(calibration["marginal"][tag]["q_hat"])
        marginal[level] = (lower - q_marginal, upper + q_marginal)
        q_by_bin = calibration["conditioned"][tag]
        q_values = np.asarray([float(q_by_bin[str(index)]["q_hat"]) for index in assignments])
        q_tensor = torch.from_numpy(q_values).to(lower.device, lower.dtype).view(-1, 1, 1)
        conditioned[level] = (lower - q_tensor, upper + q_tensor)
    return {"marginal": marginal, "mask_severity_conditioned": conditioned}


def interval_metric_row(
    target: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, level: float
) -> Dict[str, float]:
    alpha = 1.0 - float(level)
    inside = ((target >= lower) & (target <= upper)).float()
    width = upper - lower
    score = (
        width
        + (2.0 / alpha) * (lower - target) * (target < lower)
        + (2.0 / alpha) * (target - upper) * (target > upper)
    )
    picp = float(inside.mean())
    mpiw = float(width.mean())
    return {
        "picp": picp,
        "mpiw": mpiw,
        "coverage_gap": picp - float(level),
        "absolute_coverage_error": abs(picp - float(level)),
        "interval_score": float(score.mean()),
    }


def interval_metrics(
    target: torch.Tensor, intervals: Mapping[float, Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, object]:
    by_level = {
        str(int(round(level * 100))): interval_metric_row(target, lower, upper, level)
        for level, (lower, upper) in intervals.items()
    }
    rows = list(by_level.values())
    return {
        "by_level": by_level,
        "mean_absolute_coverage_error": float(np.mean([row["absolute_coverage_error"] for row in rows])),
        "mean_interval_width": float(np.mean([row["mpiw"] for row in rows])),
        "mean_interval_score": float(np.mean([row["interval_score"] for row in rows])),
    }


def severity_bin_metrics(
    target: torch.Tensor,
    intervals: Mapping[float, Tuple[torch.Tensor, torch.Tensor]],
    assignments: np.ndarray,
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for level, (lower, upper) in intervals.items():
        tag = str(int(round(level * 100)))
        rows: Dict[str, Dict[str, float]] = {}
        for bin_index in range(3):
            chosen = torch.from_numpy(assignments == bin_index)
            rows[str(bin_index)] = interval_metric_row(
                target[chosen], lower[chosen], upper[chosen], level
            ) | {"window_count": int(chosen.sum())}
        errors = [row["absolute_coverage_error"] for row in rows.values()]
        coverages = [row["picp"] for row in rows.values()]
        output[tag] = {
            "bins": rows,
            "worst_bin_absolute_coverage_error": float(max(errors)),
            "mean_bin_absolute_coverage_error": float(np.mean(errors)),
            "bin_coverage_std": float(np.std(coverages)),
        }
    return output


def intervals_are_nested(
    intervals: Mapping[float, Tuple[torch.Tensor, torch.Tensor]], tolerance: float = 0.0
) -> bool:
    lower90, upper90 = intervals[0.90]
    lower95, upper95 = intervals[0.95]
    return bool(torch.all(lower95 <= lower90 + tolerance) and torch.all(upper95 >= upper90 - tolerance))
