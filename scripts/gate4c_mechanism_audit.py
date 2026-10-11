#!/usr/bin/env python
"""Read-only mechanism audit for the completed Gate 4c conformal runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate3b_smoke_missing_eval import (  # noqa: E402
    build_args,
    import_official_modules,
    parse_simple_yaml,
    sha256_file,
)
from gate4b_probabilistic_metrics import central_intervals  # noqa: E402
from models.conformal_calibrator import assign_severity_bins  # noqa: E402
from models.probabilistic_timeperceiver import Model  # noqa: E402


LEVELS = (0.90, 0.95)
HEADS = ("gaussian_head", "student_t_head")
HEAD_LABELS = {"gaussian_head": "Gaussian", "student_t_head": "Student-t"}
CALIBRATION_INDICES = ROOT / "artifacts/gate4c_splits/etth1_random_rate0.3_seed2025/calibration_window_indices.npy"
SELECTION_INDICES = ROOT / "artifacts/gate4c_splits/etth1_random_rate0.3_seed2025/selection_window_indices.npy"
SELECTION_MASK = ROOT / "artifacts/gate4c_masks/etth1/selection/random/rate_0.3/seed_43025.npz"
CALIBRATION_MASK = ROOT / "artifacts/gate4c_masks/etth1/calibration/random/rate_0.3/seed_44025.npz"
TEST_MASK = ROOT / "artifacts/masks/etth1/random/rate_0.3/seed_2025.npz"
ARTIFACT_DIR = ROOT / "artifacts/gate4c_mechanism_audit"
REPORT_DIR = ROOT / "reports"
RESULT_DIR = ROOT / "results"

# Descriptive diagnostics only. These values never alter the formal Gate 4c gate.
DIAGNOSTIC_RULES = {
    "native_well_calibrated_mean_ace_max": 0.015,
    "coverage_shift_observed_abs_min": 0.01,
    "distribution_relative_mean_shift_min": 0.05,
    "extreme_tail_p99_over_p90_min": 3.0,
    "severity_range_p90_minus_p10_max": 0.03,
    "conditioned_qhat_relative_effect_small_max": 0.10,
    "conditioned_method_mpiw_difference_max": 0.01,
    "conditioned_method_picp_difference_max": 0.002,
    "bin_residual_pairwise_ks_max": 0.10,
    "order_statistic_local_gap_relative_min": 0.05,
    "near_zero_oracle_delta_max": 1e-6,
}


class MissingArtifact(RuntimeError):
    pass


class ReplayParityFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_ready(value), indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_ready(row.get(key, "")) for key in fields})
    temporary.replace(path)


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - reporting fallback
        return repr(exc)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise MissingArtifact(str(path))
    return path


def find_formal_run(variant: str) -> Path:
    prefix = f"timeperceiver__etth1__gate4c__{variant}__random_rate0.3__seed2025__"
    candidates = []
    for path in (ROOT / "runs").glob(prefix + "*"):
        status_path = path / "status.json"
        if status_path.is_file() and read_json(status_path).get("status") == "SUCCESS":
            candidates.append(path)
    if not candidates:
        raise MissingArtifact(f"no successful formal run for {variant}")
    return sorted(candidates)[-1]


def formal_paths(runs: Mapping[str, Path]) -> list[Path]:
    paths = [
        ROOT / "results/gate4c_formal_decision.json",
        ROOT / "reports/gate4c_interval_metrics.csv",
        ROOT / "reports/gate4c_conditioned_bin_metrics.csv",
    ]
    for run in runs.values():
        paths.extend(sorted((run / "calibration").glob("*.json")))
    return [require_file(path) for path in paths]


def hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def metric_row(target: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float) -> dict:
    target64 = np.asarray(target, dtype=np.float64)
    lower64 = np.asarray(lower, dtype=np.float64)
    upper64 = np.asarray(upper, dtype=np.float64)
    inside = (target64 >= lower64) & (target64 <= upper64)
    width = upper64 - lower64
    alpha = 1.0 - float(level)
    score = (
        width
        + (2.0 / alpha) * (lower64 - target64) * (target64 < lower64)
        + (2.0 / alpha) * (target64 - upper64) * (target64 > upper64)
    )
    picp = float(inside.mean())
    return {
        "picp": picp,
        "mpiw": float(width.mean()),
        "coverage_gap": picp - float(level),
        "absolute_coverage_error": abs(picp - float(level)),
        "interval_score": float(score.mean()),
    }


def tensor_metric_row(target: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, level: float) -> dict:
    inside = ((target >= lower) & (target <= upper)).float()
    width = upper - lower
    alpha = 1.0 - float(level)
    score = (
        width
        + (2.0 / alpha) * (lower - target) * (target < lower)
        + (2.0 / alpha) * (target - upper) * (target > upper)
    )
    picp = float(inside.mean())
    return {"picp": picp, "mpiw": float(width.mean()), "interval_score": float(score.mean())}


def score_array(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(lower - target, target - upper), 0.0).astype(np.float64, copy=False)


def descriptive_stats(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise RuntimeError("descriptive statistics require finite nonempty values")
    quantiles = np.quantile(array, [0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(array.max()),
    }


def load_calibration(run: Path) -> dict:
    return read_json(require_file(run / "calibration/calibration_summary.json"))


def make_formal_intervals(
    native: Mapping[float, Tuple[torch.Tensor, torch.Tensor]],
    calibration: Mapping[str, Any],
    rates: np.ndarray,
) -> dict[str, dict[float, Tuple[torch.Tensor, torch.Tensor]]]:
    boundaries = tuple(float(value) for value in calibration["bin_boundaries"])
    assignments = assign_severity_bins(rates, boundaries)
    marginal: dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
    conditioned: dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
    for level, (lower, upper) in native.items():
        tag = str(int(round(level * 100)))
        marginal_q = float(calibration["marginal"][tag]["q_hat"])
        marginal[level] = (lower - marginal_q, upper + marginal_q)
        q_values = np.asarray(
            [float(calibration["conditioned"][tag][str(index)]["q_hat"]) for index in assignments],
            dtype=np.float32,
        )
        q_tensor = torch.from_numpy(q_values).view(-1, 1, 1)
        conditioned[level] = (lower - q_tensor, upper + q_tensor)
    return {"marginal": marginal, "conditioned": conditioned}


def verify_order_statistics(
    native: Mapping[float, Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    calibration: Mapping[str, Any],
    rates: np.ndarray,
    tolerance: float = 1e-7,
) -> dict:
    assignments = assign_severity_bins(rates, tuple(calibration["bin_boundaries"]))
    checks: dict[str, Any] = {}
    for level, (lower, upper) in native.items():
        tag = str(int(round(level * 100)))
        scores = score_array(target.numpy(), lower.numpy(), upper.numpy())
        formal = calibration["marginal"][tag]
        sorted_scores = np.sort(scores.reshape(-1))
        computed = float(sorted_scores[int(formal["k"]) - 1])
        checks[f"marginal_{tag}"] = {
            "formal": float(formal["q_hat"]),
            "computed_order_statistic": computed,
            "match": abs(computed - float(formal["q_hat"])) <= tolerance,
        }
        for bin_index in range(3):
            selected = np.sort(scores[assignments == bin_index].reshape(-1))
            formal_bin = calibration["conditioned"][tag][str(bin_index)]
            computed_bin = float(selected[int(formal_bin["k"]) - 1])
            checks[f"conditioned_{tag}_bin{bin_index}"] = {
                "formal": float(formal_bin["q_hat"]),
                "computed_order_statistic": computed_bin,
                "match": abs(computed_bin - float(formal_bin["q_hat"])) <= tolerance,
            }
    if not all(item["match"] for item in checks.values()):
        raise ReplayParityFailure("formal q-hat does not match replayed calibration order statistic")
    return checks


def collect_outputs(
    args: Any,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    mask_array: np.ndarray,
    device: torch.device,
) -> dict:
    locations: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    dfs: list[torch.Tensor] = []
    offset = 0
    batch_y_unchanged = True
    patch_indices = (
        list(range(args.seq_len // args.patch_len)),
        list(range(args.seq_len // args.patch_len, (args.seq_len + args.pred_len) // args.patch_len)),
    )
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            current_mask = mask_array[offset:offset + len(batch_x)]
            if len(current_mask) != len(batch_x):
                raise ReplayParityFailure("mask array was truncated during replay")
            offset += len(batch_x)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            before_y = batch_y.clone()
            mask = torch.from_numpy(current_mask).float().to(device)
            decoder = torch.cat(
                [batch_y[:, :args.label_len, :], torch.zeros_like(batch_y[:, -args.pred_len:, :])], dim=1
            )
            output = model(batch_x * mask, batch_x_mark, decoder, batch_y_mark, patch_indices, mask=mask)
            batch_y_unchanged &= bool(torch.equal(before_y, batch_y))
            locations.append(output["loc"].cpu())
            scales.append(output["scale"].cpu())
            targets.append(batch_y[:, -args.pred_len:, :].cpu())
            if output["df"] is not None:
                dfs.append(output["df"].cpu())
    if offset != len(mask_array):
        raise ReplayParityFailure(f"replay consumed {offset} masks, expected {len(mask_array)}")
    return {
        "loc": torch.cat(locations),
        "scale": torch.cat(scales),
        "target": torch.cat(targets),
        "df": torch.cat(dfs) if dfs else None,
        "windows": offset,
        "batch_y_unchanged": batch_y_unchanged,
    }


def replay_variant(variant: str, run: Path, device_name: str) -> tuple[dict, dict]:
    status = read_json(require_file(run / "status.json"))
    config_path = require_file(run / "resolved_config.yaml")
    config = parse_simple_yaml(config_path)
    baseline_config = require_file(resolve(config["config"]))
    args = build_args(parse_simple_yaml(baseline_config), int(config["batch_size"]), device_name)
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the requested eval-only replay")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    provider, _ = import_official_modules()
    validation_data, _ = provider(args, "val")
    test_data, _ = provider(args, "test")
    calibration_indices = np.load(require_file(resolve(config["calibration_split_path"])), allow_pickle=False)
    calibration_data = torch.utils.data.Subset(validation_data, calibration_indices.tolist())
    calibration_loader = torch.utils.data.DataLoader(
        calibration_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    calibration_mask = np.load(require_file(resolve(config["calibration_mask_path"])), allow_pickle=False)["mask"]
    test_mask = np.load(require_file(resolve(config["test_mask_path"])), allow_pickle=False)["mask"]
    expected_calibration_shape = (len(calibration_data), args.seq_len, args.enc_in)
    expected_test_shape = (len(test_data), args.seq_len, args.enc_in)
    if tuple(calibration_mask.shape) != expected_calibration_shape:
        raise ReplayParityFailure(f"calibration mask shape {calibration_mask.shape} != {expected_calibration_shape}")
    if tuple(test_mask.shape) != expected_test_shape:
        raise ReplayParityFailure(f"test mask shape {test_mask.shape} != {expected_test_shape}")

    source_checkpoint = require_file(resolve(config["source_checkpoint"]))
    best_checkpoint = require_file(run / "checkpoints/best_prob_head.pt")
    raw_checkpoint = torch.load(source_checkpoint, map_location="cpu")
    best_object = torch.load(best_checkpoint, map_location="cpu")
    model = Model(args, variant, True).float()
    missing, unexpected = model.load_state_dict(raw_checkpoint, strict=False)
    if unexpected or not all(name.startswith("probabilistic_head.") for name in missing):
        raise ReplayParityFailure(f"source checkpoint extension mismatch: {missing=}, {unexpected=}")
    model.initialize_probabilistic_location()
    strict = model.probabilistic_head.load_state_dict(best_object["probabilistic_head_state_dict"], strict=True)
    if strict.missing_keys or strict.unexpected_keys:
        raise ReplayParityFailure("best probability-head checkpoint did not reload strictly")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()

    calibration_outputs = collect_outputs(args, model, calibration_loader, calibration_mask, device)
    test_outputs = collect_outputs(args, model, test_loader, test_mask, device)
    calibration_rates = 1.0 - calibration_mask.mean(axis=(1, 2))
    test_rates = 1.0 - test_mask.mean(axis=(1, 2))
    calibration_native = {
        level: central_intervals(
            variant,
            calibration_outputs["loc"],
            calibration_outputs["scale"],
            calibration_outputs["df"],
        )[level]
        for level in LEVELS
    }
    test_native = {
        level: central_intervals(
            variant, test_outputs["loc"], test_outputs["scale"], test_outputs["df"]
        )[level]
        for level in LEVELS
    }
    calibration = load_calibration(run)
    order_checks = verify_order_statistics(
        calibration_native, calibration_outputs["target"], calibration, calibration_rates
    )
    test_formal = make_formal_intervals(test_native, calibration, test_rates)

    difference = test_outputs["loc"] - test_outputs["target"]
    replay_point = {
        "mse": float((difference * difference).mean()),
        "mae": float(difference.abs().mean()),
        "rmse": math.sqrt(float((difference * difference).mean())),
    }
    formal_point = {
        "mse": float(status["test_mse"]),
        "mae": float(status["test_mae"]),
        "rmse": float(status["test_rmse"]),
    }
    point_diffs = {key: abs(replay_point[key] - formal_point[key]) for key in replay_point}
    point_match = all(value <= 1e-6 for value in point_diffs.values())
    native_checks: dict[str, Any] = {}
    for level in LEVELS:
        tag = str(int(round(level * 100)))
        replay_row = tensor_metric_row(test_outputs["target"], *test_native[level], level)
        formal_row = status["native_interval_metrics"]["by_level"][tag]
        differences = {
            key: abs(float(replay_row[key]) - float(formal_row[key]))
            for key in ("picp", "mpiw", "interval_score")
        }
        native_checks[tag] = {
            "replay": replay_row,
            "formal": {key: formal_row[key] for key in ("picp", "mpiw", "interval_score")},
            "absolute_differences": differences,
            "match": differences["picp"] <= 5e-7
            and differences["mpiw"] <= 1e-6
            and differences["interval_score"] <= 2e-6,
        }
    parity = {
        "point_metrics": {
            "replay": replay_point,
            "formal": formal_point,
            "absolute_differences": point_diffs,
            "match": point_match,
        },
        "native_intervals": native_checks,
        "formal_qhat_order_statistics": order_checks,
        "batch_y_unchanged": bool(
            calibration_outputs["batch_y_unchanged"] and test_outputs["batch_y_unchanged"]
        ),
    }
    parity["pass"] = bool(
        point_match
        and all(item["match"] for item in native_checks.values())
        and all(item["match"] for item in order_checks.values())
        and parity["batch_y_unchanged"]
    )
    if not parity["pass"]:
        raise ReplayParityFailure(f"{variant} replay parity failed: {parity}")

    arrays_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split, outputs, native, rates in (
        ("calibration", calibration_outputs, calibration_native, calibration_rates),
        ("test", test_outputs, test_native, test_rates),
    ):
        formal_intervals = make_formal_intervals(native, calibration, rates)
        target = outputs["target"].numpy().astype(np.float32, copy=False)
        loc = outputs["loc"].numpy().astype(np.float32, copy=False)
        scale = outputs["scale"].numpy().astype(np.float32, copy=False)
        assignments = assign_severity_bins(rates, tuple(calibration["bin_boundaries"]))
        arrays: dict[str, np.ndarray] = {
            "target": target,
            "loc": loc,
            "scale": scale,
            "df": np.asarray([], dtype=np.float32)
            if outputs["df"] is None
            else outputs["df"].numpy().astype(np.float32, copy=False),
            "window_missing_rate": np.asarray(rates, dtype=np.float64),
            "severity_bin": assignments.astype(np.int8),
            "horizon_index": np.arange(1, args.pred_len + 1, dtype=np.int16),
            "channel_index": np.arange(args.enc_in, dtype=np.int8),
            "absolute_residual": np.abs(target - loc).astype(np.float32, copy=False),
            "standardized_residual": (np.abs(target - loc) / np.maximum(scale, 1e-12)).astype(
                np.float32, copy=False
            ),
        }
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            native_lower, native_upper = native[level]
            marginal_lower, marginal_upper = formal_intervals["marginal"][level]
            conditioned_lower, conditioned_upper = formal_intervals["conditioned"][level]
            arrays.update(
                {
                    f"native_lower_{tag}": native_lower.numpy().astype(np.float32, copy=False),
                    f"native_upper_{tag}": native_upper.numpy().astype(np.float32, copy=False),
                    f"marginal_lower_{tag}": marginal_lower.numpy().astype(np.float32, copy=False),
                    f"marginal_upper_{tag}": marginal_upper.numpy().astype(np.float32, copy=False),
                    f"conditioned_lower_{tag}": conditioned_lower.numpy().astype(np.float32, copy=False),
                    f"conditioned_upper_{tag}": conditioned_upper.numpy().astype(np.float32, copy=False),
                    f"nonconformity_score_{tag}": score_array(
                        target, native_lower.numpy(), native_upper.numpy()
                    ).astype(np.float32, copy=False),
                }
            )
        arrays_by_split[split] = arrays

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays_by_split, parity


def diagnostics_path(variant: str, split: str) -> Path:
    stem = "student_t" if variant == "student_t_head" else "gaussian"
    return ARTIFACT_DIR / f"{stem}_{split}_diagnostics.npz"


def load_diagnostics(variant: str, split: str) -> dict[str, np.ndarray]:
    path = require_file(diagnostics_path(variant, split))
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def interval_arrays(data: Mapping[str, np.ndarray], method: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    return data[f"{method}_lower_{tag}"], data[f"{method}_upper_{tag}"]


def native_shift_rows(all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> list[dict]:
    rows = []
    for variant in HEADS:
        calibration = all_data[variant]["calibration"]
        test = all_data[variant]["test"]
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            calibration_row = metric_row(calibration["target"], *interval_arrays(calibration, "native", tag), level)
            test_row = metric_row(test["target"], *interval_arrays(test, "native", tag), level)
            shift = test_row["picp"] - calibration_row["picp"]
            rows.append(
                {
                    "head": variant,
                    "head_label": HEAD_LABELS[variant],
                    "coverage": level,
                    "calibration_picp": calibration_row["picp"],
                    "test_picp": test_row["picp"],
                    "calibration_coverage_gap": calibration_row["coverage_gap"],
                    "test_coverage_gap": test_row["coverage_gap"],
                    "coverage_shift_test_minus_calibration": shift,
                    "calibration_mpiw": calibration_row["mpiw"],
                    "test_mpiw": test_row["mpiw"],
                    "calibration_interval_score": calibration_row["interval_score"],
                    "test_interval_score": test_row["interval_score"],
                    "calibration_undercoverage_stronger_than_test": calibration_row["coverage_gap"]
                    < test_row["coverage_gap"],
                    "calibration_overcoverage_stronger_than_test": calibration_row["coverage_gap"]
                    > test_row["coverage_gap"],
                    "coverage_distribution_shift_observed": abs(shift)
                    >= DIAGNOSTIC_RULES["coverage_shift_observed_abs_min"],
                }
            )
    return rows


def score_distribution_rows(
    all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> list[dict]:
    rows = []
    for variant in HEADS:
        data = all_data[variant]["calibration"]
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            scores = np.asarray(data[f"nonconformity_score_{tag}"], dtype=np.float64).reshape(-1)
            positive = scores[scores > 0]
            stats = descriptive_stats(scores)
            positive_stats = descriptive_stats(positive)
            qhat = float(calibrations[variant]["marginal"][tag]["q_hat"])
            tail_ratio = stats["p99"] / max(stats["p90"], 1e-12)
            row = {
                "head": variant,
                "head_label": HEAD_LABELS[variant],
                "coverage": level,
                "score_count": int(scores.size),
                "zero_score_count": int(np.count_nonzero(scores == 0)),
                "zero_score_ratio": float(np.mean(scores == 0)),
                "positive_score_count": int(positive.size),
                "positive_score_ratio": float(np.mean(scores > 0)),
                **{key: stats[key] for key in ("mean", "std", "median", "p75", "p90", "p95", "p99", "max")},
                **{f"positive_{key}": positive_stats[key] for key in ("mean", "median", "p90", "p95", "p99", "max")},
                "formal_qhat": qhat,
                "qhat_percentile_position_le": float(np.mean(scores <= qhat)),
                "scores_greater_equal_qhat": int(np.count_nonzero(scores >= qhat)),
                "p99_over_p90": tail_ratio,
                "extreme_tail_dominated": tail_ratio > DIAGNOSTIC_RULES["extreme_tail_p99_over_p90_min"],
            }
            row["qhat_driven_by_few_extreme_errors"] = row["extreme_tail_dominated"]
            row["qhat_driven_by_many_small_errors"] = not row["extreme_tail_dominated"]
            row["positive_tail_heavy"] = row["extreme_tail_dominated"]
            rows.append(row)
    return rows


def expansion_rows(all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> list[dict]:
    rows = []
    for variant in HEADS:
        data = all_data[variant]["test"]
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            native = metric_row(data["target"], *interval_arrays(data, "native", tag), level)
            for method in ("marginal", "conditioned"):
                conformal = metric_row(data["target"], *interval_arrays(data, method, tag), level)
                width_change = conformal["mpiw"] - native["mpiw"]
                rows.append(
                    {
                        "head": variant,
                        "head_label": HEAD_LABELS[variant],
                        "method": method,
                        "coverage": level,
                        "native_picp": native["picp"],
                        "conformal_picp": conformal["picp"],
                        "coverage_change": conformal["picp"] - native["picp"],
                        "native_mpiw": native["mpiw"],
                        "conformal_mpiw": conformal["mpiw"],
                        "width_increase": width_change,
                        "native_absolute_coverage_error": native["absolute_coverage_error"],
                        "conformal_absolute_coverage_error": conformal["absolute_coverage_error"],
                        "absolute_coverage_error_change": conformal["absolute_coverage_error"]
                        - native["absolute_coverage_error"],
                        "native_interval_score": native["interval_score"],
                        "conformal_interval_score": conformal["interval_score"],
                        "interval_score_change": conformal["interval_score"] - native["interval_score"],
                        "coverage_gain_per_width": (conformal["picp"] - native["picp"])
                        / max(width_change, 1e-12),
                    }
                )
    return rows


def horizon_rows(all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> list[dict]:
    rows = []
    for variant in HEADS:
        data = all_data[variant]["test"]
        target = data["target"]
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            for horizon in range(target.shape[1]):
                metrics = {
                    method: metric_row(
                        target[:, horizon, :],
                        interval_arrays(data, method, tag)[0][:, horizon, :],
                        interval_arrays(data, method, tag)[1][:, horizon, :],
                        level,
                    )
                    for method in ("native", "marginal", "conditioned")
                }
                standardized = data["standardized_residual"][:, horizon, :].reshape(-1)
                rows.append(
                    {
                        "head": variant,
                        "head_label": HEAD_LABELS[variant],
                        "coverage": level,
                        "horizon": horizon + 1,
                        "element_count": int(target[:, horizon, :].size),
                        "native_picp": metrics["native"]["picp"],
                        "marginal_picp": metrics["marginal"]["picp"],
                        "conditioned_picp": metrics["conditioned"]["picp"],
                        "native_absolute_coverage_error": metrics["native"]["absolute_coverage_error"],
                        "marginal_absolute_coverage_error": metrics["marginal"]["absolute_coverage_error"],
                        "conditioned_absolute_coverage_error": metrics["conditioned"]["absolute_coverage_error"],
                        "native_mpiw": metrics["native"]["mpiw"],
                        "marginal_mpiw": metrics["marginal"]["mpiw"],
                        "conditioned_mpiw": metrics["conditioned"]["mpiw"],
                        "native_interval_score": metrics["native"]["interval_score"],
                        "marginal_interval_score": metrics["marginal"]["interval_score"],
                        "conditioned_interval_score": metrics["conditioned"]["interval_score"],
                        "mean_absolute_residual": float(data["absolute_residual"][:, horizon, :].mean()),
                        "mean_scale": float(data["scale"][:, horizon, :].mean()),
                        "mean_standardized_residual": float(standardized.mean()),
                        "standardized_residual_p90": float(np.quantile(standardized, 0.90)),
                        "standardized_residual_p95": float(np.quantile(standardized, 0.95)),
                    }
                )
    return rows


def channel_rows(all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> list[dict]:
    rows = []
    for variant in HEADS:
        data = all_data[variant]["test"]
        target = data["target"]
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            for channel in range(target.shape[2]):
                metrics = {
                    method: metric_row(
                        target[:, :, channel],
                        interval_arrays(data, method, tag)[0][:, :, channel],
                        interval_arrays(data, method, tag)[1][:, :, channel],
                        level,
                    )
                    for method in ("native", "marginal", "conditioned")
                }
                standardized = data["standardized_residual"][:, :, channel].reshape(-1)
                rows.append(
                    {
                        "head": variant,
                        "head_label": HEAD_LABELS[variant],
                        "coverage": level,
                        "channel": channel,
                        "element_count": int(target[:, :, channel].size),
                        "native_picp": metrics["native"]["picp"],
                        "marginal_picp": metrics["marginal"]["picp"],
                        "conditioned_picp": metrics["conditioned"]["picp"],
                        "native_absolute_coverage_error": metrics["native"]["absolute_coverage_error"],
                        "marginal_absolute_coverage_error": metrics["marginal"]["absolute_coverage_error"],
                        "conditioned_absolute_coverage_error": metrics["conditioned"]["absolute_coverage_error"],
                        "native_mpiw": metrics["native"]["mpiw"],
                        "conditioned_mpiw": metrics["conditioned"]["mpiw"],
                        "native_interval_score": metrics["native"]["interval_score"],
                        "conditioned_interval_score": metrics["conditioned"]["interval_score"],
                        "mean_absolute_residual": float(data["absolute_residual"][:, :, channel].mean()),
                        "mean_scale": float(data["scale"][:, :, channel].mean()),
                        "standardized_residual_p90": float(np.quantile(standardized, 0.90)),
                        "standardized_residual_p95": float(np.quantile(standardized, 0.95)),
                    }
                )
    return rows


def severity_rows(
    all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict], dict]:
    rows = []
    pairwise_ks: dict[str, float] = {}
    for variant in HEADS:
        calibration = calibrations[variant]
        for split in ("calibration", "test"):
            data = all_data[variant][split]
            assignments = data["severity_bin"].astype(np.int64)
            rates = data["window_missing_rate"]
            for level in LEVELS:
                tag = str(int(round(level * 100)))
                marginal_q = float(calibration["marginal"][tag]["q_hat"])
                for bin_index in range(3):
                    chosen = assignments == bin_index
                    native = metric_row(
                        data["target"][chosen],
                        interval_arrays(data, "native", tag)[0][chosen],
                        interval_arrays(data, "native", tag)[1][chosen],
                        level,
                    )
                    marginal = metric_row(
                        data["target"][chosen],
                        interval_arrays(data, "marginal", tag)[0][chosen],
                        interval_arrays(data, "marginal", tag)[1][chosen],
                        level,
                    )
                    conditioned = metric_row(
                        data["target"][chosen],
                        interval_arrays(data, "conditioned", tag)[0][chosen],
                        interval_arrays(data, "conditioned", tag)[1][chosen],
                        level,
                    )
                    conditioned_q = float(calibration["conditioned"][tag][str(bin_index)]["q_hat"])
                    selected_rates = rates[chosen]
                    rows.append(
                        {
                            "head": variant,
                            "head_label": HEAD_LABELS[variant],
                            "split": split,
                            "coverage": level,
                            "bin": bin_index,
                            "window_count": int(chosen.sum()),
                            "scalar_element_count": int(data["target"][chosen].size),
                            "mean_missing_rate": float(selected_rates.mean()),
                            "std_missing_rate": float(selected_rates.std()),
                            "min_missing_rate": float(selected_rates.min()),
                            "max_missing_rate": float(selected_rates.max()),
                            "native_picp": native["picp"],
                            "native_mpiw": native["mpiw"],
                            "native_interval_score": native["interval_score"],
                            "marginal_picp": marginal["picp"],
                            "conditioned_picp": conditioned["picp"],
                            "marginal_mpiw": marginal["mpiw"],
                            "conditioned_mpiw": conditioned["mpiw"],
                            "marginal_interval_score": marginal["interval_score"],
                            "conditioned_interval_score": conditioned["interval_score"],
                            "formal_marginal_qhat": marginal_q,
                            "formal_conditioned_qhat": conditioned_q,
                            "conditioned_minus_marginal_qhat": conditioned_q - marginal_q,
                            "conditioned_qhat_relative_difference": (conditioned_q - marginal_q)
                            / max(abs(marginal_q), 1e-12),
                        }
                    )
            residuals = [
                data["standardized_residual"][assignments == index].reshape(-1) for index in range(3)
            ]
            maximum = max(
                float(ks_2samp(residuals[left], residuals[right]).statistic)
                for left, right in ((0, 1), (0, 2), (1, 2))
            )
            pairwise_ks[f"{variant}_{split}"] = maximum
    return rows, pairwise_ks


def distribution_shift_rows(
    all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]
) -> list[dict]:
    rows = []
    for variant in HEADS:
        calibration = all_data[variant]["calibration"]
        test = all_data[variant]["test"]
        variables: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "absolute_residual": (calibration["absolute_residual"], test["absolute_residual"]),
            "scale": (calibration["scale"], test["scale"]),
            "standardized_residual": (
                calibration["standardized_residual"],
                test["standardized_residual"],
            ),
        }
        if variant == "student_t_head":
            calibration_df = calibration["df"]
            test_df = test["df"]
            calibration_predictive_std = calibration["scale"] * np.sqrt(calibration_df / (calibration_df - 2.0))
            test_predictive_std = test["scale"] * np.sqrt(test_df / (test_df - 2.0))
            variables.update(
                {
                    "df": (calibration_df, test_df),
                    "predictive_std": (calibration_predictive_std, test_predictive_std),
                    "predictive_standardized_residual": (
                        calibration["absolute_residual"] / calibration_predictive_std,
                        test["absolute_residual"] / test_predictive_std,
                    ),
                }
            )
        for variable, (calibration_values, test_values) in variables.items():
            calibration_flat = np.asarray(calibration_values, dtype=np.float64).reshape(-1)
            test_flat = np.asarray(test_values, dtype=np.float64).reshape(-1)
            calibration_stats = descriptive_stats(calibration_flat)
            test_stats = descriptive_stats(test_flat)
            mean_shift = test_stats["mean"] - calibration_stats["mean"]
            row = {
                "head": variant,
                "head_label": HEAD_LABELS[variant],
                "variable": variable,
                **{f"calibration_{key}": calibration_stats[key] for key in calibration_stats},
                **{f"test_{key}": test_stats[key] for key in test_stats},
                "mean_shift_test_minus_calibration": mean_shift,
                "median_shift_test_minus_calibration": test_stats["median"] - calibration_stats["median"],
                "p90_shift_test_minus_calibration": test_stats["p90"] - calibration_stats["p90"],
                "relative_mean_shift": mean_shift / max(abs(calibration_stats["mean"]), 1e-12),
                "ks_statistic": float(ks_2samp(calibration_flat, test_flat).statistic),
                "wasserstein_distance": float(wasserstein_distance(calibration_flat, test_flat)),
            }
            rows.append(row)
    return rows


def signed_margin(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    below = target < lower
    above = target > upper
    inside_margin = -np.minimum(target - lower, upper - target)
    return np.where(below, lower - target, np.where(above, target - upper, inside_margin))


def counterfactual_rows(
    all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> list[dict]:
    rows = []
    for variant in HEADS:
        data = all_data[variant]["test"]
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            lower, upper = interval_arrays(data, "native", tag)
            native = metric_row(data["target"], lower, upper, level)
            margins = signed_margin(data["target"], lower, upper).reshape(-1)
            count = int(margins.size)
            k = min(count, int(math.ceil((count + 1) * level)))
            oracle_delta = float(np.partition(margins, k - 1)[k - 1])
            threshold = DIAGNOSTIC_RULES["near_zero_oracle_delta_max"]
            if oracle_delta > threshold:
                interpretation = "test_target_requires_expansion"
            elif oracle_delta < -threshold:
                interpretation = "test_target_would_prefer_shrinkage"
            else:
                interpretation = "test_target_requires_near_zero_adjustment"
            formal_qhat = float(calibrations[variant]["marginal"][tag]["q_hat"])
            rows.append(
                {
                    "head": variant,
                    "head_label": HEAD_LABELS[variant],
                    "coverage": level,
                    "native_picp": native["picp"],
                    "formal_additive_qhat": formal_qhat,
                    "oracle_signed_delta": oracle_delta,
                    "oracle_delta_sign": "positive" if oracle_delta > 0 else "negative" if oracle_delta < 0 else "zero",
                    "interpretation": interpretation,
                    "formal_width_increase": 2.0 * formal_qhat,
                    "oracle_implied_width_change": 2.0 * oracle_delta,
                    "n_test_signed_margins": count,
                    "order_statistic_k": k,
                    "uses_test_labels": True,
                    "counterfactual_only": True,
                    "formal_calibration": False,
                    "formal_gate_uses_result": False,
                    "warning": "COUNTERFACTUAL_ONLY; USES_TEST_LABELS; NOT_FORMAL_CALIBRATION",
                }
            )
    return rows


def order_statistic_rows(
    all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> list[dict]:
    rows = []
    for variant in HEADS:
        data = all_data[variant]["calibration"]
        assignments = data["severity_bin"].astype(np.int64)
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            scores = np.asarray(data[f"nonconformity_score_{tag}"], dtype=np.float64)
            pools: list[tuple[str, str, np.ndarray, Mapping[str, Any]]] = [
                ("marginal", "all", scores.reshape(-1), calibrations[variant]["marginal"][tag])
            ]
            for bin_index in range(3):
                pools.append(
                    (
                        "conditioned",
                        str(bin_index),
                        scores[assignments == bin_index].reshape(-1),
                        calibrations[variant]["conditioned"][tag][str(bin_index)],
                    )
                )
            for method, bin_name, pool, formal in pools:
                ordered = np.sort(pool)
                n = int(formal["n"])
                k = int(formal["k"])
                qhat = float(formal["q_hat"])
                score_minus = float(ordered[max(k - 2, 0)])
                score_at = float(ordered[k - 1])
                score_plus = float(ordered[min(k, n - 1)])
                local_gap = score_plus - score_minus
                relative_gap = local_gap / max(abs(qhat), 1e-12)
                sensitive = relative_gap > DIAGNOSTIC_RULES["order_statistic_local_gap_relative_min"]
                rows.append(
                    {
                        "head": variant,
                        "head_label": HEAD_LABELS[variant],
                        "coverage": level,
                        "method": method,
                        "bin": bin_name,
                        "n_calibration_scores": n,
                        "alpha": 1.0 - level,
                        "k": k,
                        "k_over_n": k / n,
                        "qhat": qhat,
                        "score_at_k_minus_1": score_minus,
                        "score_at_k": score_at,
                        "score_at_k_plus_1": score_plus,
                        "local_order_statistic_gap": local_gap,
                        "local_gap_relative_to_qhat": relative_gap,
                        "qhat_sensitive_to_single_order_statistic": sensitive,
                        "qhat_plateau_or_jump": "jump" if sensitive else "plateau_or_smooth",
                        "finite_sample_discreteness_large": sensitive,
                    }
                )
    return rows


def conditioned_effects(
    all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> list[dict]:
    rows = []
    for variant in HEADS:
        data = all_data[variant]["test"]
        assignments = data["severity_bin"].astype(np.int64)
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            marginal = metric_row(data["target"], *interval_arrays(data, "marginal", tag), level)
            conditioned = metric_row(data["target"], *interval_arrays(data, "conditioned", tag), level)
            marginal_bins = []
            conditioned_bins = []
            for bin_index in range(3):
                chosen = assignments == bin_index
                marginal_bins.append(
                    metric_row(
                        data["target"][chosen],
                        interval_arrays(data, "marginal", tag)[0][chosen],
                        interval_arrays(data, "marginal", tag)[1][chosen],
                        level,
                    )
                )
                conditioned_bins.append(
                    metric_row(
                        data["target"][chosen],
                        interval_arrays(data, "conditioned", tag)[0][chosen],
                        interval_arrays(data, "conditioned", tag)[1][chosen],
                        level,
                    )
                )
            marginal_q = float(calibrations[variant]["marginal"][tag]["q_hat"])
            conditioned_q = np.asarray(
                [float(calibrations[variant]["conditioned"][tag][str(index)]["q_hat"]) for index in range(3)]
            )
            q_differences = conditioned_q - marginal_q
            marginal_lower, marginal_upper = interval_arrays(data, "marginal", tag)
            conditioned_lower, conditioned_upper = interval_arrays(data, "conditioned", tag)
            endpoint_difference = np.maximum(
                np.abs(conditioned_lower - marginal_lower), np.abs(conditioned_upper - marginal_upper)
            )
            mpiw_difference = conditioned["mpiw"] - marginal["mpiw"]
            picp_difference = conditioned["picp"] - marginal["picp"]
            negligible = (
                abs(mpiw_difference) < DIAGNOSTIC_RULES["conditioned_method_mpiw_difference_max"]
                and abs(picp_difference) < DIAGNOSTIC_RULES["conditioned_method_picp_difference_max"]
            )
            rows.append(
                {
                    "head": variant,
                    "coverage": level,
                    "marginal_qhat": marginal_q,
                    "conditioned_qhat_per_bin": conditioned_q.tolist(),
                    "max_absolute_qhat_difference": float(np.max(np.abs(q_differences))),
                    "mean_absolute_qhat_difference": float(np.mean(np.abs(q_differences))),
                    "max_relative_qhat_difference": float(np.max(np.abs(q_differences)) / max(abs(marginal_q), 1e-12)),
                    "overall_picp_difference": picp_difference,
                    "overall_mpiw_difference": mpiw_difference,
                    "overall_interval_score_difference": conditioned["interval_score"] - marginal["interval_score"],
                    "worst_bin_coverage_error_difference": max(
                        row["absolute_coverage_error"] for row in conditioned_bins
                    )
                    - max(row["absolute_coverage_error"] for row in marginal_bins),
                    "mean_bin_coverage_error_difference": float(
                        np.mean([row["absolute_coverage_error"] for row in conditioned_bins])
                        - np.mean([row["absolute_coverage_error"] for row in marginal_bins])
                    ),
                    "bin_coverage_std_difference": float(
                        np.std([row["picp"] for row in conditioned_bins])
                        - np.std([row["picp"] for row in marginal_bins])
                    ),
                    "max_interval_endpoint_difference": float(endpoint_difference.max()),
                    "fraction_test_elements_interval_differs": float(np.mean(endpoint_difference > 0)),
                    "conditioned_method_effect_negligible": negligible,
                }
            )
    return rows


def channel_tail_concentration(
    all_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> dict:
    details = {}
    for variant in HEADS:
        data = all_data[variant]["calibration"]
        for level in LEVELS:
            tag = str(int(round(level * 100)))
            scores = data[f"nonconformity_score_{tag}"]
            qhat = float(calibrations[variant]["marginal"][tag]["q_hat"])
            counts = np.asarray([np.count_nonzero(scores[:, :, channel] >= qhat) for channel in range(scores.shape[2])])
            total = max(int(counts.sum()), 1)
            top_two = np.argsort(counts)[-2:][::-1]
            details[f"{variant}_{tag}"] = {
                "counts_at_or_above_qhat_by_channel": counts.tolist(),
                "top_two_channels": top_two.tolist(),
                "top_two_share": float(counts[top_two].sum() / total),
                "one_or_two_channels_dominate_qhat": float(counts[top_two].sum() / total) >= 0.50,
            }
    return details


def compact_horizon_summary(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict]:
    lines = ["# Gate 4c Horizon Mechanism Summary", ""]
    summary: dict[str, Any] = {}
    for variant in HEADS:
        for level in LEVELS:
            selected = [row for row in rows if row["head"] == variant and row["coverage"] == level]
            worst = sorted(selected, key=lambda row: row["native_absolute_coverage_error"], reverse=True)[:10]
            best = sorted(selected, key=lambda row: row["native_absolute_coverage_error"])[:10]
            standardized = sorted(selected, key=lambda row: row["mean_standardized_residual"], reverse=True)[:10]
            thirds = {
                "early_1_32": float(np.mean([row["native_absolute_coverage_error"] for row in selected[:32]])),
                "middle_33_64": float(np.mean([row["native_absolute_coverage_error"] for row in selected[32:64]])),
                "late_65_96": float(np.mean([row["native_absolute_coverage_error"] for row in selected[64:96]])),
            }
            concentration = max(thirds, key=thirds.get)
            key = f"{variant}_{int(level * 100)}"
            summary[key] = {
                "worst_native_coverage_horizons": [row["horizon"] for row in worst],
                "best_native_coverage_horizons": [row["horizon"] for row in best],
                "worst_standardized_residual_horizons": [row["horizon"] for row in standardized],
                "mean_native_ace_by_third": thirds,
                "largest_error_third": concentration,
                "undercoverage_concentrated_early_horizon": concentration == "early_1_32",
                "undercoverage_concentrated_late_horizon": concentration == "late_65_96",
                "fraction_horizons_marginal_worsens_ace": float(
                    np.mean(
                        [row["marginal_absolute_coverage_error"] > row["native_absolute_coverage_error"] for row in selected]
                    )
                ),
            }
            lines.extend(
                [
                    f"## {HEAD_LABELS[variant]} {int(level * 100)}%",
                    "",
                    f"- Worst native coverage horizons: {summary[key]['worst_native_coverage_horizons']}",
                    f"- Best native coverage horizons: {summary[key]['best_native_coverage_horizons']}",
                    f"- Worst standardized residual horizons: {summary[key]['worst_standardized_residual_horizons']}",
                    f"- Mean native ACE by third: {thirds}",
                    f"- Largest-error third: {concentration}",
                    f"- Marginal conformal worsens horizon ACE fraction: {summary[key]['fraction_horizons_marginal_worsens_ace']:.3f}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n", summary


def compact_channel_summary(
    rows: Sequence[Mapping[str, Any]], tail_details: Mapping[str, Any]
) -> tuple[str, dict]:
    lines = ["# Gate 4c Channel Mechanism Summary", ""]
    summary: dict[str, Any] = {}
    for variant in HEADS:
        for level in LEVELS:
            selected = [row for row in rows if row["head"] == variant and row["coverage"] == level]
            worst = sorted(selected, key=lambda row: row["native_absolute_coverage_error"], reverse=True)
            best = sorted(selected, key=lambda row: row["native_absolute_coverage_error"])
            easy_overwidened = [
                row["channel"]
                for row in selected
                if row["native_picp"] >= level
                and row["marginal_picp"] > row["native_picp"]
                and row["marginal_absolute_coverage_error"] > row["native_absolute_coverage_error"]
            ]
            key = f"{variant}_{int(level * 100)}"
            tail = tail_details[key]
            summary[key] = {
                "worst_native_channels": [row["channel"] for row in worst[:3]],
                "best_native_channels": [row["channel"] for row in best[:3]],
                "qhat_tail_concentration": tail,
                "global_marginal_qhat_overwidens_easy_channels": bool(easy_overwidened),
                "overwidened_easy_channels": easy_overwidened,
            }
            lines.extend(
                [
                    f"## {HEAD_LABELS[variant]} {int(level * 100)}%",
                    "",
                    f"- Worst native channels: {summary[key]['worst_native_channels']}",
                    f"- Best native channels: {summary[key]['best_native_channels']}",
                    f"- Top-two q-hat exceedance channels/share: {tail['top_two_channels']} / {tail['top_two_share']:.3f}",
                    f"- Overwidened easy channels: {easy_overwidened}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n", summary


def main_report(
    audit: Mapping[str, Any],
    native_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    expansion: Sequence[Mapping[str, Any]],
    distribution: Sequence[Mapping[str, Any]],
    counterfactual: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Gate 4c Conformal Calibration Mechanism Audit",
        "",
        "## Status",
        "",
        f"**{audit['status']}**",
        "",
        "This is a read-only diagnostic audit. It does not change the formal Gate 4c decision.",
        "",
        "## Execution Safety",
        "",
        f"- training: `{str(audit['execution']['training']).lower()}`",
        f"- optimizer_created: `{str(audit['execution']['optimizer_created']).lower()}`",
        f"- backward_called: `{str(audit['execution']['backward_called']).lower()}`",
        f"- checkpoint_updated: `{str(audit['execution']['checkpoint_updated']).lower()}`",
        f"- formal_calibrator_updated: `{str(audit['execution']['formal_calibrator_updated']).lower()}`",
        f"- scheduler_started: `{str(audit['execution']['scheduler_started']).lower()}`",
        f"- eval_only_replay: `{str(audit['execution']['eval_only_replay']).lower()}`",
        f"- replay_parity: `{str(audit['replay_parity']['pass']).lower()}`",
        "",
        "## Native Calibration Shift",
        "",
        "| Head | Level | Calibration PICP | Test PICP | Shift | Calibration MPIW | Test MPIW |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in native_rows:
        lines.append(
            f"| {row['head_label']} | {int(row['coverage'] * 100)}% | {row['calibration_picp']:.6f} | "
            f"{row['test_picp']:.6f} | {row['coverage_shift_test_minus_calibration']:+.6f} | "
            f"{row['calibration_mpiw']:.6f} | {row['test_mpiw']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Score Distributions",
            "",
            "| Head | Level | Zero ratio | p90 | p95 | p99 | Max | Formal q-hat |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in score_rows:
        lines.append(
            f"| {row['head_label']} | {int(row['coverage'] * 100)}% | {row['zero_score_ratio']:.6f} | "
            f"{row['p90']:.6f} | {row['p95']:.6f} | {row['p99']:.6f} | {row['max']:.6f} | "
            f"{row['formal_qhat']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Expansion Cost",
            "",
            "| Head | Method | Level | Delta PICP | Delta MPIW | Delta abs error | Delta score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in expansion:
        lines.append(
            f"| {row['head_label']} | {row['method']} | {int(row['coverage'] * 100)}% | "
            f"{row['coverage_change']:+.6f} | {row['width_increase']:+.6f} | "
            f"{row['absolute_coverage_error_change']:+.6f} | {row['interval_score_change']:+.6f} |"
        )
    horizon = audit["horizon_findings"]
    channel = audit["channel_findings"]
    severity = audit["severity_findings"]
    lines.extend(
        [
            "",
            "## Horizon Findings",
            "",
            f"- Worst native coverage horizons: `{horizon['worst_native_coverage_horizons']}`",
            f"- Best native coverage horizons: `{horizon['best_native_coverage_horizons']}`",
            f"- Early/late concentration: `{horizon['concentration_summary']}`",
            f"- Conformal overexpansion pattern: `{horizon['overexpansion_summary']}`",
            "",
            "## Channel Findings",
            "",
            f"- Worst native channels: `{channel['worst_native_channels']}`",
            f"- Best native channels: `{channel['best_native_channels']}`",
            f"- Channels dominating q-hat: `{channel['channels_dominating_qhat']}`",
            f"- Global q-hat overwidth evidence: `{channel['global_qhat_overwidth_evidence']}`",
            "",
            "## Severity Findings",
            "",
            f"- Calibration severity range: `{severity['calibration_severity_range']}`",
            f"- Test severity range: `{severity['test_severity_range']}`",
            f"- Marginal q-hat: `{severity['marginal_qhat']}`",
            f"- Conditioned q-hat range: `{severity['conditioned_qhat_range']}`",
            f"- Max conditioned-minus-marginal difference: `{severity['max_conditioned_minus_marginal_difference']}`",
            f"- Conditioned effect negligible: `{severity['conditioned_effect_negligible']}`",
            "",
            "## Distribution Shift",
            "",
            "| Head | Variable | Calibration mean | Test mean | Relative shift | KS | Wasserstein |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    report_variables = {
        "gaussian_head": ("absolute_residual", "scale", "standardized_residual"),
        "student_t_head": ("absolute_residual", "scale", "standardized_residual", "df"),
    }
    for row in distribution:
        if row["variable"] not in report_variables[row["head"]]:
            continue
        lines.append(
            f"| {row['head_label']} | {row['variable']} | {row['calibration_mean']:.6f} | "
            f"{row['test_mean']:.6f} | {row['relative_mean_shift']:+.6f} | "
            f"{row['ks_statistic']:.6f} | {row['wasserstein_distance']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Counterfactual Shrinkage",
            "",
            "**COUNTERFACTUAL_ONLY / USES_TEST_LABELS / NOT_FORMAL_CALIBRATION**",
            "",
            "| Head | Level | Native PICP | Formal q-hat | Oracle signed delta | Interpretation |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in counterfactual:
        lines.append(
            f"| {row['head_label']} | {int(row['coverage'] * 100)}% | {row['native_picp']:.6f} | "
            f"{row['formal_additive_qhat']:.6f} | {row['oracle_signed_delta']:+.6f} | "
            f"{row['interpretation']} |"
        )
    lines.extend(["", "## Diagnostic Conclusions", ""])
    for key, value in audit["diagnostic_conclusions"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(
        [
            "",
            "## Final Interpretation",
            "",
            audit["final_interpretation"],
            "",
            "The formal state remains `GATE4C_SIGNAL_POSITIVE=False` and "
            "`ENTER_GATE4C_EXPANDED_MATRIX=False`.",
            "",
            "## Protected Formal Artifacts",
            "",
            f"- Bit-identical after audit: `{str(audit['formal_artifacts']['all_hashes_unchanged']).lower()}`",
            "",
            "## Next",
            "",
            "Stop Gate 4c method expansion. Keep Student-t probabilistic forecasting as the main innovation result, "
            "and retain the conformal negative result plus the additive-only limitation in the ablation and limitations analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def write_supporting_markdown(
    score_rows: Sequence[Mapping[str, Any]],
    expansion: Sequence[Mapping[str, Any]],
    severity: Sequence[Mapping[str, Any]],
    distribution: Sequence[Mapping[str, Any]],
    counterfactual: Sequence[Mapping[str, Any]],
    order_rows: Sequence[Mapping[str, Any]],
) -> None:
    score_lines = [
        "# Gate 4c Nonconformity Score Distribution",
        "",
        f"Diagnostic extreme-tail rule: `p99 / max(p90, 1e-12) > {DIAGNOSTIC_RULES['extreme_tail_p99_over_p90_min']}`.",
        "",
    ]
    for row in score_rows:
        score_lines.append(
            f"- {row['head_label']} {int(row['coverage'] * 100)}%: zero={row['zero_score_ratio']:.6f}, "
            f"positive={row['positive_score_ratio']:.6f}, p90={row['p90']:.6f}, p99={row['p99']:.6f}, "
            f"max={row['max']:.6f}, q-hat={row['formal_qhat']:.6f}, extreme-tail={row['extreme_tail_dominated']}."
        )
    atomic_text(REPORT_DIR / "gate4c_mechanism_score_distribution.md", "\n".join(score_lines) + "\n")

    expansion_lines = ["# Gate 4c Additive Expansion Cost", ""]
    for row in expansion:
        expansion_lines.append(
            f"- {row['head_label']} {row['method']} {int(row['coverage'] * 100)}%: "
            f"Delta PICP={row['coverage_change']:+.6f}, Delta MPIW={row['width_increase']:+.6f}, "
            f"Delta ACE={row['absolute_coverage_error_change']:+.6f}, Delta score={row['interval_score_change']:+.6f}."
        )
    atomic_text(REPORT_DIR / "gate4c_mechanism_expansion_cost.md", "\n".join(expansion_lines) + "\n")

    severity_lines = ["# Gate 4c Severity-bin Mechanism", ""]
    for row in severity:
        if row["split"] == "test":
            severity_lines.append(
                f"- {row['head_label']} {int(row['coverage'] * 100)}% bin {row['bin']}: "
                f"windows={row['window_count']}, missing-rate={row['mean_missing_rate']:.6f}, "
                f"native PICP={row['native_picp']:.6f}, marginal q-hat={row['formal_marginal_qhat']:.6f}, "
                f"conditioned q-hat={row['formal_conditioned_qhat']:.6f}."
            )
    atomic_text(REPORT_DIR / "gate4c_mechanism_severity_bins.md", "\n".join(severity_lines) + "\n")

    distribution_lines = [
        "# Gate 4c Calibration/Test Distribution Shift",
        "",
        "These are descriptive effect sizes, not statistical-significance claims.",
        "",
    ]
    for row in distribution:
        distribution_lines.append(
            f"- {row['head_label']} {row['variable']}: calibration mean={row['calibration_mean']:.6f}, "
            f"test mean={row['test_mean']:.6f}, relative shift={row['relative_mean_shift']:+.6f}, "
            f"KS={row['ks_statistic']:.6f}, Wasserstein={row['wasserstein_distance']:.6f}."
        )
    atomic_text(REPORT_DIR / "gate4c_mechanism_distribution_shift.md", "\n".join(distribution_lines) + "\n")

    counter_lines = [
        "# Gate 4c Counterfactual Signed Adjustment",
        "",
        "**COUNTERFACTUAL_ONLY / USES_TEST_LABELS / NOT_FORMAL_CALIBRATION**",
        "",
    ]
    for row in counterfactual:
        counter_lines.append(
            f"- {row['head_label']} {int(row['coverage'] * 100)}%: native PICP={row['native_picp']:.6f}, "
            f"formal q-hat={row['formal_additive_qhat']:.6f}, oracle delta={row['oracle_signed_delta']:+.6f} "
            f"({row['interpretation']})."
        )
    atomic_text(REPORT_DIR / "gate4c_mechanism_counterfactual_shrinkage.md", "\n".join(counter_lines) + "\n")

    order_lines = [
        "# Gate 4c Calibration Order-statistic Audit",
        "",
        f"Sensitivity rule: local `(k-1, k+1)` gap / q-hat > {DIAGNOSTIC_RULES['order_statistic_local_gap_relative_min']}`.",
        "",
    ]
    for row in order_rows:
        order_lines.append(
            f"- {row['head_label']} {int(row['coverage'] * 100)}% {row['method']} bin={row['bin']}: "
            f"n={row['n_calibration_scores']}, k={row['k']}, q-hat={row['qhat']:.6f}, "
            f"local gap={row['local_order_statistic_gap']:.8f}, sensitive={row['qhat_sensitive_to_single_order_statistic']}."
        )
    atomic_text(REPORT_DIR / "gate4c_mechanism_order_statistic.md", "\n".join(order_lines) + "\n")


def execute(device_name: str) -> dict:
    runs = {variant: find_formal_run(variant) for variant in HEADS}
    required_static = [
        ROOT / "results/gate4c_status.json",
        ROOT / "results/gate4c_formal_decision.json",
        ROOT / "results/gate4c_split_audit.json",
        ROOT / "results/gate4c_mask_inventory.json",
        ROOT / "reports/gate4c_interval_metrics.csv",
        ROOT / "reports/gate4c_conditioned_bin_metrics.csv",
        ROOT / "reports/gate4c_calibration_quantiles.csv",
        ROOT / "configs/queues/gate4c_minimal_conformal.yaml",
        SELECTION_INDICES,
        CALIBRATION_INDICES,
        SELECTION_MASK,
        CALIBRATION_MASK,
        TEST_MASK,
    ]
    for path in required_static:
        require_file(path)
    for run in runs.values():
        for relative in (
            "resolved_config.yaml",
            "metrics.json",
            "status.json",
            "checkpoints/best_prob_head.pt",
            "calibration/marginal_qhat.json",
            "calibration/conditioned_qhat.json",
            "calibration/severity_bins.json",
            "calibration/calibration_summary.json",
        ):
            require_file(run / relative)

    # Load and validate selection assets even though no model selection is replayed.
    selection_indices = np.load(SELECTION_INDICES, allow_pickle=False)
    calibration_indices = np.load(CALIBRATION_INDICES, allow_pickle=False)
    selection_mask = np.load(SELECTION_MASK, allow_pickle=False)["mask"]
    if selection_mask.shape[0] != selection_indices.size:
        raise MissingArtifact("selection split/mask count mismatch")
    if np.intersect1d(selection_indices, calibration_indices).size:
        raise MissingArtifact("selection/calibration split overlap")

    protected_paths = formal_paths(runs)
    hashes_before = hashes(protected_paths)
    calibrations = {variant: load_calibration(run) for variant, run in runs.items()}
    if calibrations["gaussian_head"]["bin_boundaries"] != calibrations["student_t_head"]["bin_boundaries"]:
        raise ReplayParityFailure("formal severity boundaries differ between heads")

    all_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    parity: dict[str, Any] = {}
    diagnostic_hashes: dict[str, str] = {}
    for variant, run in runs.items():
        arrays_by_split, variant_parity = replay_variant(variant, run, device_name)
        parity[variant] = variant_parity
        all_data[variant] = {}
        for split, arrays in arrays_by_split.items():
            path = diagnostics_path(variant, split)
            save_npz(path, arrays)
            diagnostic_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
            all_data[variant][split] = arrays

    native_rows = native_shift_rows(all_data)
    score_rows = score_distribution_rows(all_data, calibrations)
    expansion = expansion_rows(all_data)
    horizons = horizon_rows(all_data)
    channels = channel_rows(all_data)
    severity, pairwise_bin_ks = severity_rows(all_data, calibrations)
    distribution = distribution_shift_rows(all_data)
    counterfactual = counterfactual_rows(all_data, calibrations)
    order_rows = order_statistic_rows(all_data, calibrations)
    condition_effects = conditioned_effects(all_data, calibrations)
    channel_tails = channel_tail_concentration(all_data, calibrations)

    write_csv(REPORT_DIR / "gate4c_mechanism_native_shift.csv", native_rows)
    write_csv(REPORT_DIR / "gate4c_mechanism_score_distribution.csv", score_rows)
    write_csv(REPORT_DIR / "gate4c_mechanism_expansion_cost.csv", expansion)
    write_csv(REPORT_DIR / "gate4c_mechanism_horizon_metrics.csv", horizons)
    write_csv(REPORT_DIR / "gate4c_mechanism_channel_metrics.csv", channels)
    write_csv(REPORT_DIR / "gate4c_mechanism_severity_bins.csv", severity)
    write_csv(REPORT_DIR / "gate4c_mechanism_distribution_shift.csv", distribution)
    write_csv(REPORT_DIR / "gate4c_mechanism_counterfactual_shrinkage.csv", counterfactual)
    write_csv(REPORT_DIR / "gate4c_mechanism_order_statistic.csv", order_rows)

    horizon_markdown, horizon_summary = compact_horizon_summary(horizons)
    channel_markdown, channel_summary = compact_channel_summary(channels, channel_tails)
    atomic_text(REPORT_DIR / "gate4c_mechanism_horizon_summary.md", horizon_markdown)
    atomic_text(REPORT_DIR / "gate4c_mechanism_channel_summary.md", channel_markdown)
    write_supporting_markdown(score_rows, expansion, severity, distribution, counterfactual, order_rows)

    calibration_rates = all_data["gaussian_head"]["calibration"]["window_missing_rate"]
    test_rates = all_data["gaussian_head"]["test"]["window_missing_rate"]
    calibration_range = {
        "min": float(calibration_rates.min()),
        "p10": float(np.quantile(calibration_rates, 0.10)),
        "p90": float(np.quantile(calibration_rates, 0.90)),
        "max": float(calibration_rates.max()),
    }
    test_range = {
        "min": float(test_rates.min()),
        "p10": float(np.quantile(test_rates, 0.10)),
        "p90": float(np.quantile(test_rates, 0.90)),
        "max": float(test_rates.max()),
    }
    severity_range_narrow = test_range["p90"] - test_range["p10"] < DIAGNOSTIC_RULES[
        "severity_range_p90_minus_p10_max"
    ]
    per_effect_small = {
        f"{row['head']}_{int(row['coverage'] * 100)}": row["max_absolute_qhat_difference"]
        < DIAGNOSTIC_RULES["conditioned_qhat_relative_effect_small_max"] * abs(row["marginal_qhat"])
        for row in condition_effects
    }
    conditioned_qhat_effect_size_small = all(per_effect_small.values())
    binwise_residual_similar = all(
        value < DIAGNOSTIC_RULES["bin_residual_pairwise_ks_max"] for value in pairwise_bin_ks.values()
    )

    native_mean_ace = {
        variant: float(
            np.mean(
                [
                    abs(row["test_picp"] - row["coverage"])
                    for row in native_rows
                    if row["head"] == variant
                ]
            )
        )
        for variant in HEADS
    }
    native_well_calibrated = all(
        value <= DIAGNOSTIC_RULES["native_well_calibrated_mean_ace_max"]
        for value in native_mean_ace.values()
    )
    standardized_shift_rows = [row for row in distribution if "standardized_residual" in row["variable"]]
    calibration_test_shift = any(row["coverage_distribution_shift_observed"] for row in native_rows) or any(
        abs(row["relative_mean_shift"]) >= DIAGNOSTIC_RULES["distribution_relative_mean_shift_min"]
        for row in standardized_shift_rows
    )
    additive_limitation = all(
        row["native_absolute_coverage_error"] < row["conformal_absolute_coverage_error"]
        and row["conformal_mpiw"] > row["native_mpiw"]
        for row in expansion
    )
    global_overwidth = any(
        item["global_marginal_qhat_overwidens_easy_channels"] for item in channel_summary.values()
    )
    conditioned_negligible = all(row["conditioned_method_effect_negligible"] for row in condition_effects)
    extreme_tail = any(row["extreme_tail_dominated"] for row in score_rows)
    order_sensitive = any(row["qhat_sensitive_to_single_order_statistic"] for row in order_rows)
    diagnostic_conclusions = {
        "NATIVE_ALREADY_WELL_CALIBRATED": native_well_calibrated,
        "CALIBRATION_TEST_DISTRIBUTION_SHIFT": calibration_test_shift,
        "ADDITIVE_ONLY_EXPANSION_LIMITATION": additive_limitation,
        "GLOBAL_QHAT_OVERWIDENS_EASY_SUBGROUPS": global_overwidth,
        "SEVERITY_CONDITION_LOW_INFORMATION": severity_range_narrow and conditioned_qhat_effect_size_small,
        "CONDITIONED_QHAT_EFFECT_NEGLIGIBLE": conditioned_negligible,
        "EXTREME_SCORE_TAIL_DRIVES_QHAT": extreme_tail,
        "FINITE_SAMPLE_ORDER_STATISTIC_SENSITIVITY": order_sensitive,
    }

    worst_horizons = {
        key: value["worst_native_coverage_horizons"] for key, value in horizon_summary.items()
    }
    best_horizons = {key: value["best_native_coverage_horizons"] for key, value in horizon_summary.items()}
    worst_channels = {key: value["worst_native_channels"] for key, value in channel_summary.items()}
    best_channels = {key: value["best_native_channels"] for key, value in channel_summary.items()}
    conditioned_ranges = {
        f"{variant}_{tag}": [
            min(float(item["q_hat"]) for item in calibrations[variant]["conditioned"][tag].values()),
            max(float(item["q_hat"]) for item in calibrations[variant]["conditioned"][tag].values()),
        ]
        for variant in HEADS
        for tag in ("90", "95")
    }
    marginal_qhats = {
        f"{variant}_{tag}": float(calibrations[variant]["marginal"][tag]["q_hat"])
        for variant in HEADS
        for tag in ("90", "95")
    }

    hashes_after = hashes(protected_paths)
    hashes_unchanged = hashes_before == hashes_after
    if not hashes_unchanged:
        raise RuntimeError("a protected formal artifact changed during the audit")

    formal_state = read_json(ROOT / "results/gate4c_formal_decision.json")
    if formal_state.get("GATE4C_SIGNAL_POSITIVE") is not False or formal_state.get(
        "ENTER_GATE4C_EXPANDED_MATRIX"
    ) is not False:
        raise RuntimeError("formal Gate 4c negative decision unexpectedly changed")

    final_interpretation = (
        "Strict split conformal did not outperform the native intervals because the native Gaussian and Student-t "
        "intervals were already close to their target coverage on test, while calibration produced positive additive "
        "q-hat values under a descriptively different error distribution. The one-sided additive correction can only "
        "widen intervals, so it moved test coverage into overcoverage and increased width; severity conditioning changed "
        "bin-specific q-hat values but had negligible aggregate impact and did not repair the worst-bin trade-off. This "
        "is a mechanism diagnosis, not an exchangeability or significance claim."
    )
    audit = {
        "stage": "Gate 4c conformal calibration mechanism audit",
        "status": "COMPLETE",
        "created_at": now(),
        "formal_gate_unchanged": True,
        "GATE4C_SIGNAL_POSITIVE": False,
        "ENTER_GATE4C_EXPANDED_MATRIX": False,
        "execution": {
            "mode": "EVAL_ONLY_REPLAY_IF_REQUIRED",
            "device": device_name,
            "training": False,
            "optimizer_created": False,
            "backward_called": False,
            "checkpoint_updated": False,
            "formal_calibrator_updated": False,
            "scheduler_started": False,
            "eval_only_replay": True,
            "diagnostic_replay_required": True,
            "formal_metrics_unchanged": True,
        },
        "formal_runs": {variant: str(path.relative_to(ROOT)) for variant, path in runs.items()},
        "asset_audit": {
            "selection_windows": int(selection_indices.size),
            "calibration_windows": int(calibration_indices.size),
            "test_windows": int(all_data["gaussian_head"]["test"]["target"].shape[0]),
            "selection_calibration_index_overlap": int(np.intersect1d(selection_indices, calibration_indices).size),
            "diagnostic_artifact_hashes": diagnostic_hashes,
        },
        "replay_parity": {"pass": all(item["pass"] for item in parity.values()), "by_head": parity},
        "native_shift": native_rows,
        "score_distributions": score_rows,
        "expansion_cost": expansion,
        "horizon_findings": {
            "worst_native_coverage_horizons": worst_horizons,
            "best_native_coverage_horizons": best_horizons,
            "concentration_summary": {
                key: value["largest_error_third"] for key, value in horizon_summary.items()
            },
            "overexpansion_summary": {
                key: value["fraction_horizons_marginal_worsens_ace"] for key, value in horizon_summary.items()
            },
            "details": horizon_summary,
        },
        "channel_findings": {
            "worst_native_channels": worst_channels,
            "best_native_channels": best_channels,
            "channels_dominating_qhat": channel_tails,
            "global_qhat_overwidth_evidence": {
                key: value["overwidened_easy_channels"] for key, value in channel_summary.items()
            },
            "details": channel_summary,
        },
        "severity_findings": {
            "boundaries": calibrations["gaussian_head"]["bin_boundaries"],
            "calibration_severity_range": calibration_range,
            "test_severity_range": test_range,
            "severity_range_narrow": severity_range_narrow,
            "marginal_qhat": marginal_qhats,
            "conditioned_qhat_range": conditioned_ranges,
            "max_conditioned_minus_marginal_difference": max(
                row["max_absolute_qhat_difference"] for row in condition_effects
            ),
            "conditioned_qhat_effect_size_small": conditioned_qhat_effect_size_small,
            "conditioned_qhat_effect_small_by_head_level": per_effect_small,
            "binwise_residual_distributions_similar": binwise_residual_similar,
            "pairwise_bin_standardized_residual_ks_max": pairwise_bin_ks,
            "conditioned_effect_negligible": conditioned_negligible,
        },
        "distribution_shift": distribution,
        "counterfactual_shrinkage": {
            "COUNTERFACTUAL_ONLY": True,
            "USES_TEST_LABELS": True,
            "NOT_FORMAL_CALIBRATION": True,
            "formal_gate_uses_result": False,
            "rows": counterfactual,
        },
        "marginal_conditioned_effects": condition_effects,
        "order_statistic": order_rows,
        "diagnostic_rules": DIAGNOSTIC_RULES,
        "native_mean_absolute_coverage_error_by_head": native_mean_ace,
        "diagnostic_conclusions": diagnostic_conclusions,
        "formal_artifacts": {
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "all_hashes_unchanged": hashes_unchanged,
        },
        "upstream": {
            "commit": command_output(["git", "-C", str(ROOT / "upstream/TimePerceiver"), "rev-parse", "HEAD"]),
            "tracked_diff": command_output(["git", "-C", str(ROOT / "upstream/TimePerceiver"), "diff", "--stat"]),
            "untracked": command_output(
                ["git", "-C", str(ROOT / "upstream/TimePerceiver"), "status", "--short"]
            ),
        },
        "final_interpretation": final_interpretation,
        "next": [
            "STOP_GATE4C_METHOD_EXPANSION",
            "KEEP_STUDENT_T_PROBABILISTIC_FORECASTING_AS_MAIN_INNOVATION",
            "REPORT_CONFORMAL_NEGATIVE_RESULT_AND_ADDITIVE_ONLY_LIMITATION",
        ],
    }
    write_json(ARTIFACT_DIR / "metadata.json", {
        "created_at": audit["created_at"],
        "source_runs": audit["formal_runs"],
        "checkpoint_sha256": {
            variant: sha256_file(runs[variant] / "checkpoints/best_prob_head.pt") for variant in HEADS
        },
        "split_sha256": {
            "selection": sha256_file(SELECTION_INDICES),
            "calibration": sha256_file(CALIBRATION_INDICES),
        },
        "mask_sha256": {
            "selection": sha256_file(SELECTION_MASK),
            "calibration": sha256_file(CALIBRATION_MASK),
            "test": sha256_file(TEST_MASK),
        },
        "formal_qhat_sha256": {
            variant: {
                "marginal": sha256_file(runs[variant] / "calibration/marginal_qhat.json"),
                "conditioned": sha256_file(runs[variant] / "calibration/conditioned_qhat.json"),
            }
            for variant in HEADS
        },
        "formal_bin_boundary_sha256": {
            variant: sha256_file(runs[variant] / "calibration/severity_bins.json") for variant in HEADS
        },
        "diagnostic_artifact_hashes": diagnostic_hashes,
        "eval_only": True,
        "training": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_updated": False,
        "formal_calibrator_updated": False,
        "formal_artifacts_modified": False,
    })
    write_json(RESULT_DIR / "gate4c_mechanism_audit.json", audit)
    atomic_text(
        REPORT_DIR / "gate4c_mechanism_audit.md",
        main_report(audit, native_rows, score_rows, expansion, distribution, counterfactual),
    )
    return audit


def write_blocked(status: str, error: Exception) -> None:
    payload = {
        "stage": "Gate 4c conformal calibration mechanism audit",
        "status": status,
        "created_at": now(),
        "error": repr(error),
        "execution": {
            "training": False,
            "optimizer_created": False,
            "backward_called": False,
            "checkpoint_updated": False,
            "formal_calibrator_updated": False,
            "scheduler_started": False,
        },
        "GATE4C_SIGNAL_POSITIVE": False,
        "ENTER_GATE4C_EXPANDED_MATRIX": False,
    }
    write_json(RESULT_DIR / "gate4c_mechanism_audit.json", payload)
    atomic_text(
        REPORT_DIR / "gate4c_mechanism_audit.md",
        f"# Gate 4c Mechanism Audit\n\n**{status}**\n\n`{error!r}`\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        audit = execute(args.device)
    except MissingArtifact as error:
        write_blocked("BLOCKED_GATE4C_MECHANISM_AUDIT_MISSING_ARTIFACT", error)
        print(f"BLOCKED_GATE4C_MECHANISM_AUDIT_MISSING_ARTIFACT: {error}")
        return 2
    except ReplayParityFailure as error:
        write_blocked("BLOCKED_GATE4C_MECHANISM_REPLAY_PARITY_FAILURE", error)
        print(f"BLOCKED_GATE4C_MECHANISM_REPLAY_PARITY_FAILURE: {error}")
        return 3
    except Exception as error:  # pragma: no cover - preserves a durable failure record
        write_blocked("FAILED", error)
        raise
    print(json.dumps({
        "status": audit["status"],
        "replay_parity": audit["replay_parity"]["pass"],
        "diagnostic_conclusions": audit["diagnostic_conclusions"],
        "formal_hashes_unchanged": audit["formal_artifacts"]["all_hashes_unchanged"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
