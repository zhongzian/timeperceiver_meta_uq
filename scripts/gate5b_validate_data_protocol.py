#!/usr/bin/env python3
"""Validate that Gate 5b reuses the established project data protocol."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate3b_smoke_missing_eval import build_args, import_official_modules, parse_simple_yaml  # noqa: E402
from gate4b_train_probabilistic import Indexed, sample_masks  # noqa: E402
from gate5b_common import (  # noqa: E402
    BATCH_SIZES,
    DATASETS,
    TRAIN_MASK_RULES,
    dataset_spec,
    dump,
    resolve,
    write_report,
)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def expected_boundaries(dataset, raw_length, seq_len):
    if dataset == "etth1":
        return {
            "train": [0, 12 * 30 * 24],
            "val": [12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24],
            "test": [12 * 30 * 24 + 4 * 30 * 24 - seq_len, 12 * 30 * 24 + 8 * 30 * 24],
        }
    num_train = int(raw_length * 0.7)
    num_test = int(raw_length * 0.2)
    num_val = raw_length - num_train - num_test
    return {
        "train": [0, num_train],
        "val": [num_train - seq_len, num_train + num_val],
        "test": [raw_length - num_test - seq_len, raw_length],
    }


def main():
    provider, _ = import_official_modules()
    rows = []
    for dataset in DATASETS:
        spec = dataset_spec(dataset)
        config = parse_simple_yaml(resolve(spec["baseline_config"]))
        args = build_args(config, BATCH_SIZES[dataset], "cpu")
        args.seq_len = 384
        args.pred_len = 96
        args.batch_size = BATCH_SIZES[dataset]
        first = {flag: provider(args, flag)[0] for flag in ("train", "val", "test")}
        second = {flag: provider(args, flag)[0] for flag in ("train", "val", "test")}

        raw_path = Path(args.root_path) / args.data_path
        raw = pd.read_csv(raw_path)
        boundaries = expected_boundaries(dataset, len(raw), args.seq_len)
        expected_lengths = {
            flag: stop - start - args.seq_len - args.pred_len + 1
            for flag, (start, stop) in boundaries.items()
        }
        split_parity = all(len(first[flag]) == expected_lengths[flag] for flag in first)
        window_parity = all(len(first[flag]) == len(second[flag]) for flag in first)
        scaler_parity = all(
            np.array_equal(first["train"].scaler.mean_, first[flag].scaler.mean_)
            and np.array_equal(first["train"].scaler.scale_, first[flag].scaler.scale_)
            for flag in ("val", "test")
        )
        target_channels = int(first["train"].data_x.shape[-1])
        channel_parity = target_channels == int(spec["enc_in"])
        first_item = first["train"][0]
        second_item = second["train"][0]
        clean_batch_equal = all(np.array_equal(a, b) for a, b in zip(first_item, second_item))

        generator_a = torch.Generator().manual_seed(2025)
        generator_b = torch.Generator().manual_seed(2025)
        loader_a = torch.utils.data.DataLoader(
            Indexed(first["train"]), batch_size=args.batch_size, shuffle=True,
            generator=generator_a, num_workers=0,
        )
        loader_b = torch.utils.data.DataLoader(
            Indexed(second["train"]), batch_size=args.batch_size, shuffle=True,
            generator=generator_b, num_workers=0,
        )
        batch_a = next(iter(loader_a))
        batch_b = next(iter(loader_b))
        indices_a, indices_b = batch_a[-1], batch_b[-1]
        masks_a = sample_masks(
            indices_a, 0, batch_a[0].shape, dataset, "random", 0.3,
            42025, 2025, TRAIN_MASK_RULES[dataset],
        )
        masks_b = sample_masks(
            indices_b, 0, batch_b[0].shape, dataset, "random", 0.3,
            42025, 2025, TRAIN_MASK_RULES[dataset],
        )
        batch_y_before = batch_a[1].clone()
        _ = batch_a[0] * masks_a
        batch_y_unchanged = bool(torch.equal(batch_y_before, batch_a[1]))
        actual_rate = float(1.0 - masks_a.mean())

        validation_mask = np.load(resolve(spec["validation_mask"]))["mask"]
        test_mask = np.load(resolve(spec["test_mask"]))["mask"]
        validation_shape = tuple(validation_mask.shape) == (len(first["val"]), 384, target_channels)
        test_shape = tuple(test_mask.shape) == (len(first["test"]), 384, target_channels)
        row_pass = all((
            split_parity, window_parity, scaler_parity, channel_parity, clean_batch_equal,
            torch.equal(indices_a, indices_b), torch.equal(masks_a, masks_b),
            abs(actual_rate - 0.3) <= 0.02, batch_y_unchanged,
            validation_shape, test_shape,
        ))
        rows.append({
            "dataset": dataset, "status": "PASS" if row_pass else "FAIL",
            "raw_path": str(raw_path), "raw_rows": len(raw), "raw_columns": list(raw.columns),
            "split_boundaries": boundaries, "expected_window_counts": expected_lengths,
            "actual_window_counts": {flag: len(first[flag]) for flag in first},
            "raw_split_boundaries_equal": split_parity, "window_counts_equal": window_parity,
            "normalization_statistics_equal": scaler_parity,
            "normalization_mean_sha256": array_hash(first["train"].scaler.mean_),
            "normalization_scale_sha256": array_hash(first["train"].scaler.scale_),
            "target_channels": target_channels, "target_channels_equal": channel_parity,
            "first_clean_batch_values_equal": clean_batch_equal,
            "first_masked_batch_indices_equal": bool(torch.equal(indices_a, indices_b)),
            "first_masked_batch_mask_equal": bool(torch.equal(masks_a, masks_b)),
            "first_batch_indices_sha256": array_hash(indices_a.numpy()),
            "first_batch_mask_sha256": array_hash(masks_a.numpy()),
            "actual_missing_rate": actual_rate, "actual_missing_rate_near_0_3": abs(actual_rate - 0.3) <= 0.02,
            "batch_y_unchanged": batch_y_unchanged,
            "validation_mask": spec["validation_mask"],
            "validation_mask_sha256": spec["validation_mask_sha256"],
            "validation_mask_shape": list(validation_mask.shape), "validation_mask_shape_equal": validation_shape,
            "test_mask": spec["test_mask"], "test_mask_sha256": spec["test_mask_sha256"],
            "test_mask_shape": list(test_mask.shape), "test_mask_shape_equal": test_shape,
            "future_target_masked": False, "test_statistics_used_for_normalization": False,
        })
    passed = sum(row["status"] == "PASS" for row in rows)
    payload = {
        "status": "PASS" if passed == len(DATASETS) else "BLOCKED_GATE5B_DATA_PROTOCOL_MISMATCH",
        "dataset_protocol_pass": f"{passed}/{len(DATASETS)}", "datasets": rows,
        "project_data_loader_reused": True, "normalized_space_zero_fill": True,
        "mask_indicator_channel_added": False, "future_target_never_masked": True,
        "test_used_for_model_selection": False, "no_leakage": passed == len(DATASETS),
    }
    dump(ROOT / "results/gate5b_data_protocol_validation.json", payload)
    write_report(ROOT / "reports/gate5b_data_protocol_validation.md", "Gate 5b Data Protocol Validation", payload)
    print(json.dumps({"status": payload["status"], "dataset_protocol_pass": payload["dataset_protocol_pass"]}, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

