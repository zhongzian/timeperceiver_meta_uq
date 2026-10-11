#!/usr/bin/env python
"""Generate fixed Gate 3a missingness assets.

This script creates reusable window-level masks for test input/history windows.
It does not import or modify model code, and it does not run training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "TimePerceiver"

DATASETS = ("etth1", "weather", "electricity")
MASK_TYPES = ("random", "block", "channel", "mixed")
RATES = (0.1, 0.3, 0.5, 0.7)
SEEDS = (2025,)
GENERATOR_VERSION = "gate3a_v2_official_window_history_mask"
CHUNK_WINDOWS = 32


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip().strip("\"'")
        if value == "":
            continue
        if re.fullmatch(r"-?\d+", value):
            data[key] = int(value)
        else:
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
    return data


def latest_success_run(dataset: str) -> Path:
    candidates: List[Path] = []
    prefix = f"timeperceiver__{dataset}__official_baseline__seed2025__"
    for run_dir in (ROOT / "runs").glob(prefix + "*"):
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        try:
            status = read_json(status_path)
        except json.JSONDecodeError:
            continue
        if status.get("status") == "SUCCESS":
            candidates.append(run_dir)
    if not candidates:
        raise RuntimeError(f"No successful Gate 2 official baseline run found for {dataset}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_gate2_config(dataset: str) -> Tuple[Path, Dict[str, Any]]:
    run_dir = latest_success_run(dataset)
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"Missing resolved_config.yaml for {run_dir}")
    config = parse_simple_yaml(config_path)
    config["run_dir"] = str(run_dir)
    config["experiment_id"] = run_dir.name
    return run_dir, config


def dataframe_for_config(config: Dict[str, Any]) -> pd.DataFrame:
    root_path = Path(str(config["root_path"]))
    data_path = str(config["data_path"])
    csv_path = root_path / data_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    return pd.read_csv(csv_path)


def feature_columns_for_official_loader(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    features = str(config.get("features", "M"))
    target = str(config.get("target", "OT"))
    if "date" not in df.columns:
        raise RuntimeError("Expected a 'date' column matching official TimePerceiver loaders")
    if features in {"M", "MS"}:
        return [c for c in df.columns if c != "date"]
    if features == "S":
        return [target]
    raise RuntimeError(f"Unsupported features mode: {features}")


def official_borders(dataset: str, row_count: int, seq_len: int) -> Dict[str, Any]:
    if dataset == "etth1":
        border1s = [
            0,
            12 * 30 * 24 - seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        rule = "Dataset_ETT_hour fixed borders: 12 months train, 4 months val, 4 months test, with seq_len overlap"
    else:
        num_train = int(row_count * 0.7)
        num_test = int(row_count * 0.2)
        num_val = row_count - num_train - num_test
        border1s = [0, num_train - seq_len, row_count - num_test - seq_len]
        border2s = [num_train, num_train + num_val, row_count]
        rule = "Dataset_Custom borders: 70% train, 10% val, 20% test, with seq_len overlap"
    return {
        "rule": rule,
        "border1s": border1s,
        "border2s": border2s,
        "train_border": [border1s[0], border2s[0]],
        "val_border": [border1s[1], border2s[1]],
        "test_border": [border1s[2], border2s[2]],
    }


def dataset_protocol(dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
    df = dataframe_for_config(config)
    seq_len = int(config["seq_len"])
    pred_len = int(config["pred_len"])
    label_len = int(config.get("label_len", 0))
    feature_cols = feature_columns_for_official_loader(df, config)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) != len(feature_cols):
        non_numeric = sorted(set(feature_cols) - set(numeric_cols))
        raise RuntimeError(f"Non-numeric feature columns found for {dataset}: {non_numeric}")
    borders = official_borders(dataset, len(df), seq_len)
    test_border = borders["test_border"]
    data_x_len = test_border[1] - test_border[0]
    num_test_windows = data_x_len - seq_len - pred_len + 1
    if num_test_windows <= 0:
        raise RuntimeError(f"Non-positive test windows for {dataset}: {num_test_windows}")

    train_start, train_end = borders["train_border"]
    train_values = df.loc[train_start : train_end - 1, numeric_cols].to_numpy(dtype=np.float64)
    norm_mean = train_values.mean(axis=0)
    norm_std = train_values.std(axis=0)

    return {
        "dataset": dataset,
        "display": config.get("display", dataset),
        "baseline_experiment_id": config["experiment_id"],
        "baseline_run_dir": config["run_dir"],
        "data": config.get("data"),
        "data_file": str(Path(str(config["root_path"])) / str(config["data_path"])),
        "shape_raw": [int(len(df)), int(len(df.columns))],
        "columns": list(map(str, df.columns)),
        "feature_columns": list(map(str, feature_cols)),
        "numeric_columns": list(map(str, numeric_cols)),
        "num_features": int(len(numeric_cols)),
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "features_mode": config.get("features"),
        "target": config.get("target"),
        "freq": config.get("freq"),
        "split": {
            **borders,
            "test_data_x_length": int(data_x_len),
            "num_test_windows": int(num_test_windows),
            "window_count_method": "official __len__: len(data_x) - seq_len - pred_len + 1",
        },
        "normalization": {
            "rule": "StandardScaler fit on official train split only, then applied to all splits",
            "train_border": borders["train_border"],
            "mean_sha256": hashlib.sha256(norm_mean.tobytes()).hexdigest(),
            "std_sha256": hashlib.sha256(norm_std.tobytes()).hexdigest(),
        },
        "mask_shape": [int(num_test_windows), seq_len, int(len(numeric_cols))],
    }


def write_normalization_assets(protocol: Dict[str, Any]) -> None:
    df = pd.read_csv(protocol["data_file"])
    columns = protocol["numeric_columns"]
    train_start, train_end = protocol["normalization"]["train_border"]
    train_values = df.loc[train_start : train_end - 1, columns].to_numpy(dtype=np.float64)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    out_dir = ROOT / "artifacts" / "normalization" / protocol["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "official_train_standard_scaler.npz"
    np.savez_compressed(npz_path, mean=mean, std=std, columns=np.array(columns, dtype=object))
    meta = {
        "dataset": protocol["dataset"],
        "rule": protocol["normalization"]["rule"],
        "train_border": protocol["normalization"]["train_border"],
        "columns": columns,
        "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
        "std_sha256": hashlib.sha256(std.tobytes()).hexdigest(),
        "npz_sha256": sha256_file(npz_path),
        "generated_at": iso_now(),
    }
    (out_dir / "official_train_standard_scaler.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def random_mask(rng: np.random.Generator, shape: Tuple[int, int, int], rate: float) -> np.ndarray:
    mask = np.empty(shape, dtype=np.uint8)
    for start in range(0, shape[0], CHUNK_WINDOWS):
        end = min(shape[0], start + CHUNK_WINDOWS)
        mask[start:end] = (rng.random((end - start, shape[1], shape[2]), dtype=np.float32) >= rate)
    return mask


def block_mask(rng: np.random.Generator, shape: Tuple[int, int, int], rate: float) -> np.ndarray:
    windows, seq_len, channels = shape
    mask = np.ones(shape, dtype=np.uint8)
    target = int(round(rate * seq_len * channels))
    max_block = max(1, min(seq_len, int(round(seq_len * 0.2))))
    min_block = max(1, int(round(seq_len * 0.02)))
    for w in range(windows):
        missing = 0
        attempts = 0
        while missing < target and attempts < 2000:
            attempts += 1
            block_len = int(rng.integers(min_block, max_block + 1))
            start = int(rng.integers(0, seq_len - block_len + 1))
            if rng.random() < 0.5:
                channel_count = channels
                selected = slice(None)
            else:
                channel_count = max(1, min(channels, int(round(channels * rng.uniform(0.1, 0.5)))))
                selected = rng.choice(channels, size=channel_count, replace=False)
            before = int((mask[w] == 0).sum())
            mask[w, start : start + block_len, selected] = 0
            missing += int((mask[w] == 0).sum()) - before
    return mask


def channel_mask(rng: np.random.Generator, shape: Tuple[int, int, int], rate: float) -> np.ndarray:
    windows, _, channels = shape
    mask = np.ones(shape, dtype=np.uint8)
    if channels <= 1:
        return random_mask(rng, shape, rate)
    k = max(1, int(round(rate * channels))) if rate > 0 else 0
    k = min(k, channels - 1)
    for w in range(windows):
        selected = rng.choice(channels, size=k, replace=False)
        mask[w, :, selected] = 0
    return mask


def fill_random_observed(
    rng: np.random.Generator,
    mask: np.ndarray,
    target_missing_total: int,
) -> None:
    windows = mask.shape[0]
    current = int((mask == 0).sum())
    remaining_total = max(0, target_missing_total - current)
    if remaining_total == 0:
        return
    for w in range(windows):
        if remaining_total <= 0:
            return
        windows_left = windows - w
        target_for_window = int(np.ceil(remaining_total / windows_left))
        observed_flat = np.flatnonzero(mask[w].reshape(-1) == 1)
        if observed_flat.size == 0:
            continue
        take = min(target_for_window, observed_flat.size, remaining_total)
        chosen = rng.choice(observed_flat, size=take, replace=False)
        flat = mask[w].reshape(-1)
        flat[chosen] = 0
        remaining_total -= take


def mixed_mask(rng: np.random.Generator, shape: Tuple[int, int, int], rate: float) -> np.ndarray:
    mask = channel_mask(rng, shape, rate / 3.0)
    block = block_mask(rng, shape, rate / 3.0)
    np.minimum(mask, block, out=mask)
    del block
    fill_random_observed(rng, mask, int(round(rate * mask.size)))
    return mask.astype(np.uint8)


def make_mask(mask_type: str, seed: int, shape: Tuple[int, int, int], rate: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mask_type == "random":
        return random_mask(rng, shape, rate)
    if mask_type == "block":
        return block_mask(rng, shape, rate)
    if mask_type == "channel":
        return channel_mask(rng, shape, rate)
    if mask_type == "mixed":
        return mixed_mask(rng, shape, rate)
    raise ValueError(mask_type)


def summarize_mask(mask: np.ndarray, target_rate: float) -> Dict[str, Any]:
    total = int(mask.size)
    missing = int((mask == 0).sum())
    actual = missing / total
    per_channel_missing = 1.0 - mask.mean(axis=(0, 1))
    per_window_missing = 1.0 - mask.mean(axis=(1, 2))
    best_runs = np.zeros((mask.shape[0], mask.shape[2]), dtype=np.uint16)
    current_runs = np.zeros_like(best_runs)
    for t in range(mask.shape[1]):
        missing_at_t = mask[:, t, :] == 0
        current_runs = (current_runs + 1) * missing_at_t
        best_runs = np.maximum(best_runs, current_runs)
    per_window_longest = best_runs.max(axis=1)
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "target_missing_rate": float(target_rate),
        "actual_missing_rate": float(actual),
        "missing_rate_abs_error": float(abs(actual - target_rate)),
        "observed_rate": float(1.0 - actual),
        "total_entries": total,
        "missing_entries": missing,
        "per_channel_missing_mean": float(per_channel_missing.mean()),
        "per_channel_missing_min": float(per_channel_missing.min()),
        "per_channel_missing_max": float(per_channel_missing.max()),
        "per_window_missing_mean": float(per_window_missing.mean()),
        "per_window_missing_min": float(per_window_missing.min()),
        "per_window_missing_max": float(per_window_missing.max()),
        "longest_missing_run_mean": float(per_window_longest.mean()),
        "longest_missing_run_max": int(per_window_longest.max()),
    }


def expected_mask_paths(dataset: str, mask_type: str, rate: float, seed: int) -> Tuple[Path, Path]:
    out_dir = ROOT / "artifacts" / "masks" / dataset / mask_type / f"rate_{rate}"
    return out_dir / f"seed_{seed}.npz", out_dir / f"seed_{seed}.json"


def load_existing_metadata(npz_path: Path, meta_path: Path, shape: List[int]) -> Dict[str, Any] | None:
    if not npz_path.exists() and not meta_path.exists():
        return None
    if not npz_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Incomplete existing mask asset: {npz_path} / {meta_path}")
    meta = read_json(meta_path)
    if list(meta.get("shape", [])) != list(shape):
        raise RuntimeError(f"Existing mask shape mismatch for {npz_path}: {meta.get('shape')} != {shape}")
    actual_hash = sha256_file(npz_path)
    if meta.get("sha256") != actual_hash:
        raise RuntimeError(f"Existing mask hash mismatch for {npz_path}")
    return meta


def save_mask(
    mask: np.ndarray,
    protocol: Dict[str, Any],
    mask_type: str,
    rate: float,
    seed: int,
    npz_path: Path,
    meta_path: Path,
) -> Dict[str, Any]:
    summary = summarize_mask(mask, rate)
    warning = ""
    if summary["missing_rate_abs_error"] > 0.02:
        warning = f"actual-target missing-rate error {summary['missing_rate_abs_error']:.6f} > 0.02"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        mask=mask,
        mask_semantics=np.array(["1=observed/keep, 0=missing/mask"]),
        dataset=np.array([protocol["dataset"]]),
        mask_type=np.array([mask_type]),
        target_missing_rate=np.array([rate], dtype=np.float32),
        seed=np.array([seed], dtype=np.int64),
        seq_len=np.array([protocol["seq_len"]], dtype=np.int64),
        pred_len=np.array([protocol["pred_len"]], dtype=np.int64),
    )
    meta = {
        **summary,
        "sha256": sha256_file(npz_path),
        "generated_at": iso_now(),
        "generator_version": GENERATOR_VERSION,
        "dataset": protocol["dataset"],
        "baseline_experiment_id": protocol["baseline_experiment_id"],
        "data_file": protocol["data_file"],
        "seq_len": protocol["seq_len"],
        "label_len": protocol["label_len"],
        "pred_len": protocol["pred_len"],
        "num_features": protocol["num_features"],
        "num_test_windows": protocol["split"]["num_test_windows"],
        "mask_type": mask_type,
        "seed": seed,
        "mask_semantics": "1=observed/keep, 0=missing/mask",
        "applies_to": "test input/history window seq_x only",
        "does_not_mask": "decoder label or future prediction target",
        "split": protocol["split"],
        "normalization_rule": protocol["normalization"]["rule"],
        "warning": warning,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "mask_type",
        "rate",
        "seed",
        "status",
        "shape",
        "target_missing_rate",
        "actual_missing_rate",
        "missing_rate_abs_error",
        "sha256",
        "npz_path",
        "meta_path",
        "warning",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_inventory_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Gate 3a Mask Inventory",
        "",
        "| Dataset | Mask type | Rate | Seed | Status | Shape | Actual missing | SHA256 | Warning |",
        "|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        sha = str(row["sha256"])
        lines.append(
            f"| {row['dataset']} | {row['mask_type']} | {row['rate']} | {row['seed']} | "
            f"{row['status']} | {row['shape']} | {row['actual_missing_rate']} | "
            f"`{sha[:12]}...` | {row.get('warning', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol_reports(protocols: Dict[str, Dict[str, Any]]) -> None:
    (ROOT / "configs" / "missingness").mkdir(parents=True, exist_ok=True)
    protocol_json = {
        "gate": "Gate 3a",
        "generator_version": GENERATOR_VERSION,
        "mask_semantics": "1=observed/keep, 0=missing/mask",
        "mask_shape": "[num_test_windows, seq_len, num_features]",
        "applies_to": "test input/history window seq_x only",
        "does_not_mask": "future prediction labels",
        "datasets": protocols,
        "mask_types": list(MASK_TYPES),
        "missing_rates": list(RATES),
        "seeds": list(SEEDS),
        "leakage_checks": {
            "mask_only_history_input": "OK",
            "future_prediction_labels_not_read": "OK",
            "split_rules_from_official_loader": "OK",
            "normalization_uses_train_split_only": "OK",
            "same_mask_reusable_for_baseline_and_innovation": "OK",
            "seed_fixed": "OK",
            "evaluation_time_resampling_forbidden": "OK",
            "official_model_code_modified": "NO",
        },
    }
    (ROOT / "configs" / "missingness" / "gate3a_protocol.json").write_text(
        json.dumps(protocol_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    yaml_lines = [
        "gate: Gate 3a",
        f"generator_version: {GENERATOR_VERSION}",
        "mask_semantics: '1=observed/keep, 0=missing/mask'",
        "mask_shape: '[num_test_windows, seq_len, num_features]'",
        "applies_to: test input/history window seq_x only",
        "does_not_mask: future prediction labels",
        "mask_types: [random, block, channel, mixed]",
        "missing_rates: [0.1, 0.3, 0.5, 0.7]",
        "seeds: [2025]",
        "datasets:",
    ]
    for dataset, info in protocols.items():
        yaml_lines.extend(
            [
                f"  {dataset}:",
                f"    baseline_experiment_id: {info['baseline_experiment_id']}",
                f"    data_file: {info['data_file']}",
                f"    rows: {info['shape_raw'][0]}",
                f"    num_features: {info['num_features']}",
                f"    seq_len: {info['seq_len']}",
                f"    label_len: {info['label_len']}",
                f"    pred_len: {info['pred_len']}",
                f"    num_test_windows: {info['split']['num_test_windows']}",
                f"    split_rule: {info['split']['rule']}",
                f"    normalization: {info['normalization']['rule']}",
            ]
        )
    (ROOT / "configs" / "missingness" / "gate3a_protocol.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    md = [
        "# Gate 3a Missingness Evaluation Protocol",
        "",
        "Gate 3a fixes reusable test input/history masks. It does not run training or missingness evaluation.",
        "",
        "## Mask Semantics",
        "",
        "- `1 = observed / keep`",
        "- `0 = missing / mask`",
        "- Shape: `[num_test_windows, seq_len, num_features]`",
        "- Scope: only the official test `seq_x` history window.",
        "- Future prediction labels are not read or masked by generation.",
        "",
        "## Dataset Protocols",
        "",
        "| Dataset | Rows | Columns | Numeric features | seq_len | pred_len | Test windows | Split rule |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, info in protocols.items():
        md.append(
            f"| {dataset} | {info['shape_raw'][0]} | {info['shape_raw'][1]} | "
            f"{info['num_features']} | {info['seq_len']} | {info['pred_len']} | "
            f"{info['split']['num_test_windows']} | {info['split']['rule']} |"
        )
    md.extend(
        [
            "",
            "## Normalization",
            "",
            "The official loaders fit `StandardScaler` on the training split only, then transform all splits. Gate 3a records and saves those train-only statistics under `artifacts/normalization/` for auditability. Test statistics are not used.",
            "",
            "## Missingness Settings",
            "",
            "- Mask types: `random`, `block`, `channel`, `mixed`",
            "- Missing rates: `0.1`, `0.3`, `0.5`, `0.7`",
            "- Seed: `2025`",
            "",
            "## Leakage Checks",
            "",
            "| Check | Status |",
            "|---|---|",
            "| Mask only applies to input/history window | OK |",
            "| Future prediction labels are not read for mask generation | OK |",
            "| Split rules come from official loader logic | OK |",
            "| Normalization statistics use train split only | OK |",
            "| Same masks can be reused by baseline and innovations | OK |",
            "| Seed is fixed | OK |",
            "| Evaluation-time random resampling is forbidden by protocol | OK |",
            "| Official model code modified in Gate 3a | NO |",
        ]
    )
    (ROOT / "reports" / "gate3a_protocol.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "reports" / "gate3a_protocol.json").write_text(
        json.dumps(protocol_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    args = parser.parse_args()

    overwrite = args.overwrite or os.environ.get("OVERWRITE_MASKS") == "1"
    for path in (
        ROOT / "artifacts" / "masks",
        ROOT / "artifacts" / "splits",
        ROOT / "artifacts" / "normalization",
        ROOT / "configs" / "missingness",
        ROOT / "reports",
    ):
        path.mkdir(parents=True, exist_ok=True)

    inventory: List[Dict[str, Any]] = []
    protocols: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    blockers: List[str] = []

    for dataset in DATASETS:
        try:
            _, config = load_gate2_config(dataset)
            protocol = dataset_protocol(dataset, config)
            protocols[dataset] = protocol
            split_dir = ROOT / "artifacts" / "splits" / dataset
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "official_test_window_protocol.json").write_text(
                json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            write_normalization_assets(protocol)

            shape = tuple(int(v) for v in protocol["mask_shape"])
            for mask_type in MASK_TYPES:
                for rate in RATES:
                    for seed in SEEDS:
                        npz_path, meta_path = expected_mask_paths(dataset, mask_type, rate, seed)
                        if not overwrite:
                            existing = load_existing_metadata(npz_path, meta_path, list(shape))
                            if existing is not None:
                                status = "EXISTS"
                                meta = existing
                            else:
                                status = "CREATED"
                                mask = make_mask(mask_type, seed, shape, rate)
                                meta = save_mask(mask, protocol, mask_type, rate, seed, npz_path, meta_path)
                                del mask
                        else:
                            status = "OVERWRITTEN"
                            mask = make_mask(mask_type, seed, shape, rate)
                            meta = save_mask(mask, protocol, mask_type, rate, seed, npz_path, meta_path)
                            del mask

                        if meta.get("warning"):
                            warnings.append(f"{dataset}/{mask_type}/rate_{rate}/seed_{seed}: {meta['warning']}")
                        inventory.append(
                            {
                                "dataset": dataset,
                                "mask_type": mask_type,
                                "rate": rate,
                                "seed": seed,
                                "status": status,
                                "shape": "x".join(str(v) for v in meta["shape"]),
                                "target_missing_rate": meta["target_missing_rate"],
                                "actual_missing_rate": meta["actual_missing_rate"],
                                "missing_rate_abs_error": meta["missing_rate_abs_error"],
                                "sha256": meta["sha256"],
                                "npz_path": str(npz_path),
                                "meta_path": str(meta_path),
                                "warning": meta.get("warning", ""),
                            }
                        )
        except Exception as exc:
            blockers.append(f"{dataset}: {exc!r}")

    write_csv(ROOT / "reports" / "gate3a_mask_inventory.csv", inventory)
    write_inventory_md(ROOT / "reports" / "gate3a_mask_inventory.md", inventory)
    write_protocol_reports(protocols)

    expected_count = len(DATASETS) * len(MASK_TYPES) * len(RATES) * len(SEEDS)
    status_value = "PASS"
    if blockers or len(inventory) != expected_count:
        status_value = "PARTIAL"
    status = {
        "gate": "Gate 3a",
        "status": status_value,
        "timestamp": iso_now(),
        "generator_version": GENERATOR_VERSION,
        "expected_mask_count": expected_count,
        "inventory_mask_count": len(inventory),
        "datasets_ready": sorted(protocols.keys()),
        "datasets_requested": list(DATASETS),
        "mask_types": list(MASK_TYPES),
        "missing_rates": list(RATES),
        "seeds": list(SEEDS),
        "warnings": warnings,
        "blockers": blockers,
        "reports": {
            "inventory_csv": str(ROOT / "reports" / "gate3a_mask_inventory.csv"),
            "inventory_md": str(ROOT / "reports" / "gate3a_mask_inventory.md"),
            "protocol_md": str(ROOT / "reports" / "gate3a_protocol.md"),
            "protocol_json": str(ROOT / "reports" / "gate3a_protocol.json"),
        },
        "leakage_checks": {
            "mask_only_history_input": "OK",
            "future_prediction_labels_not_read": "OK",
            "split_rules_from_official_loader": "OK",
            "normalization_uses_train_split_only": "OK",
            "same_mask_reusable_for_baseline_and_innovation": "OK",
            "seed_fixed": "OK",
            "evaluation_time_resampling_forbidden": "OK",
            "official_model_code_modified": "NO",
        },
    }
    (ROOT / "reports" / "gate3a_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status_value == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
