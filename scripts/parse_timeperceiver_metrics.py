#!/usr/bin/env python
"""Parse official TimePerceiver outputs from one run directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


EPOCH_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*Steps:\s*(?P<steps>\d+)\s*\|\s*"
    r"Train Loss:\s*(?P<train>[0-9eE+\-.]+)\s*"
    r"Vali Loss:\s*(?P<vali>[0-9eE+\-.]+)\s*"
    r"Test Loss:\s*(?P<test>[0-9eE+\-.]+)"
)
METRIC_RE = re.compile(r"mse:\s*(?P<mse>[0-9eE+\-.]+),\s*mae:\s*(?P<mae>[0-9eE+\-.]+)")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def parse_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    train_log = run_dir / "train.log"
    result: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "mse": None,
        "mae": None,
        "rmse": None,
        "mape": None,
        "mspe": None,
        "best_epoch": None,
        "final_epoch": None,
        "checkpoint_path": None,
        "metrics_source": None,
        "parse_warnings": [],
    }

    metrics_files = sorted((run_dir / "results").glob("*/metrics.npy"))
    if metrics_files:
        metrics_path = metrics_files[-1]
        try:
            arr = np.load(metrics_path, allow_pickle=False).tolist()
            if len(arr) >= 5:
                result.update(
                    {
                        "mae": _float_or_none(arr[0]),
                        "mse": _float_or_none(arr[1]),
                        "rmse": _float_or_none(arr[2]),
                        "mape": _float_or_none(arr[3]),
                        "mspe": _float_or_none(arr[4]),
                        "metrics_source": str(metrics_path),
                    }
                )
        except Exception as exc:  # pragma: no cover - diagnostic path
            result["parse_warnings"].append(f"failed to read metrics.npy: {exc}")

    if train_log.exists():
        text = train_log.read_text(errors="replace")
        epochs = []
        for match in EPOCH_RE.finditer(text):
            epochs.append(
                {
                    "epoch": int(match.group("epoch")),
                    "steps": int(match.group("steps")),
                    "train_loss": _float_or_none(match.group("train")),
                    "vali_loss": _float_or_none(match.group("vali")),
                    "test_loss": _float_or_none(match.group("test")),
                }
            )
        if epochs:
            result["final_epoch"] = epochs[-1]["epoch"]
            best = min(epochs, key=lambda row: row["vali_loss"] if row["vali_loss"] is not None else float("inf"))
            result["best_epoch"] = best["epoch"]
            result["best_vali_loss"] = best["vali_loss"]
            result["final_epoch_record"] = epochs[-1]

        if result["mse"] is None or result["mae"] is None:
            matches = list(METRIC_RE.finditer(text))
            if matches:
                last = matches[-1]
                result["mse"] = _float_or_none(last.group("mse"))
                result["mae"] = _float_or_none(last.group("mae"))
                result["metrics_source"] = str(train_log)

    checkpoint_files = sorted((run_dir / "checkpoints").glob("*/checkpoint.pth"))
    if checkpoint_files:
        result["checkpoint_path"] = str(checkpoint_files[-1])
    elif (run_dir / "checkpoints").exists():
        result["parse_warnings"].append("checkpoint directory exists but checkpoint.pth was not found")
    else:
        result["parse_warnings"].append("checkpoint directory was not found")

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = parse_run(args.run_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
