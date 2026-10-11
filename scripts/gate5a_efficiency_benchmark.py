#!/usr/bin/env python3
"""Controlled same-device Gate 5a inference and metric latency benchmark."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_common import DATASETS, DISPLAY_NAMES, HEADS, dump, load, write_csv, write_json_markdown
from gate5a_distribution_metrics import central_intervals, empirical_crps, nll_elements
from gate5a_eval_common import forward_batch, load_model_and_data, prepare_batch, queue_run_map


WARMUP = 20
MEASURED = 100


def timed(operation):
    for _ in range(WARMUP):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(MEASURED):
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": statistics.median(values),
        "p90_ms": float(np.quantile(values, 0.9)),
    }


def training_metadata(run):
    metrics = load(run / "metrics.json")
    env = (run / "env.txt").read_text() if (run / "env.txt").is_file() else ""
    gpu_line = next((line for line in env.splitlines() if line.startswith("gpu_before:")), "")
    return {
        "training_wall_seconds": metrics.get("elapsed_seconds"),
        "epochs_completed": metrics.get("epochs_completed"),
        "mean_epoch_seconds": (
            metrics.get("elapsed_seconds") / metrics.get("epochs_completed")
            if metrics.get("elapsed_seconds") and metrics.get("epochs_completed") else None
        ),
        "formal_peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
        "formal_training_gpu_snapshot": gpu_line.removeprefix("gpu_before: "),
        "trainable_parameter_count": metrics.get("trainable_parameter_count"),
        "total_parameter_count": metrics.get("total_parameter_count"),
        "parameter_overhead_ratio": metrics.get("parameter_overhead_ratio"),
    }


def main():
    decision = load(ROOT / "results/gate5a_formal_decision.json")
    if decision.get("GATE5A_IMPLEMENTATION_PASS") is not True:
        raise RuntimeError("Gate 5a implementation is incomplete")
    if not torch.cuda.is_available():
        raise RuntimeError("WAITING_RESOURCE: CUDA required for controlled benchmark")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_uuid = getattr(
        torch.cuda.get_device_properties(device), "uuid", os.environ.get("CUDA_VISIBLE_DEVICES", "logical_cuda_0")
    )
    mapping = queue_run_map()
    rows = []
    for dataset in DATASETS:
        expected_batch = None
        for variant in HEADS:
            task, run = mapping[(dataset, variant)]
            model, args, loader, test_mask, indices = load_model_and_data(task, run, device)
            batch = next(iter(loader))
            batch_size = len(batch[0])
            if expected_batch is None:
                expected_batch = batch_size
            if batch_size != expected_batch or batch_size != int(task["batch_size"]):
                raise RuntimeError(f"benchmark batch mismatch for {dataset}/{variant}")
            prepared = prepare_batch(batch, test_mask[:batch_size], args, indices, device)

            def forward_only():
                with torch.no_grad():
                    return forward_batch(model, prepared, args)[0]

            def forward_nll():
                with torch.no_grad():
                    output, target = forward_batch(model, prepared, args)
                    return nll_elements(variant, output["loc"], output["scale"], target, output["df"]).mean()

            def forward_intervals():
                with torch.no_grad():
                    output, _ = forward_batch(model, prepared, args)
                    values = central_intervals(variant, output["loc"], output["scale"], output["df"])
                    return values[0.9], values[0.95]

            def forward_crps():
                with torch.no_grad():
                    output, target = forward_batch(model, prepared, args)
                    return empirical_crps(
                        variant, output["loc"], output["scale"], target, output["df"], 64, 52025, 65536
                    )

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            forward_stats = timed(forward_only)
            nll_stats = timed(forward_nll)
            interval_stats = timed(forward_intervals)
            crps_stats = timed(forward_crps)
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1048576
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1048576
            row = {
                "dataset": dataset, "head": DISPLAY_NAMES[variant], "variant": variant,
                "batch_size": batch_size, "precision": "float32", "warmup_iterations": WARMUP,
                "measured_iterations": MEASURED, "gpu_name": gpu_name, "gpu_uuid": str(gpu_uuid),
                "forward_median_ms": forward_stats["median_ms"], "forward_p90_ms": forward_stats["p90_ms"],
                "forward_throughput_windows_per_second": batch_size * 1000 / forward_stats["median_ms"],
                "forward_nll_median_ms": nll_stats["median_ms"], "forward_nll_p90_ms": nll_stats["p90_ms"],
                "forward_interval_median_ms": interval_stats["median_ms"], "forward_interval_p90_ms": interval_stats["p90_ms"],
                "forward_crps64_median_ms": crps_stats["median_ms"], "forward_crps64_p90_ms": crps_stats["p90_ms"],
                "benchmark_peak_allocated_mib": peak_allocated, "benchmark_peak_reserved_mib": peak_reserved,
                "run_dir": str(run), **training_metadata(run),
            }
            rows.append(row)
            del model, prepared, batch
            torch.cuda.empty_cache()
    by_key = {(row["dataset"], row["variant"]): row for row in rows}
    parameter_ratios = []
    runtime_ratios = []
    for dataset in DATASETS:
        gaussian = by_key[(dataset, "gaussian_head")]
        student = by_key[(dataset, "student_t_head")]
        parameter_ratios.append(
            (student["trainable_parameter_count"] - gaussian["trainable_parameter_count"])
            / gaussian["total_parameter_count"]
        )
        runtime_ratios.append(student["forward_median_ms"] / gaussian["forward_median_ms"])
    parameter_cost_minimal = max(parameter_ratios) <= 0.001
    runtime_cost_minimal = max(runtime_ratios) <= 1.10
    payload = {
        "status": "PASS", "gpu_name": gpu_name, "gpu_uuid": str(gpu_uuid),
        "warmup_iterations": WARMUP, "measured_iterations": MEASURED,
        "same_device_within_benchmark": True, "formal_training_hardware_strictly_comparable": False,
        "STUDENT_T_PARAMETER_COST_MINIMAL": parameter_cost_minimal,
        "STUDENT_T_RUNTIME_COST_MINIMAL": runtime_cost_minimal,
        "parameter_cost_rule": "maximum extra trainable Student-t parameters / total Gaussian model parameters <= 0.001",
        "runtime_cost_rule": "maximum per-dataset learnable-t / Gaussian forward median ratio <= 1.10",
        "rows": rows,
    }
    dump(ROOT / "results/gate5a_efficiency.json", payload)
    write_csv(ROOT / "reports/gate5a_efficiency.csv", rows)
    write_json_markdown(
        ROOT / "reports/gate5a_efficiency.md", "Gate 5a Efficiency", payload,
        "Controlled latency rows share one GPU, batch, precision, warmup, and iteration count. Formal run wall times are hardware-tagged and not treated as strict speed comparisons.",
    )
    mechanism_path = ROOT / "results/gate5a_mechanism_audit.json"
    if mechanism_path.is_file():
        mechanism = load(mechanism_path)
        mechanism["STUDENT_T_PARAMETER_COST_MINIMAL"] = parameter_cost_minimal
        mechanism["STUDENT_T_RUNTIME_COST_MINIMAL"] = runtime_cost_minimal
        dump(mechanism_path, mechanism)
        write_json_markdown(ROOT / "reports/gate5a_mechanism_audit.md", "Gate 5a Mechanism Audit", mechanism)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
