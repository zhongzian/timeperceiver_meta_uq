#!/usr/bin/env python3
"""Eval-only PatchTST tail-NLL decomposition for Gate 5b."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate3b_smoke_missing_eval import build_args, import_official_modules, parse_simple_yaml  # noqa: E402
from gate5a_distribution_metrics import nll_elements  # noqa: E402
from gate5b_common import (  # noqa: E402
    DATASETS,
    dump,
    load,
    load_resolved,
    resolve,
    sha256_file,
    write_csv,
    write_report,
)
from models.patchtst_backbone import PatchTSTBackbone  # noqa: E402
from models.probabilistic_patchtst import ProbabilisticPatchTST  # noqa: E402


CACHE = ROOT / "artifacts/gate5b_tail_nll_gain"
FIGURES = ROOT / "paper/figures/gate5b"
BIN_LABELS = ("0-50", "50-90", "90-95", "95-99", "99-100")
MAX_QUANTILE_SAMPLE = 2_000_000


def run_map():
    state = load(ROOT / "runtime/gate5b_patchtst_probability_scheduler_state.json")
    if state.get("status") != "COMPLETE":
        raise RuntimeError("Gate 5b probability queue is not COMPLETE")
    mapping = {}
    for task_state in state["tasks"].values():
        run = resolve(task_state["run_dir"])
        task = load_resolved(run / "resolved_config.yaml")
        mapping[(task["dataset"], task["distribution"])] = (task, run)
    if len(mapping) != 6:
        raise RuntimeError(f"expected six formal runs, found {len(mapping)}")
    return mapping


def load_model(task, run, device):
    backbone_object = torch.load(resolve(task["source_backbone_checkpoint"]), map_location="cpu")
    backbone = PatchTSTBackbone(task).float()
    backbone.load_state_dict(backbone_object["model_state_dict"], strict=True)
    model = ProbabilisticPatchTST(
        backbone,
        task["distribution"],
        float(task["initial_scale"]),
        float(task["scale_floor"]),
        float(task.get("fixed_t_df") or 8.0),
    ).float()
    head_object = torch.load(run / "checkpoints/best_probability_head.pt", map_location="cpu")
    result = model.load_state_dict(head_object["head_state_dict"], strict=False)
    expected = {name for name in model.state_dict() if name.startswith(("loc_projection.", "scale_projection."))}
    if set(head_object["head_state_dict"]) != expected or result.unexpected_keys:
        raise RuntimeError(f"strict head reload failed for {run.name}")
    return model.to(device).eval()


def data_bundle(task):
    provider, _ = import_official_modules()
    baseline = parse_simple_yaml(resolve(task["config"]))
    args = build_args(baseline, int(task["batch_size"]), "cuda")
    args.seq_len = int(task["seq_len"])
    args.pred_len = int(task["pred_len"])
    args.batch_size = int(task["batch_size"])
    test_data, _ = provider(args, "test")
    loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
    )
    mask_path = resolve(task["test_mask"])
    if sha256_file(mask_path) != task["test_mask_sha256"]:
        raise RuntimeError("test mask hash mismatch")
    mask = np.load(mask_path)["mask"]
    expected = (len(test_data), int(task["seq_len"]), int(task["enc_in"]))
    if tuple(mask.shape) != expected:
        raise RuntimeError(f"test mask shape {mask.shape} != {expected}")
    return args, loader, mask


def prepared(batch, mask, offset, args, device):
    batch_x, batch_y = batch[0], batch[1]
    size = len(batch_x)
    batch_mask = torch.from_numpy(mask[offset : offset + size]).float().to(device)
    batch_x = batch_x.float().to(device)
    target = batch_y[:, -args.pred_len :, :].float().to(device)
    return batch_x * batch_mask, target


def quantile_boundaries(model, loader, mask, args, device):
    total = len(mask) * args.pred_len * args.enc_in
    stride = max(1, math.ceil(total / MAX_QUANTILE_SAMPLE))
    offset = 0
    flattened_offset = 0
    samples = []
    with torch.no_grad():
        for batch in loader:
            masked_x, target = prepared(batch, mask, offset, args, device)
            offset += len(batch[0])
            output = model(masked_x)
            values = ((target - output["loc"]).abs() / output["scale"]).reshape(-1).cpu().numpy()
            first = (-flattened_offset) % stride
            samples.append(values[first::stride])
            flattened_offset += len(values)
    sample = np.concatenate(samples).astype(np.float64, copy=False)
    boundaries = np.quantile(sample, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return boundaries, len(sample), stride, total


def evaluate_dataset(dataset, mapping, device):
    gaussian_task, gaussian_run = mapping[(dataset, "gaussian")]
    fixed_task, fixed_run = mapping[(dataset, "fixed_t8")]
    cache_path = CACHE / f"{dataset}.json"
    identity = {
        "gaussian_experiment_id": gaussian_run.name,
        "fixed_t8_experiment_id": fixed_run.name,
        "gaussian_head_sha256": sha256_file(gaussian_run / "checkpoints/best_probability_head.pt"),
        "fixed_t8_head_sha256": sha256_file(fixed_run / "checkpoints/best_probability_head.pt"),
    }
    if cache_path.is_file():
        cached = load(cache_path)
        if cached.get("status") == "PASS" and all(cached.get(key) == value for key, value in identity.items()):
            return cached["rows"], cached

    args, loader, mask = data_bundle(gaussian_task)
    gaussian = load_model(gaussian_task, gaussian_run, device)
    boundaries, sample_count, stride, total = quantile_boundaries(gaussian, loader, mask, args, device)
    fixed = load_model(fixed_task, fixed_run, device)

    counts = np.zeros(5, dtype=np.int64)
    gaussian_sums = np.zeros(5, dtype=np.float64)
    fixed_sums = np.zeros(5, dtype=np.float64)
    offset = 0
    internal = torch.tensor(boundaries[1:-1], dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in loader:
            masked_x, target = prepared(batch, mask, offset, args, device)
            offset += len(batch[0])
            gaussian_output = gaussian(masked_x)
            fixed_output = fixed(masked_x)
            residual = ((target - gaussian_output["loc"]).abs() / gaussian_output["scale"]).reshape(-1)
            bins = torch.bucketize(residual, internal, right=False)
            gaussian_nll = nll_elements(
                "gaussian_head", gaussian_output["loc"], gaussian_output["scale"], target, None,
            ).reshape(-1)
            fixed_nll = nll_elements(
                "student_t_fixed_df8_head", fixed_output["loc"], fixed_output["scale"],
                target, fixed_output["df"],
            ).reshape(-1)
            for index in range(5):
                selected = bins == index
                count = int(selected.sum())
                if count:
                    counts[index] += count
                    gaussian_sums[index] += float(gaussian_nll[selected].double().sum())
                    fixed_sums[index] += float(fixed_nll[selected].double().sum())

    rows = []
    for index, label in enumerate(BIN_LABELS):
        gaussian_mean = gaussian_sums[index] / counts[index]
        fixed_mean = fixed_sums[index] / counts[index]
        rows.append({
            "dataset": dataset,
            "gaussian_residual_percentile_bin": label,
            "lower_gaussian_standardized_residual": float(boundaries[index]),
            "upper_gaussian_standardized_residual": float(boundaries[index + 1]),
            "element_count": int(counts[index]),
            "gaussian_mean_nll": gaussian_mean,
            "fixed_t8_mean_nll": fixed_mean,
            "fixed_t8_minus_gaussian_nll": fixed_mean - gaussian_mean,
            "descriptive_only": True,
            "formal_gate_evidence": False,
        })
    weighted_gaussian = float(gaussian_sums.sum() / counts.sum())
    weighted_fixed = float(fixed_sums.sum() / counts.sum())
    gaussian_metrics = load(gaussian_run / "metrics.json")
    fixed_metrics = load(fixed_run / "metrics.json")
    consistency = {
        "element_count_complete": int(counts.sum()) == total,
        "gaussian_nll_matches_formal": math.isclose(weighted_gaussian, gaussian_metrics["test_mean_nll"], rel_tol=0, abs_tol=2e-6),
        "fixed_t8_nll_matches_formal": math.isclose(weighted_fixed, fixed_metrics["test_mean_nll"], rel_tol=0, abs_tol=2e-6),
    }
    payload = {
        "status": "PASS" if all(consistency.values()) else "FAILED",
        "dataset": dataset,
        **identity,
        "quantile_method": "deterministic systematic sample; full-data bin aggregation",
        "quantile_sample_count": sample_count,
        "quantile_sample_stride": stride,
        "total_element_count": total,
        "consistency": consistency,
        "rows": rows,
        "evaluation_only": True,
        "training": False,
        "backward": False,
        "descriptive_only": True,
        "formal_gate_evidence": False,
    }
    dump(cache_path, payload)
    del gaussian, fixed
    torch.cuda.empty_cache()
    if payload["status"] != "PASS":
        raise RuntimeError(f"tail diagnostic consistency failed for {dataset}: {consistency}")
    return rows, payload


def markdown_table(rows):
    fields = (
        "dataset", "gaussian_residual_percentile_bin", "element_count",
        "gaussian_mean_nll", "fixed_t8_mean_nll", "fixed_t8_minus_gaussian_nll",
    )
    text = "# Gate 5b PatchTST Tail NLL Gain\n\n"
    text += "This evaluation-only diagnostic is descriptive and is not formal gate evidence.\n\n"
    text += "| " + " | ".join(fields) + " |\n"
    text += "| " + " | ".join("---" for _ in fields) + " |\n"
    for row in rows:
        text += "| " + " | ".join(str(row[field]) for field in fields) + " |\n"
    return text


def make_figure(rows):
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_csv(FIGURES / "fig_gate5b_patchtst_tail_nll_gain_data.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        local = [row for row in rows if row["dataset"] == dataset]
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.plot(
            BIN_LABELS,
            [row["fixed_t8_minus_gaussian_nll"] for row in local],
            marker="o",
            color="#b34d5d",
        )
        axis.set_title(dataset)
        axis.set_xlabel("Gaussian residual percentile")
        axis.set_ylabel("Fixed-t8 - Gaussian NLL")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("PatchTST tail NLL decomposition (descriptive)")
    fig.savefig(FIGURES / "fig_gate5b_patchtst_tail_nll_gain.png", dpi=320, facecolor="white")
    fig.savefig(FIGURES / "fig_gate5b_patchtst_tail_nll_gain.pdf", facecolor="white")
    plt.close(fig)


def main():
    decision = load(ROOT / "results/gate5b_formal_decision.json")
    if decision.get("GATE5B_IMPLEMENTATION_PASS") is not True:
        raise RuntimeError("Gate 5b implementation must pass before diagnostics")
    if not torch.cuda.is_available():
        raise RuntimeError("WAITING_RESOURCE: CUDA is required for eval-only diagnostics")
    CACHE.mkdir(parents=True, exist_ok=True)
    mapping = run_map()
    device = torch.device("cuda:0")
    all_rows = []
    datasets = {}
    for dataset in DATASETS:
        rows, payload = evaluate_dataset(dataset, mapping, device)
        all_rows.extend(rows)
        datasets[dataset] = {
            "status": payload["status"],
            "quantile_sample_count": payload["quantile_sample_count"],
            "total_element_count": payload["total_element_count"],
            "consistency": payload["consistency"],
        }
        print(json.dumps({"dataset": dataset, **datasets[dataset]}), flush=True)
    write_csv(ROOT / "reports/gate5b_patchtst_tail_nll_gain.csv", all_rows)
    (ROOT / "reports/gate5b_patchtst_tail_nll_gain.md").write_text(markdown_table(all_rows))
    make_figure(all_rows)
    result = {
        "status": "PASS" if len(all_rows) == 15 and all(value["status"] == "PASS" for value in datasets.values()) else "FAILED",
        "rows": len(all_rows),
        "datasets": datasets,
        "evaluation_only": True,
        "training": False,
        "backward": False,
        "descriptive_only": True,
        "formal_gate_evidence": False,
        "statistical_significance_claim": False,
    }
    dump(ROOT / "results/gate5b_tail_diagnostic.json", result)
    write_report(ROOT / "reports/gate5b_tail_diagnostic.md", "Gate 5b Tail Diagnostic", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
