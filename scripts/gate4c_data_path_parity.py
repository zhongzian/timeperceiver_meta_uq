#!/usr/bin/env python
"""Compare the calibration-disabled Gate 4c path with corrected Gate 4b fixed-mask evaluation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gate3b_smoke_missing_eval import build_args, import_official_modules, load_checkpoint_strict, parse_simple_yaml
from models.probabilistic_timeperceiver import Model


def main() -> int:
    baseline = ROOT / "runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054"
    checkpoint = next((baseline / "checkpoints").rglob("checkpoint.pth"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = build_args(
        parse_simple_yaml(baseline / "resolved_config.yaml"), 32,
        "cuda" if device.type == "cuda" else "cpu",
    )
    provider, official_module = import_official_modules()
    _, loader = provider(args, "test")
    mask = np.load(ROOT / "artifacts/masks/etth1/random/rate_0.3/seed_2025.npz", allow_pickle=False)["mask"]
    corrected_gate4b = official_module.Model(args).float()
    load_checkpoint_strict(corrected_gate4b, checkpoint)
    corrected_gate4b.to(device).eval()
    gate4c = Model(args, "official", False).float()
    gate4c.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    gate4c.to(device).eval()
    patch_indices = (
        list(range(args.seq_len // args.patch_len)),
        list(range(args.seq_len // args.patch_len, (args.seq_len + args.pred_len) // args.patch_len)),
    )
    reference_sse = reference_sae = gate4c_sse = gate4c_sae = 0.0
    count = 0
    offset = 0
    max_diff = 0.0
    batches = 0
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if batch_index >= 8:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            mask_batch = torch.from_numpy(mask[offset:offset + len(batch_x)]).float().to(device)
            offset += len(batch_x)
            masked_x = batch_x * mask_batch
            decoder = torch.cat([
                batch_y[:, :args.label_len, :], torch.zeros_like(batch_y[:, -args.pred_len:, :])
            ], dim=1)
            reference = corrected_gate4b(masked_x, batch_x_mark, decoder, batch_y_mark, patch_indices)
            candidate = gate4c(masked_x, batch_x_mark, decoder, batch_y_mark, patch_indices, mask=mask_batch)
            target = batch_y[:, -args.pred_len:, :]
            max_diff = max(max_diff, float((candidate - reference).abs().max()))
            ref_error = reference - target
            candidate_error = candidate - target
            reference_sse += float((ref_error * ref_error).sum())
            reference_sae += float(ref_error.abs().sum())
            gate4c_sse += float((candidate_error * candidate_error).sum())
            gate4c_sae += float(candidate_error.abs().sum())
            count += candidate_error.numel()
            batches += 1
    reference_mse = reference_sse / count
    reference_mae = reference_sae / count
    candidate_mse = gate4c_sse / count
    candidate_mae = gate4c_sae / count
    differences = {
        "forecast_max_abs_diff": max_diff,
        "mse_diff": candidate_mse - reference_mse,
        "mae_diff": candidate_mae - reference_mae,
        "rmse_diff": math.sqrt(candidate_mse) - math.sqrt(reference_mse),
    }
    passed = all(value == 0.0 for value in differences.values()) and batches == 8
    output = {
        "stage": "Gate 4c data-path parity",
        "status": "PASS" if passed else "BLOCKED_GATE4C_DATA_PATH_PARITY_FAILURE",
        "evaluated_batches": batches,
        "evaluated_samples": offset,
        "device": str(device),
        "reference": "corrected Gate 4b fixed-mask official evaluator",
        "candidate": "Gate 4c wrapper with calibration disabled",
        "fixed_test_mask": "artifacts/masks/etth1/random/rate_0.3/seed_2025.npz",
        **differences,
    }
    (ROOT / "results/gate4c_data_path_parity.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "reports/gate4c_data_path_parity.md").write_text(
        "# Gate 4c Data-Path Parity\n\n"
        f"Status: **{output['status']}**\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in differences.items()) + "\n"
    )
    print(json.dumps(output, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
