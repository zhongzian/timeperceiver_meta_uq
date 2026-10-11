#!/usr/bin/env python3
"""Shared eval-only model and fixed-mask data loading for Gate 5a."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate3b_smoke_missing_eval import build_args, import_official_modules, parse_simple_yaml
from gate5a_common import load, resolve
from models.probabilistic_timeperceiver import Model


def queue_run_map():
    queue = load(ROOT / "configs/queues/gate5a_distribution_ablation.yaml")
    state = load(ROOT / "runtime/gate5a_distribution_ablation_scheduler_state.json")
    mapping = {}
    for task in queue["tasks"]:
        key = task["task_id"]
        run = Path(state["tasks"][key]["run_dir"])
        mapping[(task["dataset"], task["variant"])] = (task, run if run.is_absolute() else ROOT / run)
    return mapping


def load_model_and_data(task, run, device):
    config = parse_simple_yaml(resolve(task["config"]))
    args = build_args(config, int(task["batch_size"]), "cuda")
    provider, _ = import_official_modules()
    test_data, _ = provider(args, "test")
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=int(task["batch_size"]), shuffle=False, num_workers=0
    )
    raw = torch.load(resolve(task["source_checkpoint"]), map_location="cpu")
    model = Model(args, task["variant"], True).float()
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if unexpected or not all(name.startswith("probabilistic_head.") for name in missing):
        raise RuntimeError(f"source checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.initialize_probabilistic_location()
    checkpoint = torch.load(run / "checkpoints/best_prob_head.pt", map_location="cpu")
    expected = {name for name in model.state_dict() if name.startswith("probabilistic_head.")}
    actual = set(checkpoint["prob_head_state_dict"])
    if actual != expected:
        raise RuntimeError(f"probabilistic head checkpoint mismatch: expected={expected}, actual={actual}")
    result = model.load_state_dict(checkpoint["prob_head_state_dict"], strict=False)
    if result.unexpected_keys or any(name in expected for name in result.missing_keys):
        raise RuntimeError(f"head reload failed: {result}")
    model.to(device).eval()
    test_mask = np.load(resolve(task["test_mask"]))["mask"]
    if tuple(test_mask.shape) != (len(test_data), args.seq_len, args.enc_in):
        raise RuntimeError(f"test mask shape mismatch: {test_mask.shape}")
    indices = (
        list(range(args.seq_len // args.patch_len)),
        list(range(args.seq_len // args.patch_len, (args.seq_len + args.pred_len) // args.patch_len)),
    )
    return model, args, test_loader, test_mask, indices


def prepare_batch(batch, mask, args, indices, device):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    mask = torch.from_numpy(mask).float().to(device)
    decoder = torch.cat(
        [batch_y[:, : args.label_len, :], torch.zeros_like(batch_y[:, -args.pred_len :, :])], dim=1
    )
    return batch_x, batch_y, batch_x_mark, batch_y_mark, mask, decoder, indices


def forward_batch(model, prepared, args):
    batch_x, batch_y, batch_x_mark, batch_y_mark, mask, decoder, indices = prepared
    output = model(batch_x * mask, batch_x_mark, decoder, batch_y_mark, indices, mask=mask)
    target = batch_y[:, -args.pred_len :, :]
    return output, target

