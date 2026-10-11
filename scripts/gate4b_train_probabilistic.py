#!/usr/bin/env python
"""Shared Gate 4b probabilistic training path for smoke and formal tasks."""
import argparse
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gate3a_generate_missingness_assets import make_mask
from gate3b_smoke_missing_eval import (
    build_args,
    import_official_modules,
    load_checkpoint_strict,
    parse_simple_yaml,
    sha256_file,
)
from gate5a_distribution_metrics import central_intervals, empirical_crps, interval_metrics, nll_elements
from models.probabilistic_timeperceiver import Model, freeze_backbone

BASELINE_RUN = ROOT / "runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054"
DEFAULT_CHECKPOINT = next((BASELINE_RUN / "checkpoints").rglob("checkpoint.pth"))
LEGACY_ETTH1_MASK_RULE = "sha256(base_seed|epoch|sample_index|missing_type|target_rate)"
LEGACY_DATASET_MASK_RULE = "sha256(base_seed|epoch|sample_index|dataset|missing_type|target_rate)"
MULTISEED_MASK_RULE = (
    "sha256(dataset|experiment_seed|base_seed|epoch|sample_index|missing_type|target_rate)"
)
SUPPORTED_VARIANTS = {
    "gaussian_head",
    "laplace_head",
    "student_t_fixed_df8_head",
    "student_t_head",
}
STUDENT_VARIANTS = {"student_t_fixed_df8_head", "student_t_head"}


class Indexed(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return (*self.base[index], index)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def cmdout(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return repr(exc)


def stable_mask_seed(base_seed, experiment_seed, epoch, sample_index, dataset, missing_type, target_rate,
                     seed_rule=MULTISEED_MASK_RULE):
    if seed_rule == LEGACY_ETTH1_MASK_RULE:
        key_text = f"{base_seed}|{epoch}|{sample_index}|{missing_type}|{target_rate:.6f}"
    elif seed_rule == LEGACY_DATASET_MASK_RULE:
        key_text = f"{base_seed}|{epoch}|{sample_index}|{dataset}|{missing_type}|{target_rate:.6f}"
    elif seed_rule == MULTISEED_MASK_RULE:
        key_text = (
            f"{dataset}|{experiment_seed}|{base_seed}|{epoch}|{sample_index}|"
            f"{missing_type}|{target_rate:.6f}"
        )
    else:
        raise ValueError(f"unsupported train mask seed rule: {seed_rule}")
    key = key_text.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def sample_masks(indices, epoch, shape, dataset, missing_type, target_rate, base_seed=42025,
                 experiment_seed=2025, seed_rule=LEGACY_DATASET_MASK_RULE):
    masks = []
    one_shape = (1, int(shape[1]), int(shape[2]))
    for sample_index in indices.tolist():
        seed = stable_mask_seed(
            base_seed, experiment_seed, epoch, int(sample_index), dataset, missing_type, target_rate, seed_rule
        )
        masks.append(make_mask(missing_type, seed, one_shape, target_rate)[0])
    return torch.from_numpy(np.stack(masks)).float()


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def head_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if name.startswith("probabilistic_head.")}


def default_task(variant, smoke):
    return {
        "stage": "gate4b_protocol_smoke" if smoke else "gate4b",
        "variant": variant,
        "dataset": "etth1",
        "seq_len": 384,
        "pred_len": 96,
        "seed": 2025,
        "missing_type": "random",
        "missing_rate": 0.3,
        "train_mask_base_seed": 42025,
        "freeze_backbone": True,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "max_epochs": 2 if smoke else 20,
        "patience": 3,
        "early_stopping_metric": "validation_nll",
        "test_used_for_model_selection": False,
        "source_checkpoint": str(DEFAULT_CHECKPOINT.relative_to(ROOT)),
        "validation_mask": "artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz",
        "test_mask": "artifacts/masks/etth1/random/rate_0.3/seed_2025.npz",
        "max_train_batches": 32 if smoke else None,
        "max_val_batches": 8 if smoke else None,
        "max_test_batches": 8 if smoke else None,
        "smoke_only": smoke,
        "training_protocol_hash": os.environ.get("GATE4B_TRAINING_PROTOCOL_HASH"),
        "metric_protocol_hash": os.environ.get("GATE4B_METRIC_PROTOCOL_HASH"),
    }


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def classify_error(error):
    text = error.lower()
    if "out of memory" in text:
        return "OOM"
    if "gpu_required" in text:
        return "WAITING_RESOURCE"
    if "no such file" in text or "missing path" in text:
        return "DATA_ERROR"
    if "config" in text or "expected" in text:
        return "CONFIG_ERROR"
    if "nan" in text or "non-finite" in text:
        return "NAN_OR_DIVERGENCE"
    return "CODE_ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(SUPPORTED_VARIANTS))
    parser.add_argument("--task-file")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    cli = parser.parse_args()
    if cli.task_file:
        task = json.loads(Path(cli.task_file).read_text())
    else:
        if not cli.variant:
            raise SystemExit("--variant or --task-file is required")
        task = default_task(cli.variant, cli.smoke)
    variant = task["variant"]
    if variant not in SUPPORTED_VARIANTS:
        raise RuntimeError(f"unsupported probabilistic variant: {variant}")
    dataset = str(task.get("dataset", "etth1")).lower()
    seed = int(task.get("seed", 2025))
    experiment_seed = int(task.get("experiment_seed", seed))
    if experiment_seed != seed:
        raise RuntimeError(f"experiment_seed mismatch: seed={seed}, experiment_seed={experiment_seed}")
    train_mask_base_seed = int(task.get("train_mask_base_seed", 42025))
    train_mask_seed_rule = str(task.get("train_mask_seed_rule", LEGACY_DATASET_MASK_RULE))
    if str(task.get("stage", "")).startswith("gate4b_multiseed"):
        expected_base_seed = 40000 + experiment_seed
        if train_mask_base_seed != expected_base_seed:
            raise RuntimeError(
                f"train_mask_base_seed mismatch: expected {expected_base_seed}, got {train_mask_base_seed}"
            )
    smoke_only = bool(task.get("smoke_only", cli.smoke))
    missing_type = str(task["missing_type"])
    target_rate = float(task["missing_rate"])
    run = Path(cli.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    (run / "checkpoints").mkdir()
    started = time.time()
    (run / "command.sh").write_text("#!/usr/bin/env bash\n" + sys.executable + " " + " ".join(sys.argv) + "\n")

    baseline_config = resolve_path(task.get("config", BASELINE_RUN / "resolved_config.yaml"))
    cfg = parse_simple_yaml(baseline_config)
    if str(cfg.get("dataset", "")).lower() != dataset:
        raise RuntimeError(f"config dataset mismatch: expected {dataset}, got {cfg.get('dataset')}")
    args = build_args(cfg, int(cfg["batch_size"]), "cuda")
    max_epochs = int(task["max_epochs"])
    max_train = task.get("max_train_batches")
    max_val = task.get("max_val_batches")
    max_test = task.get("max_test_batches")
    checkpoint = resolve_path(task["source_checkpoint"])
    validation_mask = resolve_path(task["validation_mask"])
    test_mask = resolve_path(task["test_mask"])
    training_hash = task.get("training_protocol_hash") or os.environ.get("GATE4B_TRAINING_PROTOCOL_HASH")
    metric_hash = task.get("metric_protocol_hash") or os.environ.get("GATE4B_METRIC_PROTOCOL_HASH")
    resolved = {
        **cfg,
        **task,
        "stage": task.get("stage", "gate4b"),
        "variant": variant,
        "smoke_only": smoke_only,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "train_epochs": max_epochs,
        "max_epochs": max_epochs,
        "training_protocol_hash": training_hash,
        "metric_protocol_hash": metric_hash,
        "normalization_reference": "official_train_split_statistics",
        "baseline_config": str(baseline_config),
        "input_mask_application": "batch_x * mask before model forward",
    }
    (run / "resolved_config.yaml").write_text(
        "\n".join(f"{key}: {json.dumps(value)}" for key, value in sorted(resolved.items())) + "\n"
    )
    (run / "train.log").write_text(
        f"{task.get('stage', 'gate4b')} frozen-backbone probabilistic training.\n"
    )
    (run / "env.txt").write_text(
        f"hostname: {socket.gethostname()}\npython: {sys.version.split()[0]}\ntorch: {torch.__version__}\n"
        f"cuda_runtime: {torch.version.cuda}\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        f"gpu_before: {cmdout(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free', '--format=csv,noheader'])}\n"
    )
    (run / "git_state.txt").write_text(
        cmdout(["git", "-C", str(ROOT / "upstream/TimePerceiver"), "rev-parse", "HEAD"]) + "\n" +
        cmdout(["git", "-C", str(ROOT / "upstream/TimePerceiver"), "status", "--short"]) + "\n"
    )
    (run / "code.diff").write_text(
        cmdout(["git", "-C", str(ROOT / "upstream/TimePerceiver"), "diff", "--binary"]) + "\n"
    )
    status = "FAILED"
    metrics = {}
    error = ""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_REQUIRED")
        for path in (checkpoint, validation_mask, test_mask):
            if not path.is_file():
                raise FileNotFoundError(path)
        random.seed(experiment_seed)
        np.random.seed(experiment_seed)
        torch.manual_seed(experiment_seed)
        torch.cuda.manual_seed_all(experiment_seed)
        torch.cuda.set_device(0)
        torch.cuda.init()
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        provider, official_module = import_official_modules()
        train_data, _ = provider(args, "train")
        val_data, _ = provider(args, "val")
        test_data, _ = provider(args, "test")
        generator = torch.Generator().manual_seed(experiment_seed)
        train_loader = torch.utils.data.DataLoader(
            Indexed(train_data), batch_size=args.batch_size, shuffle=True, generator=generator,
            worker_init_fn=seed_worker, num_workers=0, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        validation_array = np.load(validation_mask)["mask"]
        if tuple(validation_array.shape) != (len(val_data), args.seq_len, args.enc_in):
            raise RuntimeError(f"validation mask shape mismatch: {validation_array.shape}")

        official = official_module.Model(args).float()
        load_checkpoint_strict(official, checkpoint)
        del official
        raw = torch.load(checkpoint, map_location="cpu")
        model = Model(args, variant, True).float()
        missing, unexpected = model.load_state_dict(raw, strict=False)
        if unexpected or not all(name.startswith("probabilistic_head.") for name in missing):
            raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
        model.initialize_probabilistic_location()
        trainable_names = freeze_backbone(model)
        model.to(device)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        optimizer_backbone = sum(1 for group in optimizer.param_groups for parameter in group["params"]
                                 if not parameter.requires_grad)
        initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                   if parameter.requires_grad}
        indices = (
            list(range(args.seq_len // args.patch_len)),
            list(range(args.seq_len // args.patch_len, (args.seq_len + args.pred_len) // args.patch_len)),
        )

        def forward(batch_x, batch_y, batch_x_mark, batch_y_mark, mask_batch, current_model):
            decoder = torch.cat([
                batch_y[:, :args.label_len, :], torch.zeros_like(batch_y[:, -args.pred_len:, :])
            ], dim=1)
            masked_x = batch_x * mask_batch
            output = current_model(masked_x, batch_x_mark, decoder, batch_y_mark, indices, mask=mask_batch)
            return output, batch_y[:, -args.pred_len:, :], masked_x

        best = float("inf")
        best_epoch = 0
        bad_epochs = 0
        early_stopped = False
        epochs_completed = 0
        train_batches = 0
        finite_train = True
        finite_grad = True
        loc_grad = False
        scale_grad = False
        expects_df_gradient = variant == "student_t_head"
        df_grad = False
        backbone_grads = 0
        batch_y_unchanged = True
        deterministic_masks = True
        input_mask_applied = False
        train_missing = 0
        train_total = 0
        first_batch_window_indices_sha256 = None
        for epoch in range(max_epochs):
            epochs_completed = epoch + 1
            model.train()
            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark, window_index) in enumerate(train_loader):
                if max_train is not None and batch_index >= int(max_train):
                    break
                mask_cpu = sample_masks(
                    window_index, epoch, batch_x.shape, dataset, missing_type, target_rate,
                    train_mask_base_seed, experiment_seed, train_mask_seed_rule,
                )
                if train_batches == 0:
                    repeated = sample_masks(
                        window_index, epoch, batch_x.shape, dataset, missing_type, target_rate,
                        train_mask_base_seed, experiment_seed, train_mask_seed_rule,
                    )
                    deterministic_masks = bool(torch.equal(mask_cpu, repeated))
                    first_batch_window_indices_sha256 = hashlib.sha256(
                        np.asarray(window_index.tolist(), dtype=np.int64).tobytes()
                    ).hexdigest()
                train_missing += int((mask_cpu == 0).sum())
                train_total += mask_cpu.numel()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                mask_batch = mask_cpu.to(device)
                before_y = batch_y.clone()
                optimizer.zero_grad()
                output, target, masked_x = forward(batch_x, batch_y, batch_x_mark, batch_y_mark, mask_batch, model)
                input_mask_applied |= bool(torch.any(mask_batch == 0) and torch.any(masked_x != batch_x))
                loss = nll_elements(variant, output["loc"], output["scale"], target, output["df"]).mean()
                finite_train &= bool(torch.isfinite(loss))
                loss.backward()
                batch_y_unchanged &= bool(torch.equal(before_y, batch_y))
                train_batches += 1
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        finite_grad &= bool(torch.isfinite(parameter.grad).all())
                        nonzero = bool(torch.count_nonzero(parameter.grad))
                        loc_grad |= nonzero and "loc_projection" in name
                        scale_grad |= nonzero and "scale_projection" in name
                        df_grad |= nonzero and "raw_df" in name
                    elif not parameter.requires_grad and parameter.grad is not None:
                        backbone_grads += 1
                clip_grad_norm_(params, 1.0)
                optimizer.step()
            model.eval()
            validation_losses = []
            offset = 0
            with torch.no_grad():
                for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                    if max_val is not None and batch_index >= int(max_val):
                        break
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    mask_batch = torch.from_numpy(validation_array[offset:offset + len(batch_x)]).float().to(device)
                    offset += len(batch_x)
                    output, target, _ = forward(batch_x, batch_y, batch_x_mark, batch_y_mark, mask_batch, model)
                    validation_losses.append(float(nll_elements(
                        variant, output["loc"], output["scale"], target, output["df"]
                    ).mean()))
            validation_nll = float(np.mean(validation_losses))
            if validation_nll < best:
                best = validation_nll
                best_epoch = epoch + 1
                bad_epochs = 0
                torch.save({
                    "prob_head_state_dict": head_state(model), "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": best_epoch, "best_validation_nll": best, "variant": variant,
                    "source_checkpoint_sha256": sha256_file(checkpoint),
                    "training_protocol_hash": training_hash, "metric_protocol_hash": metric_hash,
                }, run / "checkpoints/best_prob_head.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= int(task.get("patience", 3)):
                    early_stopped = True
                    break
        torch.save({
            "prob_head_state_dict": head_state(model), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs_completed, "variant": variant,
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "training_protocol_hash": training_hash, "metric_protocol_hash": metric_hash,
        }, run / "checkpoints/last_prob_head.pt")
        updated = [name for name, parameter in model.named_parameters()
                   if parameter.requires_grad and not torch.equal(initial[name], parameter.detach().cpu())]

        best_object = torch.load(run / "checkpoints/best_prob_head.pt", map_location="cpu")
        reload_model = Model(args, variant, True).float()
        reload_model.load_state_dict(raw, strict=False)
        reload_model.initialize_probabilistic_location()
        expected = set(head_state(reload_model))
        reload_result = reload_model.load_state_dict(best_object["prob_head_state_dict"], strict=False)
        reload_ok = (set(best_object["prob_head_state_dict"]) == expected and not reload_result.unexpected_keys
                     and all(name not in expected for name in reload_result.missing_keys))
        reload_model.to(device).eval()
        test_array = np.load(test_mask)["mask"]
        if tuple(test_array.shape) != (len(test_data), args.seq_len, args.enc_in):
            raise RuntimeError(f"test mask shape mismatch: {test_array.shape}")
        offset = 0
        locations, scales, targets, dfs = [], [], [], []
        test_batches = 0
        with torch.no_grad():
            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if max_test is not None and batch_index >= int(max_test):
                    break
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                mask_batch = torch.from_numpy(test_array[offset:offset + len(batch_x)]).float().to(device)
                offset += len(batch_x)
                output, target, _ = forward(batch_x, batch_y, batch_x_mark, batch_y_mark, mask_batch, reload_model)
                locations.append(output["loc"].cpu())
                scales.append(output["scale"].cpu())
                targets.append(target.cpu())
                dfs.append(output["df"].cpu() if output["df"] is not None else None)
                test_batches += 1
        location = torch.cat(locations)
        scale = torch.cat(scales)
        target = torch.cat(targets)
        df = torch.cat([value for value in dfs if value is not None]) if variant in STUDENT_VARIANTS else None
        nll = nll_elements(variant, location, scale, target, df)
        difference = location - target
        mse = float((difference * difference).mean())
        mae = float(difference.abs().mean())
        mean_nll = float(nll.mean())
        intervals = central_intervals(variant, location, scale, df)
        interval_summary = interval_metrics(target, intervals)
        crps = empirical_crps(variant, location, scale, target, df, 64, 52025, 65536)
        probability = {
            "test_mse": mse, "test_mae": mae, "test_rmse": math.sqrt(mse),
            "test_mean_nll": mean_nll, "test_median_element_nll": float(nll.median()),
            # PyTorch quantile has a single-tensor size limit on Weather/Electricity.
            # These diagnostics are exact NumPy CPU reductions after all core metrics.
            "test_p90_element_nll": float(np.quantile(nll.detach().cpu().numpy(), .9)),
            "test_p99_element_nll": float(np.quantile(nll.detach().cpu().numpy(), .99)), "test_crps": crps,
            **interval_summary, "mean_scale": float(scale.mean()), "median_scale": float(scale.median()),
            "p90_scale": float(np.quantile(scale.detach().cpu().numpy(), .9)), "scale_min": float(scale.min()),
            "scale_max": float(scale.max()), "scale_finite": bool(torch.isfinite(scale).all()),
            "scale_positive": bool((scale > 0).all()),
        }
        if df is not None:
            predictive_std = scale * torch.sqrt(df / (df - 2))
            per_channel_df = df[0, 0, :].tolist()
            fixed_df_exact = bool(variant != "student_t_fixed_df8_head" or torch.equal(df, torch.full_like(df, 8.0)))
            probability.update({
                "mean_df": float(df.mean()), "median_df": float(df.median()), "min_df": float(df.min()),
                "max_df": float(df.max()), "df_finite": bool(torch.isfinite(df).all()),
                "df_above_2_01": bool((df > 2.01).all()), "mean_predictive_std": float(predictive_std.mean()),
                "per_channel_df": per_channel_df, "fixed_df_exact": fixed_df_exact,
            })
        else:
            probability.update({key: None for key in (
                "mean_df", "median_df", "min_df", "max_df", "df_finite", "df_above_2_01",
                "mean_predictive_std", "per_channel_df", "fixed_df_exact")})
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        raw_df_present = any(name.endswith("raw_df") for name in trainable_names)
        df_gradient_requirement_pass = df_grad if expects_df_gradient else not raw_df_present
        distribution_support_pass = (
            variant not in STUDENT_VARIANTS
            or probability["df_finite"] and probability["df_above_2_01"]
        )
        fixed_df_requirement_pass = (
            variant != "student_t_fixed_df8_head" or probability["fixed_df_exact"] is True
        )
        passed = (finite_train and finite_grad and loc_grad and scale_grad
                  and df_gradient_requirement_pass and backbone_grads == 0
                  and optimizer_backbone == 0 and bool(updated) and batch_y_unchanged and deterministic_masks
                  and input_mask_applied and reload_ok and probability["scale_finite"] and probability["scale_positive"]
                  and distribution_support_pass and fixed_df_requirement_pass
                  and all(np.isfinite(value) for value in (mean_nll, crps, mse, mae, best)))
        distribution_name = {
            "gaussian_head": "normal",
            "laplace_head": "laplace",
            "student_t_fixed_df8_head": "student_t_fixed_df8",
            "student_t_head": "student_t_per_channel_learnable_df",
        }[variant]
        metrics = {
            "stage": task.get("stage", "gate4b"), "variant": variant, "dataset": dataset, "seed": seed,
            "distribution": distribution_name,
            "fixed_df": 8.0 if variant == "student_t_fixed_df8_head" else None,
            "learnable_df": variant == "student_t_head",
            "experiment_seed": experiment_seed,
            "probability_head_training_seed": experiment_seed,
            "backbone_seed_scope": "fixed_official_checkpoint",
            "train_mask_base_seed": train_mask_base_seed,
            "train_mask_seed_rule": train_mask_seed_rule,
            "dataloader_seed": experiment_seed,
            "worker_seed_rule": "torch.initial_seed()%2**32 -> numpy/python",
            "first_batch_window_indices_sha256": first_batch_window_indices_sha256,
            "missing_type": missing_type, "target_missing_rate": target_rate,
            "actual_train_missing_rate": train_missing / train_total,
            "actual_validation_missing_rate": float(1.0 - validation_array.mean()),
            "actual_test_missing_rate": float(1.0 - test_array.mean()),
            "smoke_only": smoke_only, "train_batches": train_batches,
            "val_batches": min(len(val_loader), int(max_val) if max_val is not None else len(val_loader)),
            "test_batches": test_batches, "train_nll_finite": finite_train,
            "validation_nll_finite": bool(np.isfinite(best)), "test_metrics_finite": True,
            "all_new_gradients_finite": finite_grad, "loc_nonzero_gradient": loc_grad,
            "scale_nonzero_gradient": scale_grad, "raw_df_nonzero_gradient": df_grad,
            "raw_df_parameter_present": raw_df_present,
            "df_gradient_requirement_pass": df_gradient_requirement_pass,
            "distribution_support_pass": distribution_support_pass,
            "fixed_df_requirement_pass": fixed_df_requirement_pass,
            "backbone_non_null_gradient_count": backbone_grads,
            "optimizer_backbone_parameter_count": optimizer_backbone, "trainable_parameter_names": trainable_names,
            "updated_parameter_names": updated, "batch_y_modified": not batch_y_unchanged,
            "test_mask_used_during_training": False, "checkpoint_save": True,
            "test_mask_loaded_after_training": True,
            "checkpoint_strict_reload": reload_ok, "evaluation_after_reload": True,
            "train_mask_deterministic": deterministic_masks, "input_mask_applied": input_mask_applied,
            "mask_application_verified": input_mask_applied, "legacy_unmasked_invalid": False,
            "max_epochs": max_epochs, "patience": int(task.get("patience", 3)),
            "epochs_completed": epochs_completed, "best_epoch": best_epoch, "best_validation_nll": best,
            "early_stopped": early_stopped, "bad_epoch_count_at_stop": bad_epochs,
            "early_stopping_metric": "validation_nll", "test_used_for_model_selection": False,
            "trainable_parameter_count": trainable_parameters, "total_parameter_count": total_parameters,
            "parameter_overhead_ratio": trainable_parameters / (total_parameters - trainable_parameters),
            "target_channels": int(args.enc_in),
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "validation_mask_sha256": sha256_file(validation_mask), "test_mask_sha256": sha256_file(test_mask),
            "training_protocol_hash": training_hash, "metric_protocol_hash": metric_hash,
            "metric_space": "official_normalized", "normalization_reference": "official_train_split_statistics",
            "crps_method": "empirical_monte_carlo", "crps_samples": 64, "crps_seed": 52025,
            "crps_chunk_elements": 65536, "interval_type": "native_parametric",
            "conformal_calibrated": False, "posthoc_calibration": False,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1048576,
            "elapsed_seconds": time.time() - started, **probability,
        }
        status = "SUCCESS" if passed else "FAILED"
    except Exception as exc:
        error = repr(exc)
        metrics = {"stage": task.get("stage", "gate4b"), "variant": variant, "smoke_only": smoke_only,
                   "missing_type": missing_type, "target_missing_rate": target_rate, "error": error,
                   "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started}
    status_payload = {
        "status": status, "stage": task.get("stage", "gate4b"), "experiment_id": run.name,
        "variant": variant, "dataset": dataset, "missing_type": missing_type,
        "distribution": metrics.get("distribution"), "fixed_df": metrics.get("fixed_df"),
        "learnable_df": metrics.get("learnable_df"), "per_channel_df": metrics.get("per_channel_df"),
        "target_missing_rate": target_rate, "actual_train_missing_rate": metrics.get("actual_train_missing_rate"),
        "actual_validation_missing_rate": metrics.get("actual_validation_missing_rate"),
        "actual_test_missing_rate": metrics.get("actual_test_missing_rate"), "seed": seed,
        "experiment_seed": experiment_seed,
        "probability_head_training_seed": experiment_seed,
        "backbone_seed_scope": "fixed_official_checkpoint",
        "train_mask_base_seed": train_mask_base_seed,
        "train_mask_seed_rule": train_mask_seed_rule,
        "smoke_only": smoke_only, "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": metrics.get("source_checkpoint_sha256"),
        "validation_mask": str(validation_mask), "validation_mask_sha256": metrics.get("validation_mask_sha256"),
        "test_mask": str(test_mask), "test_mask_sha256": metrics.get("test_mask_sha256"),
        "training_protocol_hash": metrics.get("training_protocol_hash"),
        "metric_protocol_hash": metrics.get("metric_protocol_hash"), "freeze_backbone": True,
        "trainable_parameter_count": metrics.get("trainable_parameter_count"),
        "optimizer_backbone_parameter_count": metrics.get("optimizer_backbone_parameter_count"),
        "backbone_non_null_gradient_count": metrics.get("backbone_non_null_gradient_count"),
        "max_epochs": metrics.get("max_epochs"), "patience": metrics.get("patience"),
        "epochs_completed": metrics.get("epochs_completed"), "best_epoch": metrics.get("best_epoch"),
        "best_validation_nll": metrics.get("best_validation_nll"), "early_stopped": metrics.get("early_stopped"),
        "early_stopping_metric": metrics.get("early_stopping_metric"),
        "test_used_for_model_selection": metrics.get("test_used_for_model_selection"),
        "test_mse": metrics.get("test_mse"), "test_mae": metrics.get("test_mae"),
        "test_rmse": metrics.get("test_rmse"), "test_mean_nll": metrics.get("test_mean_nll"),
        "test_crps": metrics.get("test_crps"),
        "mean_absolute_coverage_error": metrics.get("mean_absolute_coverage_error"),
        "mean_interval_width": metrics.get("mean_interval_width"),
        "mean_interval_score": metrics.get("mean_interval_score"), "scale_min": metrics.get("scale_min"),
        "scale_max": metrics.get("scale_max"), "scale_finite": metrics.get("scale_finite"),
        "df_min": metrics.get("min_df"), "df_max": metrics.get("max_df"), "df_finite": metrics.get("df_finite"),
        "fixed_df_exact": metrics.get("fixed_df_exact"),
        "raw_df_parameter_present": metrics.get("raw_df_parameter_present"),
        "df_gradient_requirement_pass": metrics.get("df_gradient_requirement_pass"),
        "test_mask_used_during_training": metrics.get("test_mask_used_during_training"),
        "batch_y_modified": metrics.get("batch_y_modified"), "input_mask_applied": metrics.get("input_mask_applied"),
        "mask_application_verified": metrics.get("mask_application_verified"),
        "legacy_unmasked_invalid": False, "target_channels": metrics.get("target_channels"),
        "total_parameter_count": metrics.get("total_parameter_count"),
        "parameter_overhead_ratio": metrics.get("parameter_overhead_ratio"),
        "metric_space": metrics.get("metric_space"),
        "normalization_reference": metrics.get("normalization_reference"),
        "training_started": bool(metrics and not metrics.get("error")),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "error_type": classify_error(error) if error else "", "error_summary": error,
    }
    dump(run / "metrics.json", metrics)
    dump(run / "status.json", status_payload)
    dump(run / "resource_usage.json", {
        "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
        "elapsed_seconds": metrics.get("elapsed_seconds"), "device": "cuda",
    })
    (run / "train.log").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"status": status, **metrics}, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
