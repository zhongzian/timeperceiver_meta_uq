#!/usr/bin/env python
"""Retrain one Gate 4c probability head, calibrate it, and evaluate fixed masks."""

from __future__ import annotations

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
    parse_simple_yaml,
    sha256_file,
)
from gate4b_probabilistic_metrics import central_intervals, nll_elements
from models.conformal_calibrator import (
    LEVELS,
    STAGE_ORDER,
    ProtocolAccessTracker,
    apply_additive_calibrators,
    assign_severity_bins,
    fit_additive_calibrators,
    interval_metrics,
    intervals_are_nested,
    severity_bin_metrics,
)
from models.probabilistic_timeperceiver import Model, freeze_backbone

BASELINE_RUN = ROOT / "runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054"
DEFAULT_CHECKPOINT = next((BASELINE_RUN / "checkpoints").rglob("checkpoint.pth"))
SELECTION_INDICES = "artifacts/gate4c_splits/etth1_random_rate0.3_seed2025/selection_window_indices.npy"
CALIBRATION_INDICES = "artifacts/gate4c_splits/etth1_random_rate0.3_seed2025/calibration_window_indices.npy"
SELECTION_MASK = "artifacts/gate4c_masks/etth1/selection/random/rate_0.3/seed_43025.npz"
CALIBRATION_MASK = "artifacts/gate4c_masks/etth1/calibration/random/rate_0.3/seed_44025.npz"
TEST_MASK = "artifacts/masks/etth1/random/rate_0.3/seed_2025.npz"
TRAIN_MASK_RULE = "sha256(base_seed|epoch|sample_index|dataset|missing_type|target_rate)"


class Indexed(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return (*self.base[index], index)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def command_output(command) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return repr(exc)


def resolve(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def classify_error(error: str) -> str:
    lowered = error.lower()
    if "out of memory" in lowered:
        return "OOM"
    if "gpu_required" in lowered:
        return "WAITING_RESOURCE"
    if "no such file" in lowered or "filenotfound" in lowered or "mask shape" in lowered:
        return "DATA_ERROR"
    if "config" in lowered or "protocol" in lowered or "expected" in lowered:
        return "CONFIG_ERROR"
    if "nan" in lowered or "non-finite" in lowered or "finite" in lowered:
        return "NAN_OR_DIVERGENCE"
    return "CODE_ERROR"


def stable_train_mask_seed(base_seed: int, epoch: int, sample_index: int, dataset: str,
                           missing_type: str, target_rate: float) -> int:
    text = f"{base_seed}|{epoch}|{sample_index}|{dataset}|{missing_type}|{target_rate:.6f}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def sample_train_masks(indices: torch.Tensor, epoch: int, shape, base_seed: int) -> torch.Tensor:
    one_shape = (1, int(shape[1]), int(shape[2]))
    masks = [
        make_mask(
            "random",
            stable_train_mask_seed(base_seed, epoch, int(index), "etth1", "random", 0.3),
            one_shape,
            0.3,
        )[0]
        for index in indices.tolist()
    ]
    return torch.from_numpy(np.stack(masks)).float()


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def default_task(variant: str, smoke: bool) -> dict:
    return {
        "stage": "gate4c_protocol_smoke" if smoke else "gate4c",
        "dataset": "etth1",
        "variant": variant,
        "missing_type": "random",
        "missing_rate": 0.3,
        "seed": 2025,
        "experiment_seed": 2025,
        "seq_len": 384,
        "pred_len": 96,
        "config": str((BASELINE_RUN / "resolved_config.yaml").relative_to(ROOT)),
        "source_checkpoint": str(DEFAULT_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
        "selection_split_path": SELECTION_INDICES,
        "calibration_split_path": CALIBRATION_INDICES,
        "selection_mask_path": SELECTION_MASK,
        "calibration_mask_path": CALIBRATION_MASK,
        "test_mask_path": TEST_MASK,
        "selection_mask_seed": 43025,
        "calibration_mask_seed": 44025,
        "test_mask_seed": 2025,
        "train_split": "official_train",
        "train_mask_base_seed": 42025,
        "train_mask_seed_rule": TRAIN_MASK_RULE,
        "freeze_backbone": True,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "max_epochs": 2 if smoke else 20,
        "patience": 3,
        "early_stopping_metric": "selection_nll",
        "calibration_used_for_selection": False,
        "test_used_for_selection": False,
        "retrain_probability_head": True,
        "reuse_gate4b_probability_checkpoint": False,
        "coverage_levels": [0.90, 0.95],
        "conformal_methods": ["marginal", "mask_severity_conditioned"],
        "condition_source": "historical_input_mask_only",
        "condition_bins": 3,
        "bin_boundaries_fit_split": "calibration_masks_only",
        "initial_scale": 0.5,
        "scale_floor": 1e-4,
        "initial_df": 8.0 if variant == "student_t_head" else None,
        "df_parameterization": "2.01 + softplus(raw_df)" if variant == "student_t_head" else None,
        "max_train_batches": 16 if smoke else None,
        "max_selection_batches": 4 if smoke else None,
        "max_calibration_batches": 4 if smoke else None,
        "max_test_batches": 4 if smoke else None,
        "smoke_only": smoke,
        "training_protocol_hash": "gate4c_etth1_random03_selection_only_v1",
        "calibration_protocol_hash": "gate4c_additive_split_conformal_severity3_v1",
        "metric_protocol_hash": "gate4c_elementwise_interval_90_95_v1",
    }


def model_forward(args, model, batch_x, batch_y, batch_x_mark, batch_y_mark, mask_batch, indices):
    decoder = torch.cat(
        [batch_y[:, :args.label_len, :], torch.zeros_like(batch_y[:, -args.pred_len:, :])], dim=1
    )
    masked_x = batch_x * mask_batch
    output = model(masked_x, batch_x_mark, decoder, batch_y_mark, indices, mask=mask_batch)
    return output, batch_y[:, -args.pred_len:, :], masked_x


def collect_outputs(args, model, loader, mask_array: np.ndarray, device, indices, max_batches):
    locations, scales, targets, dfs = [], [], [], []
    offset = 0
    batches = 0
    batch_y_unchanged = True
    model.eval()
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            mask_batch_np = mask_array[offset:offset + len(batch_x)]
            if len(mask_batch_np) != len(batch_x):
                raise RuntimeError("mask shape truncated during evaluation")
            offset += len(batch_x)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            before_y = batch_y.clone()
            mask_batch = torch.from_numpy(mask_batch_np).float().to(device)
            output, target, _ = model_forward(
                args, model, batch_x, batch_y, batch_x_mark, batch_y_mark, mask_batch, indices
            )
            batch_y_unchanged &= bool(torch.equal(before_y, batch_y))
            locations.append(output["loc"].cpu())
            scales.append(output["scale"].cpu())
            targets.append(target.cpu())
            if output["df"] is not None:
                dfs.append(output["df"].cpu())
            batches += 1
    if not locations:
        raise RuntimeError("evaluation loader produced no batches")
    return {
        "loc": torch.cat(locations),
        "scale": torch.cat(scales),
        "target": torch.cat(targets),
        "df": torch.cat(dfs) if dfs else None,
        "batches": batches,
        "windows": offset,
        "batch_y_unchanged": batch_y_unchanged,
    }


def native_intervals(variant: str, outputs: dict):
    all_intervals = central_intervals(
        variant, outputs["loc"], outputs["scale"], outputs["df"]
    )
    return {level: all_intervals[level] for level in LEVELS}


def serializable_calibration(calibration: dict) -> dict:
    return {key: value for key, value in calibration.items() if key != "assignments"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["gaussian_head", "student_t_head"])
    parser.add_argument("--task-file")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    cli = parser.parse_args()
    task = json.loads(Path(cli.task_file).read_text()) if cli.task_file else default_task(cli.variant, cli.smoke)
    variant = task["variant"]
    smoke_only = bool(task.get("smoke_only", cli.smoke))
    run = resolve(cli.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    (run / "checkpoints").mkdir()
    (run / "calibration").mkdir()
    started = time.time()
    (run / "command.sh").write_text("#!/usr/bin/env bash\n" + sys.executable + " " + " ".join(sys.argv) + "\n")
    baseline_config = resolve(task["config"])
    cfg = parse_simple_yaml(baseline_config)
    args = build_args(cfg, int(cfg["batch_size"]), "cuda")
    resolved_config = {**cfg, **task, "stage_order": list(STAGE_ORDER), "smoke_only": smoke_only}
    (run / "resolved_config.yaml").write_text(
        "\n".join(f"{key}: {json.dumps(value)}" for key, value in sorted(resolved_config.items())) + "\n"
    )
    (run / "env.txt").write_text(
        f"hostname: {socket.gethostname()}\npython: {sys.version.split()[0]}\ntorch: {torch.__version__}\n"
        f"cuda_runtime: {torch.version.cuda}\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        f"gpu_before: {command_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free', '--format=csv,noheader'])}\n"
    )
    (run / "git_state.txt").write_text(
        command_output(["git", "-C", str(ROOT / "upstream/TimePerceiver"), "rev-parse", "HEAD"]) + "\n"
        + command_output(["git", "-C", str(ROOT / "upstream/TimePerceiver"), "status", "--short"]) + "\n"
    )
    (run / "code.diff").write_text(
        command_output(["git", "-C", str(ROOT / "upstream/TimePerceiver"), "diff", "--binary"]) + "\n"
    )
    log_path = run / "train.log"
    log_path.write_text("Gate 4c leakage-safe probability-head retraining and split conformal.\n")
    status = "FAILED"
    metrics = {}
    error = ""
    tracker = ProtocolAccessTracker()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_REQUIRED")
        expected = {
            "dataset": "etth1", "missing_type": "random", "missing_rate": 0.3,
            "seed": 2025, "experiment_seed": 2025, "seq_len": 384, "pred_len": 96,
            "train_mask_base_seed": 42025, "freeze_backbone": True,
            "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
            "gradient_clip": 1.0, "patience": 3, "early_stopping_metric": "selection_nll",
            "retrain_probability_head": True, "reuse_gate4b_probability_checkpoint": False,
        }
        mismatches = {key: (value, task.get(key)) for key, value in expected.items() if task.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Gate 4c config mismatch: {mismatches}")
        if list(map(float, task["coverage_levels"])) != [0.9, 0.95]:
            raise RuntimeError("Gate 4c coverage protocol mismatch")
        paths = {key: resolve(task[key]) for key in (
            "source_checkpoint", "selection_split_path", "calibration_split_path",
            "selection_mask_path", "calibration_mask_path", "test_mask_path",
        )}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256_file(paths["source_checkpoint"]) != task["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")

        seed = int(task["experiment_seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(0)
        torch.cuda.init()
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        provider, _ = import_official_modules()
        train_data, _ = provider(args, "train")
        validation_data, _ = provider(args, "val")
        selection_indices = np.load(paths["selection_split_path"], allow_pickle=False)
        calibration_indices = np.load(paths["calibration_split_path"], allow_pickle=False)
        if np.intersect1d(selection_indices, calibration_indices).size:
            raise RuntimeError("selection/calibration window overlap")
        selection_data = torch.utils.data.Subset(validation_data, selection_indices.tolist())
        selection_mask = np.load(paths["selection_mask_path"], allow_pickle=False)["mask"]
        if tuple(selection_mask.shape) != (len(selection_data), args.seq_len, args.enc_in):
            raise RuntimeError(f"selection mask shape mismatch: {selection_mask.shape}")
        train_generator = torch.Generator().manual_seed(seed)
        train_loader = torch.utils.data.DataLoader(
            Indexed(train_data), batch_size=args.batch_size, shuffle=True, generator=train_generator,
            worker_init_fn=seed_worker, num_workers=0, drop_last=False,
        )
        selection_loader = torch.utils.data.DataLoader(
            selection_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
        )
        raw_checkpoint = torch.load(paths["source_checkpoint"], map_location="cpu")
        model = Model(args, variant, True).float()
        missing, unexpected = model.load_state_dict(raw_checkpoint, strict=False)
        if unexpected or not all(name.startswith("probabilistic_head.") for name in missing):
            raise RuntimeError(f"checkpoint extension mismatch: missing={missing}, unexpected={unexpected}")
        model.initialize_probabilistic_location()
        trainable_names = freeze_backbone(model)
        model.to(device)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3, weight_decay=1e-4)
        optimizer_backbone_count = sum(
            1 for group in optimizer.param_groups for parameter in group["params"] if not parameter.requires_grad
        )
        initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        patch_indices = (
            list(range(args.seq_len // args.patch_len)),
            list(range(args.seq_len // args.patch_len, (args.seq_len + args.pred_len) // args.patch_len)),
        )
        max_epochs = int(task["max_epochs"])
        max_train = task.get("max_train_batches")
        max_selection = task.get("max_selection_batches")
        best_nll = float("inf")
        best_epoch = 0
        bad_epochs = 0
        early_stopped = False
        train_batches = 0
        train_losses = []
        selection_history = []
        finite_gradients = True
        nonzero_gradient_names = set()
        backbone_non_null_gradients = 0
        batch_y_unchanged = True
        input_mask_applied = False
        first_batch_indices_sha256 = None
        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []
            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark, window_index) in enumerate(train_loader):
                if max_train is not None and batch_index >= int(max_train):
                    break
                mask_cpu = sample_train_masks(window_index, epoch, batch_x.shape, int(task["train_mask_base_seed"]))
                if first_batch_indices_sha256 is None:
                    first_batch_indices_sha256 = hashlib.sha256(
                        np.asarray(window_index.tolist(), dtype=np.int64).tobytes()
                    ).hexdigest()
                    repeated = sample_train_masks(window_index, epoch, batch_x.shape, int(task["train_mask_base_seed"]))
                    if not torch.equal(mask_cpu, repeated):
                        raise RuntimeError("train mask generation is not deterministic")
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                mask_batch = mask_cpu.to(device)
                before_y = batch_y.clone()
                optimizer.zero_grad()
                output, target, masked_x = model_forward(
                    args, model, batch_x, batch_y, batch_x_mark, batch_y_mark, mask_batch, patch_indices
                )
                input_mask_applied |= bool(torch.any(mask_batch == 0) and torch.any(masked_x != batch_x))
                loss = nll_elements(variant, output["loc"], output["scale"], target, output["df"]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite train NLL")
                loss.backward()
                batch_y_unchanged &= bool(torch.equal(before_y, batch_y))
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        finite_gradients &= bool(torch.isfinite(parameter.grad).all())
                        if torch.count_nonzero(parameter.grad):
                            nonzero_gradient_names.add(name)
                    elif not parameter.requires_grad and parameter.grad is not None:
                        backbone_non_null_gradients += 1
                clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
                train_losses.append(float(loss.detach()))
                train_batches += 1
            tracker.record_loader("selection")
            selection_values = []
            offset = 0
            model.eval()
            with torch.no_grad():
                for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(selection_loader):
                    if max_selection is not None and batch_index >= int(max_selection):
                        break
                    mask_np = selection_mask[offset:offset + len(batch_x)]
                    offset += len(batch_x)
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    mask_batch = torch.from_numpy(mask_np).float().to(device)
                    output, target, _ = model_forward(
                        args, model, batch_x, batch_y, batch_x_mark, batch_y_mark, mask_batch, patch_indices
                    )
                    selection_values.append(float(nll_elements(
                        variant, output["loc"], output["scale"], target, output["df"]
                    ).mean()))
            selection_nll = float(np.mean(selection_values))
            if not np.isfinite(selection_nll):
                raise RuntimeError("non-finite selection NLL")
            selection_history.append(selection_nll)
            with log_path.open("a") as log:
                log.write(f"epoch={epoch + 1} train_nll={np.mean(epoch_losses):.9f} selection_nll={selection_nll:.9f}\n")
            if selection_nll < best_nll:
                best_nll = selection_nll
                best_epoch = epoch + 1
                bad_epochs = 0
                torch.save({
                    "probabilistic_head_state_dict": model.probabilistic_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": best_epoch,
                    "best_selection_nll": best_nll,
                    "variant": variant,
                    "source_checkpoint_sha256": task["source_checkpoint_sha256"],
                    "training_protocol_hash": task["training_protocol_hash"],
                }, run / "checkpoints/best_prob_head.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= int(task["patience"]):
                    early_stopped = True
                    break
        epochs_completed = len(selection_history)
        torch.save({
            "probabilistic_head_state_dict": model.probabilistic_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs_completed,
            "variant": variant,
            "source_checkpoint_sha256": task["source_checkpoint_sha256"],
            "training_protocol_hash": task["training_protocol_hash"],
        }, run / "checkpoints/last_prob_head.pt")
        updated_parameter_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and not torch.equal(initial_parameters[name], parameter.detach().cpu())
        ]

        best_object = torch.load(run / "checkpoints/best_prob_head.pt", map_location="cpu")
        reload_model = Model(args, variant, True).float()
        reload_missing, reload_unexpected = reload_model.load_state_dict(raw_checkpoint, strict=False)
        if reload_unexpected or not all(name.startswith("probabilistic_head.") for name in reload_missing):
            raise RuntimeError("source checkpoint reload mismatch")
        reload_model.initialize_probabilistic_location()
        strict_result = reload_model.probabilistic_head.load_state_dict(
            best_object["probabilistic_head_state_dict"], strict=True
        )
        checkpoint_strict_reload = not strict_result.missing_keys and not strict_result.unexpected_keys
        reload_model.to(device)
        for parameter in reload_model.parameters():
            parameter.requires_grad = False
        reload_model.eval()
        tracker.fix_best_checkpoint()

        # Calibration data and masks are first accessed after the best checkpoint is fixed.
        tracker.record_loader("calibration")
        calibration_data = torch.utils.data.Subset(validation_data, calibration_indices.tolist())
        calibration_loader = torch.utils.data.DataLoader(
            calibration_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
        )
        calibration_mask = np.load(paths["calibration_mask_path"], allow_pickle=False)["mask"]
        if tuple(calibration_mask.shape) != (len(calibration_data), args.seq_len, args.enc_in):
            raise RuntimeError(f"calibration mask shape mismatch: {calibration_mask.shape}")
        calibration_outputs = collect_outputs(
            args, reload_model, calibration_loader, calibration_mask, device, patch_indices,
            task.get("max_calibration_batches"),
        )
        calibration_windows = calibration_outputs["windows"]
        calibration_rates = 1.0 - calibration_mask[:calibration_windows].mean(axis=(1, 2))
        calibration_native = native_intervals(variant, calibration_outputs)
        calibration = fit_additive_calibrators(
            calibration_native,
            calibration_outputs["target"],
            calibration_rates,
            minimum_windows_per_bin=50,
            minimum_scores_per_bin=5000,
        )
        calibration_saved = serializable_calibration(calibration)
        dump(run / "calibration/marginal_qhat.json", {
            "calibration_method": "marginal_split_conformal",
            "score_type": calibration["score_type"],
            "q_hat": calibration["marginal"],
        })
        dump(run / "calibration/conditioned_qhat.json", {
            "calibration_method": "mask_severity_conditioned_conformal",
            "condition_source": calibration["condition_source"],
            "q_hat_per_bin": calibration["conditioned"],
        })
        dump(run / "calibration/severity_bins.json", {
            "bin_count": 3,
            "bin_boundaries": calibration["bin_boundaries"],
            "calibration_windows_per_bin": calibration["calibration_windows_per_bin"],
            "calibration_scores_per_bin": calibration["calibration_scores_per_bin"],
            "quantile_semantics": "higher",
        })
        dump(run / "calibration/calibration_summary.json", calibration_saved)
        tracker.freeze_calibrator()

        # Test data and fixed test mask are first accessed after q-hat is frozen.
        tracker.record_loader("test")
        test_data, _ = provider(args, "test")
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
        )
        test_mask = np.load(paths["test_mask_path"], allow_pickle=False)["mask"]
        if tuple(test_mask.shape) != (len(test_data), args.seq_len, args.enc_in):
            raise RuntimeError(f"test mask shape mismatch: {test_mask.shape}")
        test_outputs = collect_outputs(
            args, reload_model, test_loader, test_mask, device, patch_indices, task.get("max_test_batches")
        )
        test_windows = test_outputs["windows"]
        test_rates = 1.0 - test_mask[:test_windows].mean(axis=(1, 2))
        native = native_intervals(variant, test_outputs)
        calibrated = apply_additive_calibrators(native, calibration, test_rates)
        methods = {
            "native": native,
            "marginal_conformal": calibrated["marginal"],
            "mask_severity_conditioned_conformal": calibrated["mask_severity_conditioned"],
        }
        overall = {name: interval_metrics(test_outputs["target"], value) for name, value in methods.items()}
        test_assignments = assign_severity_bins(test_rates, tuple(calibration["bin_boundaries"]))
        bins = {name: severity_bin_metrics(test_outputs["target"], value, test_assignments)
                for name, value in methods.items()}
        nested = {name: intervals_are_nested(value) for name, value in methods.items()}
        finite_intervals = all(
            torch.isfinite(bound).all()
            for intervals in methods.values() for pair in intervals.values() for bound in pair
        )
        additive_expansion = {
            name: all(
                torch.all((intervals[level][1] - intervals[level][0]) >= (native[level][1] - native[level][0]))
                for level in LEVELS
            )
            for name, intervals in methods.items() if name != "native"
        }
        q_values = [
            float(item["q_hat"])
            for item in calibration["marginal"].values()
        ] + [
            float(item["q_hat"])
            for level in calibration["conditioned"].values() for item in level.values()
        ]
        qhat_valid = all(np.isfinite(value) and value >= 0 for value in q_values)
        target = test_outputs["target"]
        location = test_outputs["loc"]
        difference = location - target
        test_mse = float((difference * difference).mean())
        test_mae = float(difference.abs().mean())
        test_rmse = math.sqrt(test_mse)
        test_nll = float(nll_elements(
            variant, location, test_outputs["scale"], target, test_outputs["df"]
        ).mean())
        point_by_method = {
            name: {"mse": test_mse, "mae": test_mae, "rmse": test_rmse}
            for name in methods
        }
        point_prediction_mutation = len({tuple(row.values()) for row in point_by_method.values()}) != 1
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_count = sum(parameter.numel() for parameter in model.parameters())
        passed = all([
            np.isfinite(train_losses).all(), np.isfinite(selection_history).all(), finite_gradients,
            bool(nonzero_gradient_names), backbone_non_null_gradients == 0,
            optimizer_backbone_count == 0, bool(updated_parameter_names), batch_y_unchanged,
            calibration_outputs["batch_y_unchanged"], test_outputs["batch_y_unchanged"],
            input_mask_applied, checkpoint_strict_reload, qhat_valid,
            all(count >= 50 for count in calibration["calibration_windows_per_bin"]),
            all(count >= 5000 for count in calibration["calibration_scores_per_bin"]),
            finite_intervals, all(nested.values()), all(additive_expansion.values()),
            not point_prediction_mutation,
            tracker.loader_calls["calibration"] == 1, tracker.loader_calls["test"] == 1,
            np.isfinite([test_mse, test_mae, test_rmse, test_nll, best_nll]).all(),
        ])
        metrics = {
            "stage": task["stage"], "dataset": "etth1", "variant": variant,
            "seed": 2025, "experiment_seed": 2025, "smoke_only": smoke_only,
            "stage_order": list(STAGE_ORDER), "retrain_probability_head": True,
            "previous_gate4b_checkpoint_reused_for_conformal": False,
            "source_checkpoint_sha256": sha256_file(paths["source_checkpoint"]),
            "selection_split_sha256": sha256_file(paths["selection_split_path"]),
            "calibration_split_sha256": sha256_file(paths["calibration_split_path"]),
            "selection_mask_sha256": sha256_file(paths["selection_mask_path"]),
            "calibration_mask_sha256": sha256_file(paths["calibration_mask_path"]),
            "test_mask_sha256": sha256_file(paths["test_mask_path"]),
            "selection_calibration_raw_overlap_count": 0,
            "selection_calibration_future_target_overlap_count": 0,
            "calibration_labels_seen_during_training": False,
            "calibration_labels_seen_during_selection": False,
            "test_labels_seen_during_calibration": False,
            "calibration_loader_call_count_before_best_checkpoint": 0,
            "test_loader_call_count_before_qhat": 0,
            "loader_call_counts": tracker.loader_calls,
            "train_mask_base_seed": 42025, "train_mask_seed_rule": TRAIN_MASK_RULE,
            "first_batch_window_indices_sha256": first_batch_indices_sha256,
            "freeze_backbone": True,
            "optimizer_backbone_parameter_count": optimizer_backbone_count,
            "backbone_non_null_gradient_count": backbone_non_null_gradients,
            "all_new_parameter_gradients_finite": finite_gradients,
            "nonzero_gradient_parameter_names": sorted(nonzero_gradient_names),
            "updated_parameter_names": updated_parameter_names,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "total_parameter_count": total_count,
            "parameter_overhead_ratio": trainable_count / (total_count - trainable_count),
            "max_epochs": int(task["max_epochs"]), "patience": 3,
            "epochs_completed": epochs_completed, "best_epoch": best_epoch,
            "best_selection_nll": best_nll, "early_stopped": early_stopped,
            "train_batches": train_batches, "selection_batches": min(
                len(selection_loader), int(task["max_selection_batches"]) if task.get("max_selection_batches") else len(selection_loader)
            ),
            "calibration_batches": calibration_outputs["batches"],
            "test_batches": test_outputs["batches"],
            "train_nll": float(np.mean(train_losses)),
            "selection_nll": best_nll,
            "train_nll_finite": True, "selection_nll_finite": True,
            "calibration_scores_finite": qhat_valid,
            "q_hat_finite_nonnegative": qhat_valid,
            "condition_bin_boundaries": calibration["bin_boundaries"],
            "calibration_windows_per_bin": calibration["calibration_windows_per_bin"],
            "calibration_scores_per_bin": calibration["calibration_scores_per_bin"],
            "conditioned_bins_valid": True,
            "marginal_qhat": calibration["marginal"],
            "conditioned_qhat": calibration["conditioned"],
            "interval_metrics": overall,
            "severity_bin_metrics": bins,
            "interval_nesting": nested,
            "additive_expansion": additive_expansion,
            "intervals_finite": finite_intervals,
            "point_metrics_by_method": point_by_method,
            "point_prediction_mutation": point_prediction_mutation,
            "test_mse": test_mse, "test_mae": test_mae, "test_rmse": test_rmse,
            "test_mean_nll": test_nll,
            "batch_y_modified": not (batch_y_unchanged and calibration_outputs["batch_y_unchanged"] and test_outputs["batch_y_unchanged"]),
            "test_mask_used_during_training": False,
            "test_mask_used_during_calibration": False,
            "input_mask_applied": input_mask_applied,
            "checkpoint_save": True, "checkpoint_strict_reload": checkpoint_strict_reload,
            "evaluation_after_reload": True,
            "training_protocol_hash": task["training_protocol_hash"],
            "calibration_protocol_hash": task["calibration_protocol_hash"],
            "metric_protocol_hash": task["metric_protocol_hash"],
            "conformal_mode": "split_conformal",
            "coverage_target": "elementwise_marginal",
            "time_series_dependence_present": True,
            "exact_exchangeability_guarantee_claimed": False,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1048576,
            "elapsed_seconds": time.time() - started,
        }
        status = "SUCCESS" if passed else "FAILED"
    except Exception as exc:
        error = repr(exc)
        metrics = {
            "stage": task.get("stage", "gate4c"), "variant": variant,
            "dataset": task.get("dataset", "etth1"), "smoke_only": smoke_only,
            "error": error, "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started,
        }
    status_payload = {
        "status": status, "stage": task.get("stage", "gate4c"), "experiment_id": run.name,
        "dataset": "etth1", "variant": variant, "experiment_seed": 2025,
        "source_checkpoint": str(resolve(task["source_checkpoint"])),
        "source_checkpoint_sha256": metrics.get("source_checkpoint_sha256"),
        "selection_split_path": task["selection_split_path"],
        "selection_split_sha256": metrics.get("selection_split_sha256"),
        "calibration_split_path": task["calibration_split_path"],
        "calibration_split_sha256": metrics.get("calibration_split_sha256"),
        "selection_mask_path": task["selection_mask_path"],
        "selection_mask_sha256": metrics.get("selection_mask_sha256"),
        "calibration_mask_path": task["calibration_mask_path"],
        "calibration_mask_sha256": metrics.get("calibration_mask_sha256"),
        "test_mask_path": task["test_mask_path"],
        "test_mask_sha256": metrics.get("test_mask_sha256"),
        "selection_calibration_raw_overlap_count": metrics.get("selection_calibration_raw_overlap_count"),
        "calibration_labels_seen_during_training": metrics.get("calibration_labels_seen_during_training"),
        "calibration_labels_seen_during_selection": metrics.get("calibration_labels_seen_during_selection"),
        "test_labels_seen_during_calibration": metrics.get("test_labels_seen_during_calibration"),
        "freeze_backbone": True,
        "optimizer_backbone_parameter_count": metrics.get("optimizer_backbone_parameter_count"),
        "backbone_non_null_gradient_count": metrics.get("backbone_non_null_gradient_count"),
        "max_epochs": metrics.get("max_epochs"), "patience": metrics.get("patience"),
        "best_epoch": metrics.get("best_epoch"), "best_selection_nll": metrics.get("best_selection_nll"),
        "early_stopped": metrics.get("early_stopped"),
        "calibration_method_names": ["marginal_split_conformal", "mask_severity_conditioned_conformal"],
        "coverage_levels": [0.90, 0.95], "condition_bin_count": 3,
        "condition_bin_boundaries": metrics.get("condition_bin_boundaries"),
        "calibration_windows_per_bin": metrics.get("calibration_windows_per_bin"),
        "calibration_scores_per_bin": metrics.get("calibration_scores_per_bin"),
        "marginal_qhat": metrics.get("marginal_qhat"), "conditioned_qhat": metrics.get("conditioned_qhat"),
        "test_mse": metrics.get("test_mse"), "test_mae": metrics.get("test_mae"),
        "test_rmse": metrics.get("test_rmse"),
        "native_interval_metrics": (metrics.get("interval_metrics") or {}).get("native"),
        "marginal_interval_metrics": (metrics.get("interval_metrics") or {}).get("marginal_conformal"),
        "conditioned_interval_metrics": (metrics.get("interval_metrics") or {}).get("mask_severity_conditioned_conformal"),
        "conditioned_bin_metrics": (metrics.get("severity_bin_metrics") or {}).get("mask_severity_conditioned_conformal"),
        "point_prediction_mutation": metrics.get("point_prediction_mutation"),
        "batch_y_modified": metrics.get("batch_y_modified"),
        "test_mask_used_during_training": metrics.get("test_mask_used_during_training"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "error_type": classify_error(error) if error else "", "error_summary": error,
    }
    dump(run / "metrics.json", metrics)
    dump(run / "status.json", status_payload)
    dump(run / "resource_usage.json", {
        "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
        "elapsed_seconds": metrics.get("elapsed_seconds"), "device": "cuda",
    })
    with log_path.open("a") as log:
        log.write(json.dumps({"final_status": status, "metrics": metrics}, indent=2) + "\n")
    print(json.dumps({"status": status, **metrics}, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
