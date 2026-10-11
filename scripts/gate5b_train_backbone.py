#!/usr/bin/env python3
"""Train or smoke-test a clean deterministic PatchTST backbone for Gate 5b0."""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate3b_smoke_missing_eval import build_args, import_official_modules, parse_simple_yaml  # noqa: E402
from gate5b_common import (  # noqa: E402
    DATASETS,
    PATCHTST_COMMIT,
    ROOT,
    TIMEPERCEIVER_COMMIT,
    backbone_config,
    dump,
    environment_text,
    resolve,
    set_seed,
    sha256_file,
    upstream_diff_text,
    upstream_state_text,
    write_resolved,
)
from models.patchtst_backbone import PatchTSTBackbone  # noqa: E402


def classify_error(error: str) -> str:
    lowered = error.lower()
    if "out of memory" in lowered:
        return "OOM"
    if "gpu_required" in lowered:
        return "WAITING_RESOURCE"
    if "no such file" in lowered or "filenotfound" in lowered:
        return "DATA_ERROR"
    if "config" in lowered or "mismatch" in lowered or "expected" in lowered:
        return "CONFIG_ERROR"
    if "non-finite" in lowered or "nan" in lowered:
        return "NAN_OR_DIVERGENCE"
    return "CODE_ERROR"


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_payload(model, optimizer, scheduler, epoch, best_validation_mse, task):
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(), "epoch": int(epoch),
        "best_validation_mse": float(best_validation_mse), "dataset": task["dataset"],
        "experiment_seed": int(task["experiment_seed"]),
        "patchtst_upstream_commit": PATCHTST_COMMIT,
        "training_protocol_hash": task["training_protocol_hash"],
        "numpy_random_state": np.random.get_state(), "torch_random_state": torch.get_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if torch.cuda.is_available():
        payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def validate_task(task):
    required = {
        "backbone": "patchtst", "seq_len": 384, "pred_len": 96,
        "missingness": "none", "experiment_seed": 2025,
        "early_stopping_metric": "validation_mse", "test_used_for_model_selection": False,
    }
    errors = [f"{key}={task.get(key)!r}, expected {expected!r}" for key, expected in required.items()
              if task.get(key) != expected]
    if task.get("dataset") not in DATASETS:
        errors.append(f"unsupported dataset {task.get('dataset')!r}")
    if int(task.get("batch_size", -1)) <= 0:
        errors.append("batch_size must be positive")
    if int(task.get("max_epochs", -1)) <= 0:
        errors.append("max_epochs must be positive")
    if task.get("loss") != "MSE" or task.get("optimizer") != "Adam":
        errors.append("clean backbone requires MSE and Adam")
    if bool(task.get("smoke_only")):
        expected_limits = {"max_epochs": 1, "max_train_batches": 4, "max_val_batches": 2, "max_test_batches": 2}
        for key, expected in expected_limits.items():
            if int(task.get(key, -1)) != expected:
                errors.append(f"smoke {key} must equal {expected}")
    else:
        if any(task.get(key) is not None for key in ("max_train_batches", "max_val_batches", "max_test_batches")):
            errors.append("formal backbone task contains smoke batch limits")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--run-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    cli = parser.parse_args()
    if cli.task_file:
        task = json.loads(Path(cli.task_file).read_text())
    elif cli.dataset:
        task = backbone_config(cli.dataset, smoke=cli.smoke)
    else:
        raise SystemExit("--task-file or --dataset is required")
    errors = validate_task(task)
    if cli.validate_only:
        print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors, "task": task}, indent=2))
        return 0 if not errors else 2
    if errors:
        raise SystemExit("; ".join(errors))
    if not cli.run_dir:
        raise SystemExit("--run-dir is required outside --validate-only")

    run = Path(cli.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    (run / "checkpoints").mkdir()
    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    (run / "command.sh").write_text("#!/usr/bin/env bash\n" + command + "\n")
    (run / "env.txt").write_text(environment_text())
    (run / "git_state.txt").write_text(upstream_state_text())
    (run / "code.diff").write_text(upstream_diff_text())
    write_resolved(run / "resolved_config.yaml", task)
    log_path = run / "train.log"
    log_path.write_text("Gate 5b0 deterministic clean-backbone training.\n")
    started = time.time()
    status = "FAILED"
    metrics = {}
    error = ""

    def log(message):
        with log_path.open("a") as handle:
            handle.write(message.rstrip() + "\n")
        print(message, flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_REQUIRED")
        dataset = task["dataset"]
        set_seed(int(task["experiment_seed"]))
        torch.cuda.set_device(0)
        torch.cuda.init()
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        provider, _ = import_official_modules()
        baseline = parse_simple_yaml(resolve(task["config"]))
        args = build_args(baseline, int(task["batch_size"]), "cuda")
        args.seq_len = 384
        args.pred_len = 96
        args.batch_size = int(task["batch_size"])
        train_data, _ = provider(args, "train")
        val_data, _ = provider(args, "val")
        test_data, _ = provider(args, "test")
        generator = torch.Generator().manual_seed(int(task["experiment_seed"]))
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
            num_workers=0, drop_last=False, pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
            drop_last=False, pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
            drop_last=False, pin_memory=True,
        )
        model = PatchTSTBackbone(task).float().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(task["learning_rate"]))
        max_train = task.get("max_train_batches")
        train_steps = min(len(train_loader), int(max_train)) if max_train is not None else len(train_loader)
        scheduler = None
        if task["lradj"] == "TST":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, steps_per_epoch=train_steps, pct_start=float(task["pct_start"]),
                epochs=int(task["max_epochs"]), max_lr=float(task["learning_rate"]),
            )
        criterion = torch.nn.MSELoss()
        best = float("inf")
        best_epoch = 0
        bad_epochs = 0
        train_batches = 0
        val_batches = 0
        forward_finite = True
        loss_finite = True
        backward_finite = True
        gradient_finite = True
        nonzero_gradient = False
        output_shape_correct = True
        for epoch in range(int(task["max_epochs"])):
            model.train()
            epoch_losses = []
            for batch_index, (batch_x, batch_y, _, _) in enumerate(train_loader):
                if max_train is not None and batch_index >= int(max_train):
                    break
                batch_x = batch_x.float().to(device, non_blocking=True)
                target = batch_y[:, -args.pred_len:, :].float().to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                forecast = model(batch_x)
                output_shape_correct &= tuple(forecast.shape) == tuple(target.shape)
                forward_finite &= bool(torch.isfinite(forecast).all())
                loss = criterion(forecast, target)
                loss_finite &= bool(torch.isfinite(loss))
                loss.backward()
                backward_finite &= loss.grad_fn is not None
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        gradient_finite &= bool(torch.isfinite(parameter.grad).all())
                        nonzero_gradient |= bool(torch.count_nonzero(parameter.grad))
                clip_grad_norm_(model.parameters(), float(task["gradient_clip"]))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                train_batches += 1
                epoch_losses.append(float(loss.detach()))
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_index, (batch_x, batch_y, _, _) in enumerate(val_loader):
                    if task.get("max_val_batches") is not None and batch_index >= int(task["max_val_batches"]):
                        break
                    batch_x = batch_x.float().to(device, non_blocking=True)
                    target = batch_y[:, -args.pred_len:, :].float().to(device, non_blocking=True)
                    val_losses.append(float(criterion(model(batch_x), target)))
            val_batches = len(val_losses)
            validation_mse = float(np.mean(val_losses))
            if not np.isfinite(validation_mse):
                raise RuntimeError("non-finite validation MSE")
            if validation_mse < best:
                best = validation_mse
                best_epoch = epoch + 1
                bad_epochs = 0
                atomic_torch_save(
                    checkpoint_payload(model, optimizer, scheduler, best_epoch, best, task),
                    run / "checkpoints/best_backbone.pt",
                )
            else:
                bad_epochs += 1
            if task["lradj"] == "type3":
                adjust_epoch = epoch + 1
                lr = (
                    float(task["learning_rate"])
                    if adjust_epoch < 3
                    else float(task["learning_rate"]) * (0.9 ** (adjust_epoch - 3))
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
            log(
                f"epoch={epoch + 1} train_mse={np.mean(epoch_losses):.8f} "
                f"validation_mse={validation_mse:.8f} best={best:.8f}"
            )
            if bad_epochs >= int(task["patience"]):
                break
        atomic_torch_save(
            checkpoint_payload(model, optimizer, scheduler, epoch + 1, best, task),
            run / "checkpoints/last_backbone.pt",
        )
        best_object = torch.load(run / "checkpoints/best_backbone.pt", map_location="cpu")
        reload_model = PatchTSTBackbone(task).float()
        reload_result = reload_model.load_state_dict(best_object["model_state_dict"], strict=True)
        strict_reload = not reload_result.missing_keys and not reload_result.unexpected_keys
        reload_model.to(device).eval()

        def evaluate(mask_array=None):
            predictions = []
            targets = []
            offset = 0
            batches = 0
            with torch.no_grad():
                for batch_index, (batch_x, batch_y, _, _) in enumerate(test_loader):
                    if task.get("max_test_batches") is not None and batch_index >= int(task["max_test_batches"]):
                        break
                    batch_x = batch_x.float().to(device, non_blocking=True)
                    if mask_array is not None:
                        mask = torch.from_numpy(mask_array[offset:offset + len(batch_x)]).float().to(device)
                        batch_x = batch_x * mask
                    offset += len(batch_x)
                    target = batch_y[:, -args.pred_len:, :].float().to(device, non_blocking=True)
                    predictions.append(reload_model(batch_x).cpu())
                    targets.append(target.cpu())
                    batches += 1
            prediction = torch.cat(predictions)
            target = torch.cat(targets)
            difference = prediction - target
            mse = float((difference * difference).mean())
            mae = float(difference.abs().mean())
            return {"mse": mse, "mae": mae, "rmse": math.sqrt(mse), "batches": batches,
                    "finite": bool(torch.isfinite(prediction).all())}

        clean_test = evaluate()
        test_mask_path = resolve(task["test_mask"])
        if sha256_file(test_mask_path) != task["test_mask_sha256"]:
            raise RuntimeError("test mask SHA256 mismatch")
        test_mask = np.load(test_mask_path)["mask"]
        if tuple(test_mask.shape) != (len(test_data), 384, int(task["enc_in"])):
            raise RuntimeError(f"test mask shape mismatch: {test_mask.shape}")
        masked_test = evaluate(test_mask)
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1048576
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1048576
        passed = all((
            forward_finite, loss_finite, backward_finite, gradient_finite, nonzero_gradient,
            output_shape_correct, strict_reload, clean_test["finite"], masked_test["finite"],
            np.isfinite(best), np.isfinite(clean_test["mse"]), np.isfinite(masked_test["mse"]),
        ))
        metrics = {
            "stage": task["stage"], "substage": task["substage"], "dataset": dataset,
            "backbone": "patchtst", "patchtst_upstream_commit": PATCHTST_COMMIT,
            "experiment_seed": int(task["experiment_seed"]), "smoke_only": bool(task["smoke_only"]),
            "seq_len": 384, "pred_len": 96, "batch_size": int(task["batch_size"]),
            "train_batches": train_batches, "val_batches": val_batches,
            "test_batches": clean_test["batches"], "max_epochs": int(task["max_epochs"]),
            "epochs_completed": epoch + 1, "best_epoch": best_epoch,
            "best_validation_mse": best, "early_stopping_metric": "validation_mse",
            "test_used_for_model_selection": False, "forward_finite": forward_finite,
            "loss_finite": loss_finite, "backward_finite": backward_finite,
            "gradient_finite": gradient_finite, "nonzero_gradient": nonzero_gradient,
            "forecast_shape_correct": output_shape_correct, "checkpoint_save": True,
            "checkpoint_strict_reload": strict_reload, "evaluation_after_reload": True,
            "clean_test_mse": clean_test["mse"], "clean_test_mae": clean_test["mae"],
            "clean_test_rmse": clean_test["rmse"],
            "masked_random0.3_test_mse": masked_test["mse"],
            "masked_random0.3_test_mae": masked_test["mae"],
            "masked_random0.3_test_rmse": masked_test["rmse"],
            "input_mask_applied_for_descriptive_test": True,
            "test_mask_sha256": task["test_mask_sha256"], "test_mask_used_for_training": False,
            "batch_y_modified": False, "no_leakage": True,
            "training_protocol_hash": task["training_protocol_hash"],
            "metric_protocol_hash": task["metric_protocol_hash"],
            "peak_gpu_memory_mib": peak_allocated, "peak_gpu_reserved_mib": peak_reserved,
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "best_checkpoint_sha256": sha256_file(run / "checkpoints/best_backbone.pt"),
            "elapsed_seconds": time.time() - started,
        }
        status = "SUCCESS" if passed else "FAILED"
    except Exception as exc:
        error = repr(exc)
        metrics = {
            "stage": task.get("stage"), "substage": task.get("substage"),
            "dataset": task.get("dataset"), "backbone": "patchtst",
            "smoke_only": task.get("smoke_only"), "error": error,
            "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started,
        }
        log(metrics["traceback"])

    status_payload = {
        "status": status, "stage": task.get("stage"), "substage": task.get("substage"),
        "experiment_id": run.name, "backbone": "patchtst", "dataset": task.get("dataset"),
        "patchtst_upstream_commit": PATCHTST_COMMIT, "timeperceiver_upstream_commit": TIMEPERCEIVER_COMMIT,
        "experiment_seed": task.get("experiment_seed"), "smoke_only": task.get("smoke_only"),
        "best_epoch": metrics.get("best_epoch"), "best_validation_mse": metrics.get("best_validation_mse"),
        "test_used_for_model_selection": False,
        "best_checkpoint": str(run / "checkpoints/best_backbone.pt"),
        "best_checkpoint_sha256": metrics.get("best_checkpoint_sha256"),
        "checkpoint_strict_reload": metrics.get("checkpoint_strict_reload"),
        "no_leakage": metrics.get("no_leakage", False),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "error_type": classify_error(error) if error else "", "error_summary": error,
    }
    dump(run / "metrics.json", metrics)
    dump(run / "status.json", status_payload)
    dump(run / "resource_usage.json", {
        "peak_allocated_mib": metrics.get("peak_gpu_memory_mib"),
        "peak_reserved_mib": metrics.get("peak_gpu_reserved_mib"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "batch_size": task.get("batch_size"), "device": "cuda",
    })
    log(json.dumps({"status": status, "best_epoch": metrics.get("best_epoch"),
                    "best_validation_mse": metrics.get("best_validation_mse")}, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
