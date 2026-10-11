#!/usr/bin/env python3
"""Train capacity-matched PatchTST Gaussian/fixed-t8 heads for Gate 5b1."""
from __future__ import annotations

import argparse
import hashlib
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
from gate4b_train_probabilistic import Indexed, sample_masks, seed_worker  # noqa: E402
from gate5a_distribution_metrics import central_intervals, empirical_crps, interval_metrics, nll_elements  # noqa: E402
from gate5b_common import (  # noqa: E402
    DATASETS,
    PATCHTST_COMMIT,
    ROOT,
    TIMEPERCEIVER_COMMIT,
    dump,
    environment_text,
    probability_config,
    resolve,
    set_seed,
    sha256_file,
    upstream_diff_text,
    upstream_state_text,
    write_resolved,
)
from models.patchtst_backbone import PatchTSTBackbone  # noqa: E402
from models.probabilistic_patchtst import ProbabilisticPatchTST  # noqa: E402


VARIANTS = {"gaussian": "gaussian_head", "fixed_t8": "student_t_fixed_df8_head"}


def classify_error(error):
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


def atomic_torch_save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def head_state(model):
    prefixes = ("loc_projection.", "scale_projection.")
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if name.startswith(prefixes)}


def validate_task(task, allow_manifest_placeholder=False):
    errors = []
    expected = {
        "backbone": "patchtst", "seq_len": 384, "pred_len": 96,
        "missing_type": "random", "missing_rate": 0.3, "experiment_seed": 2025,
        "freeze_backbone": True, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "early_stopping_metric": "validation_nll",
        "test_used_for_model_selection": False, "learnable_df": False,
        "conformal": False, "posthoc_calibration": False,
    }
    for key, value in expected.items():
        if task.get(key) != value:
            errors.append(f"{key}={task.get(key)!r}, expected {value!r}")
    if task.get("dataset") not in DATASETS:
        errors.append(f"unsupported dataset {task.get('dataset')!r}")
    distribution = task.get("distribution")
    if distribution not in VARIANTS:
        errors.append(f"unsupported distribution {distribution!r}")
    if task.get("variant") != VARIANTS.get(distribution):
        errors.append("variant/distribution mismatch")
    if distribution == "fixed_t8" and float(task.get("fixed_t_df", -1)) != 8.0:
        errors.append("fixed-t df must equal 8.0")
    if distribution == "gaussian" and task.get("fixed_t_df") is not None:
        errors.append("Gaussian task must not define fixed_t_df")
    checkpoint = str(task.get("source_backbone_checkpoint", ""))
    checkpoint_hash = str(task.get("source_backbone_checkpoint_sha256", ""))
    placeholder = checkpoint == "__FROM_GATE5B0_MANIFEST__" and checkpoint_hash == "__FROM_GATE5B0_MANIFEST__"
    if not (allow_manifest_placeholder and placeholder):
        if not resolve(checkpoint).is_file():
            errors.append("source backbone checkpoint is missing")
        elif sha256_file(resolve(checkpoint)) != checkpoint_hash:
            errors.append("source backbone checkpoint hash mismatch")
    if bool(task.get("smoke_only")):
        expected_limits = {"max_epochs": 1, "max_train_batches": 4, "max_val_batches": 2, "max_test_batches": 2}
        for key, value in expected_limits.items():
            if int(task.get(key, -1)) != value:
                errors.append(f"smoke {key} must equal {value}")
    elif any(task.get(key) is not None for key in ("max_train_batches", "max_val_batches", "max_test_batches")):
        errors.append("formal probability task contains smoke batch limits")
    return errors, placeholder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--distribution", choices=sorted(VARIANTS))
    parser.add_argument("--run-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-manifest-placeholder", action="store_true")
    cli = parser.parse_args()
    if cli.task_file:
        task = json.loads(Path(cli.task_file).read_text())
    elif cli.dataset and cli.distribution:
        task = probability_config(cli.dataset, cli.distribution, smoke=cli.smoke)
    else:
        raise SystemExit("--task-file or --dataset/--distribution is required")
    errors, placeholder = validate_task(task, cli.allow_manifest_placeholder)
    if cli.validate_only:
        state = "PENDING_GATE5B0_MANIFEST" if placeholder and not errors else "PASS" if not errors else "FAIL"
        print(json.dumps({"status": state, "errors": errors, "task": task}, indent=2))
        return 0 if not errors else 2
    if errors:
        raise SystemExit("; ".join(errors))
    if placeholder:
        raise SystemExit("Gate 5b0 checkpoint manifest has not been materialized")
    if not cli.run_dir:
        raise SystemExit("--run-dir is required outside --validate-only")

    run = Path(cli.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    (run / "checkpoints").mkdir()
    (run / "command.sh").write_text(
        "#!/usr/bin/env bash\n" + " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]) + "\n"
    )
    write_resolved(run / "resolved_config.yaml", task)
    (run / "env.txt").write_text(environment_text())
    (run / "git_state.txt").write_text(upstream_state_text())
    (run / "code.diff").write_text(upstream_diff_text())
    log_path = run / "train.log"
    log_path.write_text("Gate 5b1 frozen-backbone probability-head training.\n")
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
        distribution = task["distribution"]
        variant = task["variant"]
        seed = int(task["experiment_seed"])
        set_seed(seed)
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
        train_loader = torch.utils.data.DataLoader(
            Indexed(train_data), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker,
            num_workers=0, drop_last=False, pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        validation_path = resolve(task["validation_mask"])
        test_path = resolve(task["test_mask"])
        if sha256_file(validation_path) != task["validation_mask_sha256"]:
            raise RuntimeError("validation mask SHA256 mismatch")
        validation_array = np.load(validation_path)["mask"]
        if tuple(validation_array.shape) != (len(val_data), 384, int(task["enc_in"])):
            raise RuntimeError("validation mask shape mismatch")
        checkpoint_path = resolve(task["source_backbone_checkpoint"])
        checkpoint_object = torch.load(checkpoint_path, map_location="cpu")
        backbone = PatchTSTBackbone(task).float()
        strict = backbone.load_state_dict(checkpoint_object["model_state_dict"], strict=True)
        if strict.missing_keys or strict.unexpected_keys:
            raise RuntimeError("strict backbone checkpoint load failed")
        model = ProbabilisticPatchTST(
            backbone, distribution, float(task["initial_scale"]),
            float(task["scale_floor"]), float(task.get("fixed_t_df") or 8.0),
        ).float().to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(task["learning_rate"]), weight_decay=float(task["weight_decay"]))
        backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
        optimizer_backbone = sum(id(parameter) in backbone_ids for group in optimizer.param_groups for parameter in group["params"])
        initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}

        best = float("inf")
        best_epoch = 0
        bad_epochs = 0
        train_batches = 0
        finite_train = True
        finite_gradient = True
        loc_gradient = False
        scale_gradient = False
        backbone_gradients = 0
        batch_y_unchanged = True
        input_mask_applied = False
        deterministic_masks = True
        train_missing = 0
        train_total = 0
        first_indices_hash = None
        for epoch in range(int(task["max_epochs"])):
            model.train()
            epoch_losses = []
            for batch_index, (batch_x, batch_y, _, _, window_indices) in enumerate(train_loader):
                if task.get("max_train_batches") is not None and batch_index >= int(task["max_train_batches"]):
                    break
                mask_cpu = sample_masks(
                    window_indices, epoch, batch_x.shape, dataset, "random", 0.3,
                    int(task["train_mask_base_seed"]), seed, task["train_mask_seed_rule"],
                )
                if train_batches == 0:
                    repeated = sample_masks(
                        window_indices, epoch, batch_x.shape, dataset, "random", 0.3,
                        int(task["train_mask_base_seed"]), seed, task["train_mask_seed_rule"],
                    )
                    deterministic_masks = bool(torch.equal(mask_cpu, repeated))
                    first_indices_hash = hashlib.sha256(np.asarray(window_indices, dtype=np.int64).tobytes()).hexdigest()
                train_missing += int((mask_cpu == 0).sum())
                train_total += mask_cpu.numel()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                mask = mask_cpu.to(device)
                target = batch_y[:, -args.pred_len:, :]
                before_y = batch_y.clone()
                masked_x = batch_x * mask
                input_mask_applied |= bool(torch.any(mask == 0) and torch.any(masked_x != batch_x))
                optimizer.zero_grad(set_to_none=True)
                output = model(masked_x)
                loss = nll_elements(variant, output["loc"], output["scale"], target, output["df"]).mean()
                finite_train &= bool(torch.isfinite(loss))
                loss.backward()
                batch_y_unchanged &= bool(torch.equal(before_y, batch_y))
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        finite_gradient &= bool(torch.isfinite(parameter.grad).all())
                        nonzero = bool(torch.count_nonzero(parameter.grad))
                        loc_gradient |= nonzero and "loc_projection" in name
                        scale_gradient |= nonzero and "scale_projection" in name
                    elif not parameter.requires_grad and parameter.grad is not None:
                        backbone_gradients += 1
                clip_grad_norm_(trainable, float(task["gradient_clip"]))
                optimizer.step()
                train_batches += 1
                epoch_losses.append(float(loss.detach()))
            model.eval()
            val_losses = []
            offset = 0
            with torch.no_grad():
                for batch_index, (batch_x, batch_y, _, _) in enumerate(val_loader):
                    if task.get("max_val_batches") is not None and batch_index >= int(task["max_val_batches"]):
                        break
                    batch_x = batch_x.float().to(device)
                    target = batch_y[:, -args.pred_len:, :].float().to(device)
                    mask = torch.from_numpy(validation_array[offset:offset + len(batch_x)]).float().to(device)
                    offset += len(batch_x)
                    output = model(batch_x * mask)
                    val_losses.append(float(nll_elements(
                        variant, output["loc"], output["scale"], target, output["df"]
                    ).mean()))
            validation_nll = float(np.mean(val_losses))
            if not np.isfinite(validation_nll):
                raise RuntimeError("non-finite validation NLL")
            if validation_nll < best:
                best = validation_nll
                best_epoch = epoch + 1
                bad_epochs = 0
                atomic_torch_save({
                    "head_state_dict": head_state(model), "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": best_epoch, "best_validation_nll": best, "distribution": distribution,
                    "source_backbone_checkpoint_sha256": task["source_backbone_checkpoint_sha256"],
                    "training_protocol_hash": task["training_protocol_hash"],
                    "metric_protocol_hash": task["metric_protocol_hash"],
                }, run / "checkpoints/best_probability_head.pt")
            else:
                bad_epochs += 1
            log(f"epoch={epoch + 1} train_nll={np.mean(epoch_losses):.8f} validation_nll={validation_nll:.8f} best={best:.8f}")
            if bad_epochs >= int(task["patience"]):
                break
        atomic_torch_save({
            "head_state_dict": head_state(model), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1, "distribution": distribution,
            "source_backbone_checkpoint_sha256": task["source_backbone_checkpoint_sha256"],
        }, run / "checkpoints/last_probability_head.pt")
        updated = [name for name, parameter in model.named_parameters()
                   if parameter.requires_grad and not torch.equal(initial[name], parameter.detach().cpu())]

        reload_backbone = PatchTSTBackbone(task).float()
        reload_backbone.load_state_dict(checkpoint_object["model_state_dict"], strict=True)
        reload_model = ProbabilisticPatchTST(
            reload_backbone, distribution, float(task["initial_scale"]),
            float(task["scale_floor"]), float(task.get("fixed_t_df") or 8.0),
        ).float()
        head_object = torch.load(run / "checkpoints/best_probability_head.pt", map_location="cpu")
        load_result = reload_model.load_state_dict(head_object["head_state_dict"], strict=False)
        expected_head = set(head_state(reload_model))
        reload_ok = (
            set(head_object["head_state_dict"]) == expected_head and not load_result.unexpected_keys
            and all(name not in expected_head for name in load_result.missing_keys)
        )
        reload_model.to(device).eval()
        test_array = np.load(test_path)["mask"]
        if sha256_file(test_path) != task["test_mask_sha256"]:
            raise RuntimeError("test mask SHA256 mismatch")
        if tuple(test_array.shape) != (len(test_data), 384, int(task["enc_in"])):
            raise RuntimeError("test mask shape mismatch")
        locations, scales, targets, dfs = [], [], [], []
        offset = 0
        test_batches = 0
        with torch.no_grad():
            for batch_index, (batch_x, batch_y, _, _) in enumerate(test_loader):
                if task.get("max_test_batches") is not None and batch_index >= int(task["max_test_batches"]):
                    break
                batch_x = batch_x.float().to(device)
                target = batch_y[:, -args.pred_len:, :].float().to(device)
                mask = torch.from_numpy(test_array[offset:offset + len(batch_x)]).float().to(device)
                offset += len(batch_x)
                output = reload_model(batch_x * mask)
                locations.append(output["loc"].cpu())
                scales.append(output["scale"].cpu())
                targets.append(target.cpu())
                if output["df"] is not None:
                    dfs.append(output["df"].cpu())
                test_batches += 1
        location = torch.cat(locations)
        scale = torch.cat(scales)
        target = torch.cat(targets)
        df = torch.cat(dfs) if dfs else None
        nll = nll_elements(variant, location, scale, target, df)
        difference = location - target
        mse = float((difference * difference).mean())
        mae = float(difference.abs().mean())
        intervals = central_intervals(variant, location, scale, df)
        interval_summary = interval_metrics(target, intervals)
        crps = empirical_crps(variant, location, scale, target, df, 64, 52025, 65536)
        fixed_df_exact = distribution != "fixed_t8" or bool(torch.equal(df, torch.full_like(df, 8.0)))
        passed = all((
            finite_train, finite_gradient, loc_gradient, scale_gradient, backbone_gradients == 0,
            optimizer_backbone == 0, bool(updated), batch_y_unchanged, input_mask_applied,
            deterministic_masks, reload_ok, bool(torch.isfinite(scale).all()), bool((scale > 0).all()),
            fixed_df_exact, np.isfinite(best), np.isfinite(float(nll.mean())), np.isfinite(crps),
        ))
        metrics = {
            "stage": task["stage"], "substage": task["substage"], "dataset": dataset,
            "backbone": "patchtst", "patchtst_upstream_commit": PATCHTST_COMMIT,
            "variant": variant, "distribution": distribution, "fixed_df": 8.0 if distribution == "fixed_t8" else None,
            "learnable_df": False, "experiment_seed": seed, "smoke_only": bool(task["smoke_only"]),
            "train_batches": train_batches, "val_batches": len(val_losses), "test_batches": test_batches,
            "epochs_completed": epoch + 1, "best_epoch": best_epoch, "best_validation_nll": best,
            "early_stopping_metric": "validation_nll", "test_used_for_model_selection": False,
            "train_nll_finite": finite_train, "validation_nll_finite": bool(np.isfinite(best)),
            "loc_nonzero_gradient": loc_gradient, "scale_nonzero_gradient": scale_gradient,
            "all_new_gradients_finite": finite_gradient,
            "backbone_non_null_gradient_count": backbone_gradients,
            "optimizer_backbone_parameter_count": optimizer_backbone,
            "trainable_parameter_names": trainable_names, "updated_parameter_names": updated,
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
            "fixed_df_parameter_count": model.df_parameter_count(), "fixed_df_exact": fixed_df_exact,
            "input_mask_applied": input_mask_applied, "mask_application_verified": input_mask_applied,
            "actual_train_missing_rate": train_missing / train_total,
            "actual_validation_missing_rate": float(1.0 - validation_array.mean()),
            "actual_test_missing_rate": float(1.0 - test_array.mean()),
            "first_batch_window_indices_sha256": first_indices_hash,
            "train_mask_deterministic": deterministic_masks, "batch_y_modified": not batch_y_unchanged,
            "test_mask_used_during_training": False, "no_leakage": True,
            "checkpoint_save": True, "checkpoint_strict_reload": reload_ok,
            "evaluation_after_reload": True,
            "source_backbone_checkpoint": task["source_backbone_checkpoint"],
            "source_backbone_checkpoint_sha256": task["source_backbone_checkpoint_sha256"],
            "validation_mask": task["validation_mask"], "validation_mask_sha256": task["validation_mask_sha256"],
            "test_mask": task["test_mask"], "test_mask_sha256": task["test_mask_sha256"],
            "training_protocol_hash": task["training_protocol_hash"], "metric_protocol_hash": task["metric_protocol_hash"],
            "test_mse": mse, "test_mae": mae, "test_rmse": math.sqrt(mse),
            "test_mean_nll": float(nll.mean()), "test_median_element_nll": float(nll.median()),
            "test_p90_element_nll": float(np.quantile(nll.numpy(), 0.9)),
            "test_p99_element_nll": float(np.quantile(nll.numpy(), 0.99)), "test_crps": crps,
            **interval_summary, "mean_scale": float(scale.mean()), "median_scale": float(scale.median()),
            "p90_scale": float(np.quantile(scale.numpy(), 0.9)), "scale_min": float(scale.min()),
            "scale_max": float(scale.max()), "scale_finite": bool(torch.isfinite(scale).all()),
            "scale_positive": bool((scale > 0).all()), "crps_samples": 64, "crps_seed": 52025,
            "conformal_calibrated": False, "posthoc_calibration": False,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1048576,
            "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 1048576,
            "elapsed_seconds": time.time() - started,
        }
        status = "SUCCESS" if passed else "FAILED"
    except Exception as exc:
        error = repr(exc)
        metrics = {
            "stage": task.get("stage"), "substage": task.get("substage"),
            "dataset": task.get("dataset"), "variant": task.get("variant"),
            "distribution": task.get("distribution"), "smoke_only": task.get("smoke_only"),
            "error": error, "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started,
        }
        log(metrics["traceback"])

    status_payload = {
        "status": status, "stage": "gate5b", "substage": "probability_replication",
        "experiment_id": run.name, "backbone": "patchtst", "patchtst_upstream_commit": PATCHTST_COMMIT,
        "timeperceiver_upstream_commit": TIMEPERCEIVER_COMMIT, "dataset": task.get("dataset"),
        "variant": task.get("variant"), "distribution": task.get("distribution"),
        "experiment_seed": task.get("experiment_seed"),
        "source_backbone_checkpoint": task.get("source_backbone_checkpoint"),
        "source_backbone_checkpoint_sha256": task.get("source_backbone_checkpoint_sha256"),
        "validation_mask": task.get("validation_mask"), "validation_mask_sha256": task.get("validation_mask_sha256"),
        "test_mask": task.get("test_mask"), "test_mask_sha256": task.get("test_mask_sha256"),
        "input_mask_applied": metrics.get("input_mask_applied"),
        "mask_application_verified": metrics.get("mask_application_verified"),
        "batch_y_modified": metrics.get("batch_y_modified"), "test_mask_used_during_training": False,
        "training_protocol_hash": task.get("training_protocol_hash"),
        "metric_protocol_hash": task.get("metric_protocol_hash"), "freeze_backbone": True,
        "optimizer_backbone_parameter_count": metrics.get("optimizer_backbone_parameter_count"),
        "backbone_non_null_gradient_count": metrics.get("backbone_non_null_gradient_count"),
        "trainable_parameter_count": metrics.get("trainable_parameter_count"),
        "fixed_df": metrics.get("fixed_df"), "learnable_df": False,
        "max_epochs": task.get("max_epochs"), "patience": task.get("patience"),
        "best_epoch": metrics.get("best_epoch"), "best_validation_nll": metrics.get("best_validation_nll"),
        "test_used_for_model_selection": False, "test_mse": metrics.get("test_mse"),
        "test_mae": metrics.get("test_mae"), "test_rmse": metrics.get("test_rmse"),
        "test_mean_nll": metrics.get("test_mean_nll"), "test_crps": metrics.get("test_crps"),
        "mean_absolute_coverage_error": metrics.get("mean_absolute_coverage_error"),
        "mean_interval_width": metrics.get("mean_interval_width"),
        "mean_interval_score": metrics.get("mean_interval_score"),
        "scale_min": metrics.get("scale_min"), "scale_max": metrics.get("scale_max"),
        "scale_finite": metrics.get("scale_finite"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "error_type": classify_error(error) if error else "", "error_summary": error,
    }
    dump(run / "metrics.json", metrics)
    dump(run / "status.json", status_payload)
    dump(run / "resource_usage.json", {
        "peak_allocated_mib": metrics.get("peak_gpu_memory_mib"),
        "peak_reserved_mib": metrics.get("peak_gpu_reserved_mib"),
        "elapsed_seconds": metrics.get("elapsed_seconds"), "batch_size": task.get("batch_size"),
        "device": "cuda",
    })
    log(json.dumps({"status": status, "best_epoch": metrics.get("best_epoch"),
                    "best_validation_nll": metrics.get("best_validation_nll")}, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
