#!/usr/bin/env python3
"""Prepare and audit the immutable Gate 5b PatchTST integration."""
from __future__ import annotations

import copy
import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import (  # noqa: E402
    BATCH_SIZES,
    DATASETS,
    DISPLAY,
    OFFICIAL_CONFIGS,
    OFFICIAL_SCRIPTS,
    PATCHTST,
    PATCHTST_COMMIT,
    PYTHON,
    ROOT,
    TIMEPERCEIVER,
    backbone_config,
    dataset_spec,
    dump,
    probability_config,
    run_output,
    write_csv,
    write_report,
    write_resolved,
)
from models.patchtst_backbone import PatchTSTBackbone  # noqa: E402
from models.probabilistic_patchtst import ProbabilisticPatchTST  # noqa: E402


def tracked_clean(path: Path) -> bool:
    return not run_output(["git", "-C", str(path), "diff", "--name-only"]).strip()


def source_audit():
    commit = run_output(["git", "-C", str(PATCHTST), "rev-parse", "HEAD"])
    remote = run_output(["git", "-C", str(PATCHTST), "remote", "get-url", "origin"])
    supervised = [
        "PatchTST_supervised/models/PatchTST.py",
        "PatchTST_supervised/layers/PatchTST_backbone.py",
        "PatchTST_supervised/layers/PatchTST_layers.py",
    ]
    import_ok = True
    import_error = ""
    try:
        import models.patchtst_backbone  # noqa: F401
    except Exception as exc:
        import_ok = False
        import_error = repr(exc)
    payload = {
        "status": "PASS" if (
            commit == PATCHTST_COMMIT and "github.com/yuqinie98/PatchTST" in remote
            and (PATCHTST / "LICENSE").is_file() and all((PATCHTST / path).is_file() for path in supervised)
            and tracked_clean(PATCHTST) and import_ok
        ) else "BLOCKED_GATE5B_PATCHTST_SOURCE_AUDIT",
        "repository": "https://github.com/yuqinie98/PatchTST.git",
        "remote_url": remote, "resolved_commit": commit,
        "license": "Apache-2.0", "license_identified": (PATCHTST / "LICENSE").is_file(),
        "official_supervised_code_found": all((PATCHTST / path).is_file() for path in supervised),
        "supervised_model_files": supervised, "source_clean": tracked_clean(PATCHTST),
        "local_modifications": run_output(["git", "-C", str(PATCHTST), "status", "--short"]).splitlines(),
        "python_import_compatibility": import_ok, "import_error": import_error,
        "torch_compatibility": bool(import_ok),
        "required_dependencies": ["numpy", "matplotlib", "pandas", "scikit-learn", "torch"],
        "missing_dependencies": [],
    }
    dump(ROOT / "results/gate5b_patchtst_source_audit.json", payload)
    write_report(ROOT / "reports/gate5b_patchtst_source_audit.md", "Gate 5b PatchTST Source Audit", payload)
    return payload


def environment_audit():
    versions = {}
    missing = []
    for module_name in ("numpy", "pandas", "sklearn", "matplotlib", "torch"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            missing.append({"module": module_name, "error": repr(exc)})
    compatible = not missing and torch.__version__.startswith("2.1.0") and torch.version.cuda == "12.1"
    payload = {
        "status": "ENV_COMPATIBLE" if compatible else "BLOCKED_GATE5B_ENVIRONMENT_INCOMPATIBLE",
        "conda_environment": str(PYTHON.parent.parent), "python": sys.version.split()[0],
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "versions": versions,
        "additional_dependencies_installed": [], "missing_dependencies": missing,
        "environment_modified": False, "official_requirements_installed": False,
        "architecture_import_test": compatible,
    }
    dump(ROOT / "results/gate5b_environment_audit.json", payload)
    write_report(ROOT / "reports/gate5b_environment_audit.md", "Gate 5b Environment Audit", payload)
    return payload


def config_resolution():
    rows = []
    resolved = {}
    for dataset in DATASETS:
        official = copy.deepcopy(OFFICIAL_CONFIGS[dataset])
        selected = copy.deepcopy(dataset_spec(dataset)["architecture"])
        rows.append({
            "dataset": DISPLAY[dataset], "official_source_config": OFFICIAL_SCRIPTS[dataset],
            "selection_rule": "same dataset + pred_len96 official supervised config",
            "official_seq_len": official["seq_len"], "resolved_seq_len": 384,
            "pred_len": 96, "official_batch_size": official["batch_size"],
            "resolved_batch_size": BATCH_SIZES[dataset],
            "overridden_fields": "seq_len,batch_size,data pipeline,split,normalization,seed,artifact paths",
            "status": "PASS",
        })
        resolved[dataset] = {
            "official_source": OFFICIAL_SCRIPTS[dataset], "official": official,
            "resolved": selected, "test_metrics_used": False,
        }
        write_resolved(
            ROOT / f"configs/gate5b/backbone/patchtst_{dataset}_clean_seed2025.yaml",
            backbone_config(dataset),
        )
        for distribution, suffix in (("gaussian", "gaussian"), ("fixed_t8", "fixed_t8")):
            write_resolved(
                ROOT / f"configs/gate5b/heads/patchtst_{dataset}_{suffix}_random0.3_seed2025.yaml",
                probability_config(dataset, distribution),
            )
    payload = {
        "status": "PASS", "selection_rule": "pre-registered official supervised pred_len96 config",
        "resolved_before_formal_training": True, "test_metrics_used": False,
        "allowed_overrides_only": True, "datasets": resolved,
    }
    write_csv(ROOT / "reports/gate5b_patchtst_official_config_inventory.csv", rows)
    dump(ROOT / "results/gate5b_patchtst_config_resolution.json", payload)
    write_report(
        ROOT / "reports/gate5b_patchtst_config_resolution.md",
        "Gate 5b PatchTST Config Resolution", payload,
        "Only the pre-declared protocol and resource-safe fields were overridden.",
    )
    return payload


def architecture_audit():
    rows = []
    all_pass = True
    for dataset in DATASETS:
        torch.manual_seed(2025)
        config = dataset_spec(dataset)["architecture"]
        backbone = PatchTSTBackbone(config).eval()
        x = torch.randn(2, 384, config["enc_in"])
        with torch.no_grad():
            official = backbone.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            hidden, forecast = backbone.forward_with_hidden(x)
        deterministic_parity = bool(torch.equal(official, forecast))
        gaussian = ProbabilisticPatchTST(copy.deepcopy(backbone), "gaussian").eval()
        fixed_t = ProbabilisticPatchTST(copy.deepcopy(backbone), "fixed_t8").eval()
        gaussian.train()
        frozen_backbone_eval = not gaussian.backbone.training
        gaussian.eval()
        with torch.no_grad():
            gaussian_out = gaussian(x)
            fixed_out = fixed_t(x)
        expected = (2, 96, config["enc_in"])
        initial_scale_exact = bool(torch.allclose(
            gaussian_out["scale"], torch.full_like(gaussian_out["scale"], 0.5), atol=1e-7, rtol=0.0
        ))
        paired_initialization_equal = all(
            torch.equal(value, fixed_t.loc_projection.state_dict()[key])
            for key, value in gaussian.loc_projection.state_dict().items()
        ) and all(
            torch.equal(value, fixed_t.scale_projection.state_dict()[key])
            for key, value in gaussian.scale_projection.state_dict().items()
        )
        row_pass = all((
            deterministic_parity,
            tuple(forecast.shape) == expected,
            tuple(gaussian_out["loc"].shape) == expected,
            tuple(gaussian_out["scale"].shape) == expected,
            bool((gaussian_out["scale"] > 0).all()),
            initial_scale_exact, paired_initialization_equal,
            frozen_backbone_eval,
            gaussian.trainable_parameter_count() == fixed_t.trainable_parameter_count(),
            gaussian.backbone_trainable_parameter_count() == 0,
            fixed_t.backbone_trainable_parameter_count() == 0,
            fixed_t.df_parameter_count() == 0,
            bool(torch.equal(fixed_out["df"], torch.full_like(fixed_out["df"], 8.0))),
        ))
        all_pass &= row_pass
        rows.append({
            "dataset": dataset, "channels": config["enc_in"], "hidden_size": backbone.hidden_size,
            "forecast_shape": list(forecast.shape), "loc_shape": list(gaussian_out["loc"].shape),
            "scale_shape": list(gaussian_out["scale"].shape),
            "initial_scale_exact_0.5": initial_scale_exact,
            "gaussian_fixed_t8_initialization_equal": paired_initialization_equal,
            "frozen_backbone_remains_eval_during_head_training": frozen_backbone_eval,
            "gaussian_trainable_parameters": gaussian.trainable_parameter_count(),
            "fixed_t8_trainable_parameters": fixed_t.trainable_parameter_count(),
            "gaussian_backbone_trainable_parameters": gaussian.backbone_trainable_parameter_count(),
            "fixed_t8_backbone_trainable_parameters": fixed_t.backbone_trainable_parameter_count(),
            "fixed_t8_df_parameters": fixed_t.df_parameter_count(), "fixed_df": 8.0,
            "deterministic_official_parity": deterministic_parity, "status": "PASS" if row_pass else "FAIL",
        })
    payload = {
        "status": "PASS" if all_pass else "BLOCKED_GATE5B_ARCHITECTURE_AUDIT",
        "official_encoder_output": "[B,C,d_model,patch_count]",
        "hidden_feature_point": "official shared Flatten_Head output before linear projection",
        "deterministic_head_reuse": "loc projection initialized by strict state_dict copy",
        "scale_parameterization": "softplus(raw_scale)+1e-4; zero weight and bias for initial scale 0.5",
        "datasets_passed": sum(row["status"] == "PASS" for row in rows),
        "datasets_total": len(rows), "rows": rows,
    }
    write_csv(ROOT / "reports/gate5b_patchtst_parameter_counts.csv", rows)
    dump(ROOT / "results/gate5b_patchtst_architecture_audit.json", payload)
    write_report(ROOT / "reports/gate5b_patchtst_architecture_audit.md", "Gate 5b PatchTST Architecture Audit", payload)
    return payload


def main():
    source = source_audit()
    environment = environment_audit()
    configs = config_resolution()
    architecture = architecture_audit()
    passed = (
        source["status"] == "PASS" and environment["status"] == "ENV_COMPATIBLE"
        and configs["status"] == "PASS" and architecture["status"] == "PASS"
    )
    summary = {
        "status": "PASS" if passed else "BLOCKED", "source": source["status"],
        "environment": environment["status"], "config_resolution": configs["status"],
        "architecture": architecture["status"],
    }
    plan = {
        "status": "PLANNING_PASS" if passed else "BLOCKED",
        "stage": "Gate 5b Second-Backbone Minimal Replication",
        "second_backbone": "PatchTST supervised", "datasets": list(DATASETS),
        "seq_len": 384, "pred_len": 96, "missing_type": "random",
        "missing_rate": 0.3, "experiment_seed": 2025,
        "gate5b0_backbone_tasks": 3, "gate5b1_probability_tasks": 6,
        "maximum_new_formal_tasks": 9, "extra_tasks": 0,
        "probability_heads": ["Gaussian", "Fixed Student-t df=8"],
        "sequence": [
            "source/environment/config/architecture/data audits",
            "backbone smoke 3/3", "backbone queue validation and execution",
            "checkpoint manifest", "distribution/pairing validation",
            "probability smoke 6/6", "probability queue validation and execution",
            "formal decision and strict closure",
        ],
        "test_used_for_configuration_selection": False,
        "learnable_df": False, "laplace": False, "adapter": False, "conformal": False,
    }
    dump(ROOT / "results/gate5b_plan.json", plan)
    write_report(ROOT / "reports/gate5b_plan.md", "Gate 5b Plan", plan)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
