# Gate 0 Environment Report

## 1. Stage Goal

Initialize the TimePerceiver Meta-UQ project environment without starting Gate 1, downloading full datasets, or running formal training.

## 2. Executed Content

- Read AGENTS files and checked for AGENTS.override.md.
- Created minimal project directories.
- Audited server, scheduler commands, GPU inventory, TimePerceiver upstream repository, Conda environment, dependencies, imports, syntax, and CUDA tensor availability.

## 3. Server Summary

See /mnt/data/zza/code/timeperceiver_meta_uq/logs/gate0/gate0__environment_init__20260630_202933/system_inventory.log.

## 4. GPU Mapping

Inventory: environment/gpu_inventory.txt

Mapping: environment/gpu_mapping.json

## 5. Conda Environment

- Environment: timeperceiver_meta_uq
- Python: 3.9.25

## 6. PyTorch/CUDA Compatibility

- PyTorch: UNKNOWN
- CUDA runtime: UNKNOWN
- CUDA available: false
- GPU count: 0
- Minimal tensor test: false

## 7. Official Repository

- URL: https://github.com/efficient-learning-lab/TimePerceiver.git
- Path: /mnt/data/zza/code/timeperceiver_meta_uq/upstream/TimePerceiver
- Commit: 7e30cc07b51c709f408409fd60a34c81ae8990be

## 8. Dependency Installation

See /mnt/data/zza/code/timeperceiver_meta_uq/logs/gate0/gate0__environment_init__20260630_202933/dependency_install.log and environment snapshots under environment/.

## 9. Validation Commands And Exit Codes

```text
pwd	0
date	0
hostname	0
whoami	0
uname	0
os_release	0
df_h	0
free_h	0
lscpu	0
git_version	0
gcc_version	0
gpp_version	0
conda_version	0
python_version_system	NOT_AVAILABLE
pip_version_system	NOT_AVAILABLE
nvcc_version	0
nvidia_smi	0
gpu_query	0
nvidia_smi_pmon	0
nvidia_smi_topo	0
git_clone_timeperceiver	0
conda_create_env	0
env_python_version	0
pip_install_torch	2
pip_install_requirements_without_torch	1
python_version_snapshot	0
import_validation	0
compileall_upstream	0
run_py_help	1
torch_cuda_probe	2
```

## 10. Gate Decision

PARTIAL

## 11. Unresolved Issues

- BLOCKER: Failed to install torch==2.1.0 CUDA 12.1 wheel.
- BLOCKER: Torch/CUDA probe failed to import torch.
- BLOCKER: PyTorch is not importable in selected environment.
- BLOCKER: torch.cuda.is_available() is not true.
- BLOCKER: Minimal CUDA tensor test did not pass.
- WARNING: Installing upstream non-torch requirements returned non-zero; import validation will decide Gate 0 status.
- WARNING: run.py --help returned non-zero; official entry may not support --help.

## 12. Shortest Path To Gate 1

After Gate 0 is PASS, proceed to Gate 1 data-loading smoke tests for ETTh1, Weather, and Electricity without starting formal training first.


## Gate 0 Resume Install Update - 2026-07-01T12:51:11

- Status: **PASS**
- Conda env: `timeperceiver_meta_uq`
- Python: `Python 3.9.25`
- PyTorch: `2.1.0+cu121`
- Torch CUDA runtime: `12.1`
- CUDA available: `True`
- GPU count: `2`
- Repository commit: `7e30cc07b51c709f408409fd60a34c81ae8990be`
- Failed imports: `None`
- Blockers: `None`
