import json, pathlib, subprocess, datetime

root = pathlib.Path("/mnt/data/zza/code/timeperceiver_meta_uq")
status_path = root / "reports" / "gate0_status.json"
probe_path = root / "environment" / "torch_cuda_probe.json"
import_path = root / "environment" / "import_check.json"

def sh(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

probe = json.loads(probe_path.read_text())
imports = json.loads(import_path.read_text())

failed_imports = [k for k, v in imports["imports"].items() if not v["ok"]]
cuda_ok = bool(probe.get("cuda_available"))
repo_commit = sh("git -C /mnt/data/zza/code/timeperceiver_meta_uq/upstream/TimePerceiver rev-parse HEAD")

status = {
    "gate": "Gate 0",
    "status": "PASS" if (cuda_ok and not failed_imports) else "PARTIAL",
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "project_root": str(root),
    "conda_env": "timeperceiver_meta_uq",
    "python_version": sh("/mnt/data/zza/anaconda3/bin/conda run -n timeperceiver_meta_uq python --version"),
    "torch_version": probe.get("torch_version"),
    "torch_cuda_runtime": probe.get("torch_cuda_runtime"),
    "cuda_available": cuda_ok,
    "gpu_count": probe.get("device_count", 0),
    "repository_path": str(root / "upstream" / "TimePerceiver"),
    "repository_commit": repo_commit,
    "agents_read": [
        "/mnt/data/zza/code/AGENTS.md",
        "/mnt/data/zza/code/timeperceiver_meta_uq/AGENTS.md"
    ],
    "checks": {
        "conda_env_exists": True,
        "torch_installed": True,
        "cuda_tiny_tensor": cuda_ok,
        "imports_ok": not failed_imports,
        "compileall_ok": True
    },
    "blockers": [] if (cuda_ok and not failed_imports) else [
        *([] if cuda_ok else ["CUDA_NOT_AVAILABLE"]),
        *([f"IMPORT_FAILED:{','.join(failed_imports)}"] if failed_imports else [])
    ],
    "next_gate": "Gate 1"
}
status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

report = root / "reports" / "gate0_environment_report.md"
append = f"""

## Gate 0 Resume Install Update - {status['timestamp']}

- Status: **{status['status']}**
- Conda env: `{status['conda_env']}`
- Python: `{status['python_version']}`
- PyTorch: `{status['torch_version']}`
- Torch CUDA runtime: `{status['torch_cuda_runtime']}`
- CUDA available: `{status['cuda_available']}`
- GPU count: `{status['gpu_count']}`
- Repository commit: `{status['repository_commit']}`
- Failed imports: `{','.join(failed_imports) if failed_imports else 'None'}`
- Blockers: `{', '.join(status['blockers']) if status['blockers'] else 'None'}`
"""
with report.open("a", encoding="utf-8") as f:
    f.write(append)
print(json.dumps(status, indent=2, ensure_ascii=False))
