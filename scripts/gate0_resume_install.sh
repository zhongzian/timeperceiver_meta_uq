#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
ENV_NAME="timeperceiver_meta_uq"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/gate0/${TS}_resume_install"
LOG_FILE="$LOG_DIR/gate0_resume_install.log"
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/reports" "$PROJECT_ROOT/environment"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Gate 0 resume install started at $(date) ====="
cd "$PROJECT_ROOT"

echo "===== Read AGENTS files ====="
for f in /mnt/data/zza/code/AGENTS.md "$PROJECT_ROOT/AGENTS.md"; do
  if [ -f "$f" ]; then
    echo "FOUND_AGENTS: $f"
    head -n 5 "$f" || true
  else
    echo "MISSING_AGENTS: $f"
  fi
done

echo "===== Basic state ====="
pwd
date
hostname
whoami
echo "ENV_NAME=$ENV_NAME"

echo "===== Conda env check ====="
/mnt/data/zza/anaconda3/bin/conda env list | tee "$LOG_DIR/conda_env_list.txt"
if ! /mnt/data/zza/anaconda3/bin/conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ERROR: conda env $ENV_NAME not found"
  exit 20
fi

echo "===== Python check ====="
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python --version
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python -m pip --version

echo "===== Repo check ====="
if [ -d upstream/TimePerceiver/.git ]; then
  git -C upstream/TimePerceiver rev-parse HEAD | tee "$LOG_DIR/upstream_commit.txt"
  git -C upstream/TimePerceiver status --short | tee "$LOG_DIR/upstream_status.txt"
else
  echo "ERROR: upstream/TimePerceiver is not a git repo"
  exit 21
fi

echo "===== GPU snapshot before install ====="
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader | tee "$LOG_DIR/gpu_snapshot_before.txt" || true

echo "===== Install torch cu121 ====="
# Official PyTorch cu121 index. Do not install or modify system CUDA.
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python -m pip install \
  --timeout 600 \
  --retries 20 \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.0

echo "===== Verify torch immediately ====="
cat > "$LOG_DIR/verify_torch.py" <<'PY'
import json, torch
out = {
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append({
            "logical_index": i,
            "name": p.name,
            "total_memory": p.total_memory,
            "capability": f"{p.major}.{p.minor}",
        })
    out["devices"] = devices
    x = torch.tensor([1.0], device="cuda")
    y = x * 2
    torch.cuda.synchronize()
    out["tiny_cuda_result"] = float(y.item())
print(json.dumps(out, indent=2, ensure_ascii=False))
if not out["cuda_available"]:
    raise SystemExit("torch installed but CUDA is not available")
PY
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python "$LOG_DIR/verify_torch.py"

echo "===== Install non-torch requirements ====="
if [ ! -f environment/requirements_without_torch.txt ]; then
  echo "ERROR: environment/requirements_without_torch.txt not found"
  exit 22
fi

/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python -m pip install \
  --timeout 300 \
  --retries 20 \
  -r environment/requirements_without_torch.txt

echo "===== Import check ====="
cat > "$LOG_DIR/import_check.py" <<'PY'
import json, importlib, sys
mods = [
    "torch", "numpy", "pandas", "scipy", "sklearn", "einops",
    "matplotlib", "sktime", "sympy", "pywt",
    "reformer_pytorch", "local_attention"
]
out = {"python": sys.version, "imports": {}}
for m in mods:
    try:
        mod = importlib.import_module(m)
        out["imports"][m] = {"ok": True, "version": getattr(mod, "__version__", "NA")}
    except Exception as e:
        out["imports"][m] = {"ok": False, "error": repr(e)}
import torch
out["torch"] = {
    "version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
print(json.dumps(out, indent=2, ensure_ascii=False))
failed = [k for k, v in out["imports"].items() if not v["ok"]]
if failed:
    raise SystemExit("IMPORT_FAILED: " + ",".join(failed))
if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE")
PY
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python "$LOG_DIR/import_check.py" | tee environment/import_check.json

echo "===== compileall TimePerceiver ====="
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python -m compileall upstream/TimePerceiver | tee "$LOG_DIR/compileall.log"

echo "===== run.py help behavior ====="
set +e
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python upstream/TimePerceiver/run.py --help > "$LOG_DIR/run_help.stdout" 2> "$LOG_DIR/run_help.stderr"
HELP_EXIT=$?
set -e
echo "$HELP_EXIT" | tee "$LOG_DIR/run_help_exit_code.txt"
echo "run.py --help exit code: $HELP_EXIT"
echo "--- run.py help stdout tail ---"
tail -n 30 "$LOG_DIR/run_help.stdout" || true
echo "--- run.py help stderr tail ---"
tail -n 30 "$LOG_DIR/run_help.stderr" || true

echo "===== Save environment snapshots ====="
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python --version | tee environment/python_version.txt
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python -m pip freeze | tee environment/pip_freeze.txt
/mnt/data/zza/anaconda3/bin/conda list -n "$ENV_NAME" | tee environment/conda_list.txt
/mnt/data/zza/anaconda3/bin/conda list -n "$ENV_NAME" --explicit | tee environment/conda_explicit.txt

echo "===== Torch CUDA probe file ====="
cat > "$LOG_DIR/torch_cuda_probe_inline.py" <<'PY'
import json, sys, torch
out = {
    "python": sys.version,
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": []
}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out["devices"].append({
            "logical_index": i,
            "name": p.name,
            "total_memory": p.total_memory,
            "capability": f"{p.major}.{p.minor}"
        })
    x = torch.tensor([1.0], device="cuda")
    y = x * 2
    torch.cuda.synchronize()
    out["tiny_cuda_result"] = float(y.item())
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python "$LOG_DIR/torch_cuda_probe_inline.py" | tee environment/torch_cuda_probe.json

echo "===== Update status/report ====="
cat > "$LOG_DIR/update_gate0_status.py" <<'PY'
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
PY
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python "$LOG_DIR/update_gate0_status.py"

echo "===== Final status ====="
cat reports/gate0_status.json
echo "===== Gate 0 resume install finished at $(date) ====="
