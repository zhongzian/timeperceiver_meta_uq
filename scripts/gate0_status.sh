#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_JSON="${ROOT_DIR}/reports/gate0_status.json"

if [[ ! -s "${STATUS_JSON}" ]]; then
  echo "Gate 0 status: NOT_AVAILABLE"
  echo "Missing ${STATUS_JSON}"
  exit 1
fi

python3 - "${STATUS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
print(f"Gate 0 status: {data.get('status', 'UNKNOWN')}")
print(f"timestamp: {data.get('timestamp', 'UNKNOWN')}")
print(f"conda_env: {data.get('conda_env', 'UNKNOWN')}")
print(f"python_version: {data.get('python_version', 'UNKNOWN')}")
print(f"torch_version: {data.get('torch_version', 'UNKNOWN')}")
print(f"torch_cuda_runtime: {data.get('torch_cuda_runtime', 'UNKNOWN')}")
print(f"cuda_available: {data.get('cuda_available', 'UNKNOWN')}")
print(f"gpu_count: {data.get('gpu_count', 'UNKNOWN')}")
print(f"repository_path: {data.get('repository_path', 'UNKNOWN')}")
print(f"repository_commit: {data.get('repository_commit', 'UNKNOWN')}")
print(f"latest_log_dir: {data.get('latest_log_dir', 'UNKNOWN')}")
blockers = data.get("blockers") or []
if blockers:
    print("blockers:")
    for item in blockers:
        print(f"- {item}")
PY
