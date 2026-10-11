#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.project_env"
ENV_NAME="UNKNOWN"
[[ -s "${ENV_FILE}" ]] && ENV_NAME="$(tr -d '[:space:]' < "${ENV_FILE}")"

echo "project_root=${ROOT_DIR}"
echo "conda_env=${ENV_NAME}"

if [[ -s "${ENV_FILE}" ]]; then
  bash "${ROOT_DIR}/scripts/run_in_env.sh" python - <<'PY'
import sys
print("python=" + sys.version.replace("\n", " "))
try:
    import torch
    print("torch=" + torch.__version__)
    print("torch_cuda_runtime=" + str(torch.version.cuda))
    print("cuda_available=" + str(torch.cuda.is_available()))
    print("cuda_device_count=" + str(torch.cuda.device_count()))
except Exception as exc:
    print("torch_probe_error=" + repr(exc))
PY
else
  echo "python=UNKNOWN"
  echo "torch=UNKNOWN"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader || true
else
  echo "nvidia-smi=NOT_AVAILABLE"
fi

if [[ -d "${ROOT_DIR}/upstream/TimePerceiver/.git" ]]; then
  echo "timeperceiver_commit=$(git -C "${ROOT_DIR}/upstream/TimePerceiver" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
else
  echo "timeperceiver_commit=UNKNOWN"
fi
