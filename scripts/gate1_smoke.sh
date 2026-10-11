#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
ENV_NAME="timeperceiver_meta_uq"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/gate1/gate1__smoke__${TS}"
LOG_FILE="$LOG_DIR/gate1_smoke.log"
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/reports"

exec > >(tee -a "$LOG_FILE") 2>&1

cd "$PROJECT_ROOT"
echo "===== Gate 1 smoke started at $(date) ====="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "ENV_NAME=$ENV_NAME"

echo "===== AGENTS files ====="
for f in /mnt/data/zza/code/AGENTS.md "$PROJECT_ROOT/AGENTS.md"; do
  if [ -f "$f" ]; then
    echo "FOUND_AGENTS: $f"
    head -n 5 "$f" || true
  else
    echo "MISSING_AGENTS: $f"
  fi
done

echo "===== Git/upstream state ====="
git -C upstream/TimePerceiver rev-parse HEAD
git -C upstream/TimePerceiver status --short || true

echo "===== GPU snapshot ====="
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader || true

echo "===== Environment ====="
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python --version
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())'

echo "===== Run smoke runner ====="
/mnt/data/zza/anaconda3/bin/conda run -n "$ENV_NAME" python scripts/gate1_smoke.py "$@"
echo "===== Gate 1 smoke finished at $(date) ====="
