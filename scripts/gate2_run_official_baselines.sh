#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/gate2 runtime

echo "Gate 2 official TimePerceiver baselines"
echo "cwd=${ROOT_DIR}"
echo "started=$(date '+%Y-%m-%d %H:%M:%S')"
echo "nvidia-smi snapshot:"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader || true

bash scripts/run_in_env.sh python scripts/gate2_official_runner.py --action run "$@"

echo "finished=$(date '+%Y-%m-%d %H:%M:%S')"
