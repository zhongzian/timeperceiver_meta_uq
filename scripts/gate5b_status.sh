#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/zza/code/timeperceiver_meta_uq

echo "===== Gate 5b backbone scheduler ====="
cat runtime/gate5b_patchtst_backbones_scheduler_state.json 2>/dev/null || echo '{"status":"NOT_STARTED"}'
echo "===== Gate 5b probability scheduler ====="
cat runtime/gate5b_patchtst_probability_scheduler_state.json 2>/dev/null || echo '{"status":"NOT_STARTED"}'
echo "===== Gate 5b smoke ====="
cat results/gate5b_backbone_smoke.json 2>/dev/null || echo '{"status":"NOT_STARTED"}'
echo "===== Gate 5b status ====="
cat results/gate5b_status.json 2>/dev/null || echo '{"status":"NOT_STARTED"}'
echo "===== Related processes ====="
ps -u "$USER" -o pid=,ppid=,stat=,etime=,cmd= | grep -E 'gate5b|patchtst|PatchTST' | grep -v grep || true
echo "===== GPU ====="
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

