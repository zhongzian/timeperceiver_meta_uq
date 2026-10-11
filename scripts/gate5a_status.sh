#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

QUEUE="configs/queues/gate5a_distribution_ablation.yaml"
PID_FILE="runtime/gate5a_distribution_ablation_scheduler.pid"

echo "===== Gate 5a status (read only) ====="
cat results/gate5a_status.json 2>/dev/null || true
echo "===== Smoke state ====="
cat runtime/gate5a_protocol_smoke_state.json 2>/dev/null || true
echo "===== Queue state ====="
cat runtime/gate5a_distribution_ablation_scheduler_state.json 2>/dev/null || true
echo "===== Scheduler PID ====="
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  ps -p "$PID" -o pid,ppid,sid,stat,etime,cmd 2>/dev/null || true
fi
echo "===== Queue definition ====="
ls -lh "$QUEUE" 2>/dev/null || true
echo "===== Related processes ====="
ps -u "$USER" -o pid,ppid,stat,etime,cmd 2>/dev/null | \
  grep -E 'gate5a|gate4b_scheduler.py|gate4b_formal_runner.py|gate4b_train_probabilistic.py' | \
  grep -v grep || true
echo "===== GPU ====="
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader 2>/dev/null || true
