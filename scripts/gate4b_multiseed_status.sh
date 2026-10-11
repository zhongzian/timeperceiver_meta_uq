#!/usr/bin/env bash
set -u
ROOT=/mnt/data/zza/code/timeperceiver_meta_uq
QUEUE=configs/queues/gate4b_multiseed_confirmation.yaml
PID_FILE=runtime/gate4b_multiseed_confirmation_scheduler.pid
STATE=runtime/gate4b_multiseed_confirmation_scheduler_state.json
cd "$ROOT" || exit 1

echo "===== Gate 4b multi-seed scheduler PID ====="
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  ps -p "$PID" -o pid,ppid,stat,etime,pcpu,pmem,cmd || true
else
  echo "PID_FILE_NOT_FOUND"
fi

echo "===== Queue status ====="
bash scripts/gate4b_scheduler.sh status "$QUEUE" 2>/dev/null || true

echo "===== Queue-specific state ====="
cat "$STATE" 2>/dev/null || true

echo "===== Latest queue-specific log ====="
LATEST_LOG="$(find logs -maxdepth 1 -type f -name 'gate4b_multiseed_confirmation_scheduler_*.log' \
  -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
echo "LATEST_LOG=$LATEST_LOG"
if [[ -n "$LATEST_LOG" ]]; then
  tail -n 160 "$LATEST_LOG" || true
fi

echo "===== Related processes ====="
ps -u "$USER" -o pid,ppid,stat,etime,pcpu,pmem,cmd | \
  grep -E 'gate4b_multiseed|gate4b_scheduler|gate4b_formal_runner|gate4b_train_probabilistic' | \
  grep -v grep || true

echo "===== GPU ====="
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader || true
