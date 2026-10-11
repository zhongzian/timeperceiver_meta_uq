#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
QUEUE="configs/queues/gate4b_cross_dataset.yaml"

"$PYTHON" scripts/gate4b_run_cross_dataset_smokes.py
"$PYTHON" scripts/gate4b_prepare_cross_dataset_queue.py
"$PYTHON" scripts/gate4b_validate_cross_dataset_queue.py

"$PYTHON" -m py_compile \
  scripts/gate4b_train_probabilistic.py \
  scripts/gate4b_formal_runner.py \
  scripts/gate4b_scheduler.py \
  scripts/gate4b_validate_cross_dataset_queue.py \
  scripts/gate4b_summarize_cross_dataset.py \
  scripts/gate4b_cross_dataset_formal_decision.py
bash -n scripts/gate4b_scheduler.sh
bash -n scripts/gate4b_cross_dataset_status.sh
bash -n scripts/gate4b_cross_dataset_refresh_summary.sh

if ps -u "$USER" -o cmd= | grep 'gate4b_scheduler.py' | grep -F "$QUEUE" | grep -v grep >/dev/null; then
  echo "EXISTING_CROSS_DATASET_SCHEDULER"
  exit 3
fi
if ps -u "$USER" -o cmd= | grep -E 'gate4b_cross_dataset.*(formal_runner|train_probabilistic)' | grep -v grep >/dev/null; then
  echo "BLOCKED_ORPHAN_CROSS_DATASET_PROCESS"
  exit 4
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/gate4b_cross_dataset_scheduler_${TS}.log"
nohup setsid bash scripts/gate4b_scheduler.sh start "$QUEUE" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > runtime/gate4b_cross_dataset_scheduler.pid
printf '%s\n' "$LOG" > runtime/gate4b_cross_dataset_scheduler.logpath
sleep 8
echo "FORMAL_SCHEDULER_PID=$PID"
echo "FORMAL_SCHEDULER_LOG=$LOG"
ps -p "$PID" -o pid,ppid,sid,stat,etime,cmd || true
tail -n 80 "$LOG" || true
