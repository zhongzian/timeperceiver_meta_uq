#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
QUEUE="configs/queues/gate4b_cross_dataset.yaml"
STATE="runtime/gate4b_cross_dataset_scheduler_state.json"
echo "===== Gate 4b cross-dataset state ====="
cat "$STATE" 2>/dev/null || echo '{"status":"NOT_STARTED"}'
echo "===== Scheduler PID ====="
if [ -f runtime/gate4b_cross_dataset_scheduler.pid ]; then
  PID="$(cat runtime/gate4b_cross_dataset_scheduler.pid)"
  ps -p "$PID" -o pid,ppid,stat,etime,cmd 2>/dev/null || true
else
  echo "NO_PID_FILE"
fi
echo "===== Queue ====="
test -f "$QUEUE" && echo "$QUEUE" || echo "QUEUE_NOT_CREATED"
