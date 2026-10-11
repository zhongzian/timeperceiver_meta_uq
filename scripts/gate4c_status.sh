#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
QUEUE="configs/queues/gate4c_minimal_conformal.yaml"

echo "===== Gate 4c scheduler state (read only) ====="
cat runtime/gate4c_minimal_conformal_scheduler_state.json 2>/dev/null || echo '{"status":"NOT_STARTED"}'
echo "===== Gate 4c scheduler PID (read only) ====="
if [ -f runtime/gate4c_minimal_conformal_scheduler.pid ]; then
  PID="$(cat runtime/gate4c_minimal_conformal_scheduler.pid)"
  ps -p "$PID" -o pid,ppid,stat,etime,cmd 2>/dev/null || true
else
  echo "PID_FILE_NOT_FOUND"
fi
echo "===== Gate 4c run statuses (read only) ====="
find runs -maxdepth 2 -path '*gate4c*' -name status.json -print 2>/dev/null | sort | while read -r STATUS; do
  echo "----- $STATUS -----"
  cat "$STATUS"
done
echo "===== Queue definition (read only) ====="
cat "$QUEUE" 2>/dev/null || true

