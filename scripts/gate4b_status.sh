#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/zza/code/timeperceiver_meta_uq
cat runtime/gate4b_minimal_probabilistic_scheduler_state.json 2>/dev/null || echo '{"status":"NOT_STARTED"}'
PID_FILE=runtime/gate4b_minimal_probabilistic_scheduler.pid
if [[ -s "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  ps -p "$PID" -o pid,ppid,stat,etime,cmd || true
fi
