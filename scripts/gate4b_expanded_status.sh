#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/zza/code/timeperceiver_meta_uq
cat runtime/gate4b_expanded_etth1_scheduler_state.json 2>/dev/null || echo '{"status":"NOT_STARTED"}'
if [[ -s runtime/gate4b_expanded_etth1_scheduler.pid ]]; then
  PID="$(cat runtime/gate4b_expanded_etth1_scheduler.pid)"
  ps -p "$PID" -o pid,ppid,stat,etime,cmd || true
fi
