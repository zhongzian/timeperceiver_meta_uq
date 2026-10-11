#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
ACTION="${1:-status}"
QUEUE="${2:-configs/queues/gate5b_patchtst_backbones.yaml}"

cd "$ROOT"

case "$ACTION" in
  start)
    STEM="$(basename "$QUEUE" .yaml)"
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="logs/${STEM}_scheduler_${TS}.log"
    mkdir -p logs runtime
    nohup setsid "$PYTHON" -u scripts/gate5b_scheduler.py "$QUEUE" >"$LOG" 2>&1 < /dev/null &
    PID="$!"
    echo "scheduler_pid=$PID"
    echo "scheduler_log=$LOG"
    ;;
  status)
    "$PYTHON" scripts/gate5b_scheduler.py --status "$QUEUE"
    ;;
  stop)
    STOP_FILE="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["stop_file"])' "$QUEUE")"
    mkdir -p "$(dirname "$STOP_FILE")"
    touch "$STOP_FILE"
    echo "graceful_stop_requested=$STOP_FILE"
    ;;
  resume)
    "$0" start "$QUEUE"
    ;;
  *)
    echo "usage: $0 {start|status|stop|resume} [queue]" >&2
    exit 2
    ;;
esac

