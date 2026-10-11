#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/data/zza/code/timeperceiver_meta_uq
ACTION="${1:-}"
QUEUE="${2:-configs/queues/gate4a_minimal_adapter.yaml}"
case "$ACTION" in
  start) cd "$ROOT"; exec /mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python -u scripts/gate4a_scheduler.py "$QUEUE" ;;
  status) cd "$ROOT"; cat runtime/gate4a_minimal_adapter_scheduler_state.json 2>/dev/null || echo '{"status":"NOT_STARTED"}' ;;
  *) echo "usage: $0 {start|status} [queue]" >&2; exit 2 ;;
esac
