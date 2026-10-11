#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/data/zza/code/timeperceiver_meta_uq
ACTION="${1:-}"
QUEUE="${2:-configs/queues/gate4b_minimal_probabilistic.yaml}"
case "$ACTION" in
  start) cd "$ROOT"; exec /mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python -u scripts/gate4b_scheduler.py "$QUEUE" ;;
  status) cd "$ROOT"; exec /mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python scripts/gate4b_scheduler.py --status "$QUEUE" ;;
  *) echo "usage: $0 {start|status} [queue]" >&2; exit 2 ;;
esac
