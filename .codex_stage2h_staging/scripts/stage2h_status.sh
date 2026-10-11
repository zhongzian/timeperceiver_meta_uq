#!/usr/bin/env bash
set -u

ROOT="/mnt/data/zza/code/BudgetedActiveChannelTokenFormer"
QUEUE="${ROOT}/configs/queues/stage2h_minimal_robustness.yaml"
PID_FILE="${ROOT}/runtime/stage2h_minimal_robustness_scheduler.pid"

cd "${ROOT}" || exit 1
echo "===== Stage 2H queue ====="
cat "${QUEUE}" 2>/dev/null || true
echo "===== Stage 2H queue-specific PID ====="
if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  echo "PID=${PID}"
  ps -p "${PID}" -o pid,ppid,stat,etime,pcpu,pmem,cmd || true
else
  echo "PID_FILE_NOT_FOUND"
fi
echo "===== Stage 2H run statuses ====="
find runs -maxdepth 2 -path '*/stage2h__*' -name status.json -print -exec cat {} \; 2>/dev/null || true
echo "===== Stage 2H scheduler logs ====="
find logs -maxdepth 1 -type f -name 'stage2h_scheduler_*.log' -print -exec tail -n 80 {} \; 2>/dev/null || true
echo "===== Persisted summary ====="
cat results/stage2h_robustness_status.json 2>/dev/null || true
echo "===== Global scheduler state (reference only) ====="
cat runtime/queue/scheduler_state.json 2>/dev/null || true
