#!/usr/bin/env bash
set -u
cd /mnt/data/zza/code/timeperceiver_meta_uq || exit 1
echo '===== queue-specific PID ====='
[ -f runtime/gate3b_full_scheduler.pid ] && ps -p "$(cat runtime/gate3b_full_scheduler.pid)" -o pid,ppid,stat,etime,cmd || true
echo '===== latest queue-specific log ====='
LOG=$(ls -1t logs/gate3b_full_scheduler_*.log 2>/dev/null | head -n1); [ -n "${LOG:-}" ] && tail -n 80 "$LOG" || true
echo '===== static status (read-only) ====='; cat results/gate3b_full_status.json 2>/dev/null || true
echo '===== run counts ====='; grep -h '"status"' runs/timeperceiver__*__official_missing_eval_full__*/status.json 2>/dev/null | sort | uniq -c || true
echo '===== GPU ====='; nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
