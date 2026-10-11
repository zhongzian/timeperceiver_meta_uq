#!/usr/bin/env bash
set -Eeuo pipefail

cd /mnt/data/zza/code/timeperceiver_meta_uq

echo "===== Gate 3b-smoke status ====="
cat reports/gate3b_smoke_status.json 2>/dev/null || echo "NO reports/gate3b_smoke_status.json"
echo
echo "===== Latest smoke run ====="
RUN="$(find runs -maxdepth 1 -type d -name 'timeperceiver__etth1__official_missing_eval_smoke__random_rate0.3__seed2025__*' | sort | tail -n 1 || true)"
echo "$RUN"
if [ -n "$RUN" ] && [ -d "$RUN" ]; then
  echo "----- status.json -----"
  cat "$RUN/status.json" || true
  echo "----- metrics.json -----"
  cat "$RUN/metrics.json" || true
  echo "----- eval tail -----"
  tail -n 80 "$RUN/eval.log" || true
fi
