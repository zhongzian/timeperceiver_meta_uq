#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1

while true; do
  if [[ -f results/gate5a_protocol_smoke.json ]]; then
    STATUS="$($PYTHON -c 'import json; print(json.load(open("results/gate5a_protocol_smoke.json")).get("status", "UNKNOWN"))')"
    if [[ "$STATUS" == "PASS" ]]; then
      break
    fi
    if [[ "$STATUS" == "BLOCKED_GATE5A_PROTOCOL_SMOKE_FAILURE" ]]; then
      echo "Gate 5a smoke failed; formal continuation stopped."
      exit 2
    fi
  fi
  sleep 60
done

"$PYTHON" scripts/gate5a_prepare_queue.py
"$PYTHON" scripts/gate5a_validate_queue.py

"$PYTHON" -m py_compile \
  models/probabilistic_head.py \
  models/probabilistic_timeperceiver.py \
  scripts/gate4b_train_probabilistic.py \
  scripts/gate4b_formal_runner.py \
  scripts/gate4b_scheduler.py \
  scripts/gate5a_validate_distributions.py \
  scripts/gate5a_validate_queue.py \
  scripts/gate5a_summarize_distribution_ablation.py \
  scripts/gate5a_formal_decision.py \
  scripts/gate5a_heavy_tail_diagnostics.py \
  scripts/gate5a_efficiency_benchmark.py

bash -n scripts/gate5a_status.sh
bash -n scripts/gate5a_refresh_summary.sh
bash -n scripts/gate4b_scheduler.sh

TRACKED="$({ git -C upstream/TimePerceiver status --short || true; } | grep -v '^??' || true)"
if [[ -n "$TRACKED" ]]; then
  echo "Upstream tracked diff is not empty."
  printf '%s\n' "$TRACKED"
  exit 2
fi

exec "$PYTHON" scripts/gate5a_launch_formal.py
