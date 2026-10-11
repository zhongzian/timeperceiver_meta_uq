#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
PYTHON_BIN="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"

cd "$PROJECT_ROOT"
mkdir -p logs/gate3b reports runs scripts runtime/locks

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="timeperceiver__etth1__official_missing_eval_smoke__random_rate0.3__seed2025__${TS}"
RUN_DIR="runs/${RUN_ID}"
LOG_DIR="logs/gate3b/${TS}_smoke_missing_eval"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/gate3b_smoke_missing_eval.log"

{
  echo "===== Gate 3b-smoke missing eval started at $(date) ====="
  echo "RUN_ID=$RUN_ID"
  echo "RUN_DIR=$RUN_DIR"
  echo "Safety: evaluation-only smoke; no training will be started."
  echo
  echo "===== Gate status ====="
  test -f reports/gate0_status.json && cat reports/gate0_status.json || true
  test -f reports/gate1_status.json && cat reports/gate1_status.json || true
  test -f reports/gate2_status.json && cat reports/gate2_status.json || true
  test -f reports/gate3a_status.json && cat reports/gate3a_status.json || true
  test -f reports/gate3a_verify_status.json && cat reports/gate3a_verify_status.json || true
  echo
  echo "===== Execute smoke eval ====="
  set +e
  "$PYTHON_BIN" scripts/gate3b_smoke_missing_eval.py \
    --dataset etth1 \
    --mask_path artifacts/masks/etth1/random/rate_0.3/seed_2025.npz \
    --mask_meta artifacts/masks/etth1/random/rate_0.3/seed_2025.json \
    --run_dir "$RUN_DIR" \
    --max_eval_batches 8 \
    --eval_batch_size 32 \
    --device cpu \
    > "$RUN_DIR.eval.tmp" 2>&1
  RET=$?
  set -e
  if [ -d "$RUN_DIR" ]; then
    mv "$RUN_DIR.eval.tmp" "$RUN_DIR/eval.log"
  else
    mkdir -p "$RUN_DIR"
    mv "$RUN_DIR.eval.tmp" "$RUN_DIR/eval.log"
  fi
  echo "Smoke eval exit code: $RET"
  echo
  echo "===== Final status ====="
  test -f reports/gate3b_smoke_status.json && cat reports/gate3b_smoke_status.json || true
  test -f "$RUN_DIR/status.json" && cat "$RUN_DIR/status.json" || true
  test -f "$RUN_DIR/metrics.json" && cat "$RUN_DIR/metrics.json" || true
  echo
  echo "===== Eval log tail ====="
  tail -n 120 "$RUN_DIR/eval.log" || true
  echo "===== Gate 3b-smoke finished at $(date) ====="
  exit "$RET"
} 2>&1 | tee "$MAIN_LOG"
