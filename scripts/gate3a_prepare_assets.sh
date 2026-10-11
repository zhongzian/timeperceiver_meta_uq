#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
PYTHON_BIN="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"

cd "$PROJECT_ROOT"

ts="$(date +%Y%m%d_%H%M%S)"
log_dir="logs/gate3a/${ts}_prepare_assets"
mkdir -p "$log_dir" reports artifacts/masks artifacts/splits artifacts/normalization configs/missingness
log_file="$log_dir/gate3a_prepare_assets.log"

{
  echo "===== Gate 3a prepare assets started at $(date) ====="
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo
  echo "===== Gate status inputs ====="
  test -f reports/gate0_status.json && cat reports/gate0_status.json || true
  test -f reports/gate1_status.json && cat reports/gate1_status.json || true
  test -f reports/gate2_status.json && cat reports/gate2_status.json || true
  echo
  echo "===== Upstream git state ====="
  git -C upstream/TimePerceiver rev-parse HEAD
  git -C upstream/TimePerceiver status --short
  git -C upstream/TimePerceiver diff --stat
  echo
  echo "===== Generate masks ====="
  "$PYTHON_BIN" scripts/gate3a_generate_missingness_assets.py
  echo
  echo "===== Verify masks ====="
  "$PYTHON_BIN" scripts/gate3a_verify_missingness_assets.py
  echo
  echo "===== Environment and official code snapshot ====="
  {
    date
    hostname
    whoami
    "$PYTHON_BIN" --version
    "$PYTHON_BIN" -c "import numpy, pandas; print('numpy', numpy.__version__); print('pandas', pandas.__version__)"
    git -C upstream/TimePerceiver rev-parse HEAD
    git -C upstream/TimePerceiver status --short
    git -C upstream/TimePerceiver diff --stat
  } > reports/gate3a_env_git_snapshot.txt
  cat reports/gate3a_env_git_snapshot.txt
  echo
  echo "===== Final status ====="
  cat reports/gate3a_status.json
  cat reports/gate3a_verify_status.json
  echo "===== Gate 3a prepare assets finished at $(date) ====="
} 2>&1 | tee "$log_file"
