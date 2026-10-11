#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

echo "===== Gate 4c mechanism audit result ====="
cat results/gate4c_mechanism_audit.json 2>/dev/null || true

echo "===== Gate 4c mechanism audit files ====="
find artifacts/gate4c_mechanism_audit reports results -maxdepth 2 -type f 2>/dev/null | \
  grep -E 'gate4c_mechanism' | sort || true

echo "===== Protected formal artifact hashes ====="
sha256sum \
  results/gate4c_formal_decision.json \
  reports/gate4c_interval_metrics.csv \
  reports/gate4c_conditioned_bin_metrics.csv \
  runs/timeperceiver__etth1__gate4c__gaussian_head__random_rate0.3__seed2025__*/calibration/*.json \
  runs/timeperceiver__etth1__gate4c__student_t_head__random_rate0.3__seed2025__*/calibration/*.json \
  2>/dev/null || true

echo "===== Related processes (read only) ====="
ps -u "$USER" -o pid,ppid,stat,etime,pcpu,pmem,cmd | \
  grep -E 'gate4c|conformal|probabilistic|scheduler|TimePerceiver|python' | \
  grep -v grep || true

echo "===== Upstream ====="
git -C upstream/TimePerceiver rev-parse HEAD 2>/dev/null || true
git -C upstream/TimePerceiver status --short 2>/dev/null || true
git -C upstream/TimePerceiver diff --stat 2>/dev/null || true
