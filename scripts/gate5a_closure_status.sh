#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

echo "===== Gate 5a strict closure status (read only) ====="
cat results/gate5a_status.json 2>/dev/null || true

echo "===== Postprocess integrity ====="
cat results/gate5a_postprocess_integrity.json 2>/dev/null || true

echo "===== Closure verification ====="
cat results/gate5a_closure_verification.json 2>/dev/null || true

echo "===== Hash comparison ====="
cat artifacts/gate5a_closure/hash_comparison.json 2>/dev/null || true

echo "===== Related Gate 5a processes ====="
ps -u "$USER" -o pid,ppid,stat,etime,cmd 2>/dev/null | \
  grep -E 'gate5a_(run_protocol_smokes|continue_after_smokes|postformal_watcher|refresh_summary|heavy_tail_diagnostics|efficiency_benchmark)|gate5a_distribution_ablation' | \
  grep -v -E 'grep|gate5a_closure_status|gate5a_finalize_closure' || true

echo "===== Upstream ====="
git -C upstream/TimePerceiver rev-parse HEAD 2>/dev/null || true
git -C upstream/TimePerceiver status --short 2>/dev/null || true
