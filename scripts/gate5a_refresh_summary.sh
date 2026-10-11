#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND="Agg"
export MPLCONFIGDIR="/tmp/timeperceiver-gate5a-mpl"

"$PYTHON" scripts/gate5a_summarize_distribution_ablation.py
"$PYTHON" scripts/gate5a_formal_decision.py
"$PYTHON" scripts/gate5a_heavy_tail_diagnostics.py
"$PYTHON" scripts/gate5a_efficiency_benchmark.py
bash scripts/build_final_paper_artifacts.sh
"$PYTHON" scripts/gate5a_verify_integrity.py
