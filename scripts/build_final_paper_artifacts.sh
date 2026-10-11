#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND="Agg"
export MPLCONFIGDIR="/tmp/timeperceiver-final-mpl"

cd "$ROOT"

"$PYTHON" scripts/build_final_evidence_registry.py
"$PYTHON" scripts/verify_final_evidence_registry.py
"$PYTHON" scripts/verify_final_metrics_consistency.py
"$PYTHON" scripts/build_paper_tables_and_sections.py
if [[ -f results/gate5a_formal_decision.json ]] && \
   "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open("results/gate5a_formal_decision.json")).get("GATE5A_IMPLEMENTATION_PASS") is True else 1)'; then
  "$PYTHON" scripts/gate5a_update_paper_evidence.py
fi
if [[ -f results/gate5b_formal_decision.json ]] && \
   "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open("results/gate5b_formal_decision.json")).get("GATE5B_IMPLEMENTATION_PASS") is True else 1)'; then
  "$PYTHON" scripts/gate5b_update_paper.py
fi
"$PYTHON" scripts/build_paper_figures.py
"$PYTHON" scripts/verify_paper_artifacts.py
"$PYTHON" scripts/gate6_build_submission_assets.py
"$PYTHON" scripts/gate6_scan_paper_claims.py
"$PYTHON" scripts/gate6_verify_manuscript_numbers.py
"$PYTHON" scripts/gate6_anonymization_audit.py
"$PYTHON" scripts/gate6_verify_submission_ready.py

bash scripts/final_synthesis_status.sh
