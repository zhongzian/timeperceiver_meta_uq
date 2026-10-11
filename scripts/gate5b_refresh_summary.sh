#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/zza/code/timeperceiver_meta_uq
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
required = (
    Path("runtime/gate5b_patchtst_backbones_scheduler_state.json"),
    Path("runtime/gate5b_patchtst_probability_scheduler_state.json"),
)
for path in required:
    if not path.is_file() or json.loads(path.read_text()).get("status") != "COMPLETE":
        raise SystemExit(f"queue is not COMPLETE: {path}")
PY

"$PYTHON" scripts/gate5b_summarize.py
"$PYTHON" scripts/gate5b_formal_decision.py
"$PYTHON" scripts/gate5b_capture_previous_final.py
if ! "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open("results/gate5b_tail_diagnostic.json")).get("status") == "PASS" else 1)' 2>/dev/null; then
  "$PYTHON" scripts/gate5b_tail_nll_gain.py
fi
bash scripts/build_final_paper_artifacts.sh
"$PYTHON" scripts/gate5b_verify_integrity.py
"$PYTHON" scripts/gate5b_finalize_closure.py
