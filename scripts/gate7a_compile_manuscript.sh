#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANUSCRIPT="$ROOT/paper/manuscript"
BUILD="$MANUSCRIPT/build"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
TECTONIC="/mnt/data/zza/anaconda3/pkgs/tectonic-0.16.9-ha39f199_0/bin/tectonic"
ICU_LIB="/mnt/data/zza/anaconda3/pkgs/icu-78.3-h84d19a5_0/lib"

mkdir -p "$BUILD"
cd "$MANUSCRIPT" || exit 1

ENGINE=""
COMPILE_EXIT=1

if command -v latexmk >/dev/null 2>&1; then
    ENGINE="latexmk"
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex > build/compile.log 2>&1
    COMPILE_EXIT=$?
elif [ -x "$TECTONIC" ]; then
    ENGINE="Tectonic 0.16.9 (offline cached bundle)"
    LD_LIBRARY_PATH="$ICU_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$TECTONIC" --only-cached --keep-intermediates --keep-logs --outdir build main.tex \
        > build/compile.log 2>&1
    COMPILE_EXIT=$?
else
    ENGINE="unavailable"
    printf '%s\n' "No supported LaTeX engine is available." > build/compile.log
fi

if [ "$COMPILE_EXIT" -eq 0 ] && [ -s "$BUILD/main.pdf" ]; then
    cp "$BUILD/main.pdf" "$BUILD/manuscript.pdf"
fi

"$PYTHON" "$ROOT/scripts/gate7a_analyze_compile.py" \
    --exit-code "$COMPILE_EXIT" \
    --engine "$ENGINE"
ANALYZE_EXIT=$?

if [ "$COMPILE_EXIT" -ne 0 ]; then
    exit "$COMPILE_EXIT"
fi
exit "$ANALYZE_EXIT"
