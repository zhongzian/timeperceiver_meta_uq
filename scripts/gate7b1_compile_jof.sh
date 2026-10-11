#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENUE="$ROOT/paper/venues/journal_of_forecasting"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
TECTONIC="/mnt/data/zza/anaconda3/pkgs/tectonic-0.16.9-ha39f199_0/bin/tectonic"
ICU_LIB="/mnt/data/zza/anaconda3/pkgs/icu-78.3-h84d19a5_0/lib"
EXIT_FILE="$VENUE/checks/compile_exit_codes.txt"

mkdir -p \
  "$VENUE/manuscript/build" \
  "$VENUE/supplementary/build" \
  "$VENUE/submission/build" \
  "$VENUE/checks"

: > "$EXIT_FILE"

if [ ! -x "$TECTONIC" ]; then
  printf '%s\n' "Tectonic 0.16.9 is unavailable at the verified path." >&2
  exit 2
fi

compile_document() {
  local name="$1"
  local source_dir="$2"
  local source_file="$3"
  local output_dir="$4"
  local log_file="$5"
  local destination="$6"
  local stem="${source_file%.tex}"
  local exit_code

  (
    cd "$source_dir" || exit 90
    LD_LIBRARY_PATH="$ICU_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$TECTONIC" --only-cached --keep-intermediates --keep-logs \
      --outdir "$output_dir" "$source_file" > "$log_file" 2>&1
  )
  exit_code=$?
  printf '%s=%s\n' "$name" "$exit_code" >> "$EXIT_FILE"
  if [ "$exit_code" -eq 0 ] && [ -s "$output_dir/$stem.pdf" ]; then
    cp "$output_dir/$stem.pdf" "$destination"
  fi
  return 0
}

compile_document \
  manuscript \
  "$VENUE/manuscript" \
  main.tex \
  "$VENUE/manuscript/build" \
  "$VENUE/manuscript/build/main_compile.log" \
  "$VENUE/submission/manuscript.pdf"

compile_document \
  anonymous_manuscript \
  "$VENUE/manuscript" \
  anonymous_main.tex \
  "$VENUE/manuscript/build" \
  "$VENUE/manuscript/build/anonymous_compile.log" \
  "$VENUE/submission/manuscript_anonymous.pdf"

compile_document \
  supplementary \
  "$VENUE/supplementary" \
  supplementary.tex \
  "$VENUE/supplementary/build" \
  "$VENUE/supplementary/build/supplementary_compile.log" \
  "$VENUE/submission/supplementary.pdf"

compile_document \
  title_page \
  "$VENUE/submission" \
  title_page.tex \
  "$VENUE/submission/build" \
  "$VENUE/submission/build/title_page_compile.log" \
  "$VENUE/submission/title_page.pdf"

compile_document \
  cover_letter \
  "$VENUE/submission" \
  cover_letter_draft.tex \
  "$VENUE/submission/build" \
  "$VENUE/submission/build/cover_letter_compile.log" \
  "$VENUE/submission/cover_letter.pdf"

if [ -s "$VENUE/submission/cover_letter.pdf" ]; then
  cp "$VENUE/submission/cover_letter.pdf" "$VENUE/submission/cover_letter_draft.pdf"
fi

"$PYTHON" "$ROOT/scripts/gate7b1_jof_compile_audit.py"
audit_exit=$?
"$PYTHON" "$ROOT/scripts/gate7b1_jof_anonymization_audit.py"
privacy_exit=$?

if [ "$audit_exit" -ne 0 ]; then
  exit "$audit_exit"
fi
exit "$privacy_exit"
