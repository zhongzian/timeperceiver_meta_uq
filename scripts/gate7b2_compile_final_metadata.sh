#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREVIEW="$ROOT/paper/venues/journal_of_forecasting/submission_final_preview"
BUILD="/tmp/timeperceiver_gate7b2_preview_build"
TECTONIC="/mnt/data/zza/anaconda3/pkgs/tectonic-0.16.9-ha39f199_0/bin/tectonic"
ICU_LIB="/mnt/data/zza/anaconda3/pkgs/icu-78.3-h84d19a5_0/lib"

mkdir -p "$BUILD" "$ROOT/reports"

if [ ! -x "$TECTONIC" ]; then
  printf '%s\n' "Verified Tectonic executable is unavailable: $TECTONIC" >&2
  exit 2
fi

compile_preview() {
  local stem="$1"
  local log="$ROOT/reports/gate7b2_${stem}_compile.log"
  local exit_code

  (
    cd "$PREVIEW" || exit 90
    LD_LIBRARY_PATH="$ICU_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$TECTONIC" --only-cached --outdir "$BUILD" "${stem}.tex" > "$log" 2>&1
  )
  exit_code=$?
  if [ "$exit_code" -eq 0 ] && [ -s "$BUILD/${stem}.pdf" ]; then
    cp "$BUILD/${stem}.pdf" "$PREVIEW/${stem}.pdf"
  fi
  return "$exit_code"
}

title_exit=0
cover_exit=0
compile_preview title_page_preview_with_placeholders || title_exit=$?
compile_preview cover_letter_preview_with_placeholders || cover_exit=$?

printf '%s\n' "title_page_exit=$title_exit"
printf '%s\n' "cover_letter_exit=$cover_exit"

if [ "$title_exit" -ne 0 ]; then
  exit "$title_exit"
fi
exit "$cover_exit"
