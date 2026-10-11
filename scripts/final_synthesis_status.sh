#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"

test -f "$ROOT/results/final_project_status.json"
test -f "$ROOT/results/paper_artifact_verification.json"

jq '{
  status,
  final_synthesis_pass,
  main_contribution,
  valid_formal_evidence_count,
  invalid_legacy_evidence_count,
  gate4a_conditioning_signal_positive,
  gate4b_multiseed_signal_positive,
  gate4c_signal_positive,
  FINAL_EVIDENCE_REGISTRY_PASS,
  FINAL_METRICS_CONSISTENCY_PASS,
  FINAL_CLAIM_EVIDENCE_PASS,
  FINAL_PAPER_ARTIFACT_PASS,
  FINAL_SYNTHESIS_PASS,
  GATE6_MANUSCRIPT_INTEGRATION_PASS,
  GATE6_CLAIM_SCAN_PASS,
  GATE6_NUMBER_VERIFICATION_PASS,
  GATE6_ANONYMIZATION_PASS,
  GATE6_SUBMISSION_ASSET_PASS,
  GATE6_FORMAL_INTEGRITY_PASS,
  GATE6_SUBMISSION_READY,
  gate5b_decision,
  gate5b_nll_win_count,
  gate5b_crps_noninferior_count,
  gate5b_point_noninferior_count,
  gate5b_full_pair_pass_count,
  upstream_commit,
  upstream_tracked_diff_empty
}' "$ROOT/results/final_project_status.json"
