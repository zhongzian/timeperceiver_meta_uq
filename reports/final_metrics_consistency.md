# Final Metrics Consistency

**Status: PASS**

## gate2_baseline

Status: PASS

- clean_rows_3_of_3: PASS
- mse_matches_formal: PASS
- mae_matches_formal: PASS
- rmse_identity: PASS

## gate3b_48_mask_matrix

Status: PASS

- rows_48_of_48: PASS
- factorial_cells_complete: PASS
- dataset_type_curves_12_complete: PASS
- fixed_test_masks_verified: PASS
- all_status_success: PASS
- formal_status_pass: PASS
- evaluator_parity: PASS
- all_12_mse_curves_monotonic: PASS

## gate4a_decision

Status: PASS

- three_variants_found: PASS
- static_mse_delta: PASS
- missing_aware_official_delta: PASS
- missing_aware_static_delta: PASS
- static_beats_official: PASS
- conditioning_does_not_beat_static: PASS
- formal_decision_negative_conditioning: PASS

## gate4b_expanded

Status: PASS

- conditions_6_of_6: PASS
- probability_rows_12: PASS
- deltas_reproduced: PASS
- random03_corrected_fixed_mask: PASS
- old_unmasked_excluded: PASS
- formal_counts_reproduced: PASS
- formal_signal_positive: PASS

## gate4b_cross_dataset

Status: PASS

- weather_electricity_pairs: PASS
- probability_rows_4: PASS
- deltas_reproduced: PASS
- formal_counts_2_of_2: PASS
- formal_decision_reproduced: PASS

## gate4b_multiseed

Status: PASS

- datasets_3: PASS
- seeds_4: PASS
- variants_2: PASS
- probability_rows_24: PASS
- paired_cells_12: PASS
- paired_deltas_reproduced: PASS
- sample_mean_std_ddof1_reproduced: PASS
- formal_counts_12_12_12_12_3: PASS
- decision_counts_match: PASS
- probability_seed_scope_only: PASS
- formal_signal_positive: PASS

## gate4c_formal_and_mechanism

Status: PASS

- interval_rows_12: PASS
- head_method_level_factorial: PASS
- formal_implementation_pass: PASS
- formal_negative_signal: PASS
- conformal_ace_worse_in_all_8_comparisons: PASS
- mechanism_audit_complete: PASS
- mechanism_formal_gate_unchanged: PASS
- protected_formal_hashes_unchanged: PASS

## gate5a_distribution_ablation

Status: PASS

- rows_12: PASS
- datasets_3: PASS
- four_heads_per_dataset: PASS
- pairwise_rows_12: PASS
- pairwise_deltas_reproduced: PASS
- laplace_counts_match: PASS
- fixed_t_counts_match: PASS
- learned_df_counts_match: PASS
- implementation_pass: PASS
- mechanism_secondary_descriptive: PASS
- efficiency_complete: PASS

## gate5b_second_backbone

Status: PASS

- probability_rows_6: PASS
- datasets_3: PASS
- two_capacity_matched_heads_per_dataset: PASS
- paired_rows_3: PASS
- paired_deltas_reproduced: PASS
- formal_counts_match: PASS
- implementation_pass: PASS
- point_tradeoff_preserved: PASS
- tail_diagnostic_secondary: PASS
