# Gate 5a postprocess integrity

## Gate 5a postprocess integrity

**PASS**

## Execution safety

- training: NO
- optimizer: NO
- backward: NO
- GPU: NO
- scheduler: NO
- formal checkpoints modified: NO
- formal metrics modified: NO
- formal decisions modified: NO

## Formal completion

- protocol smoke: 6/6
- logical tasks: 12/12
- reused: 6
- new tasks: 6/6
- scheduler: COMPLETE
- formal decision: FIXED_HEAVY_TAIL_SUFFICIENT

## Postprocess completion

- diagnostics: PASS
- efficiency: PASS
- paper artifacts: PASS
- canonical evidence: PASS
- final synthesis: PASS

## Evidence integrity

- invalid legacy rows in paper: 0
- smoke rows in paper: 0
- unsupported claims: 0
- source hash mismatch: 0

## Formal source integrity

| Source | Pre-closure SHA256 | Post-closure SHA256 | Match |
| --- | --- | --- | --- |
| `results/gate4a_formal_decision.json` | `4fcd85430744941160cbe1108250c762764a81280ad54f2d5b2c85e3ce18d0bd` | `4fcd85430744941160cbe1108250c762764a81280ad54f2d5b2c85e3ce18d0bd` | TRUE |
| `results/gate4b_expanded_formal_decision.json` | `fa78308c9344d69cfb06474204e9f1ce46af15c8ea6751d2ad0fb2e191584047` | `fa78308c9344d69cfb06474204e9f1ce46af15c8ea6751d2ad0fb2e191584047` | TRUE |
| `results/gate4b_cross_dataset_formal_decision.json` | `073f10e4f4828a8402d854b366ec3c45e329fde60e2c53403cd6d7a940f556e3` | `073f10e4f4828a8402d854b366ec3c45e329fde60e2c53403cd6d7a940f556e3` | TRUE |
| `results/gate4b_multiseed_formal_decision.json` | `4f1abbaca515faddfa56bf635f03815028122a3d2161eb8188a1ff9881b9c0aa` | `4f1abbaca515faddfa56bf635f03815028122a3d2161eb8188a1ff9881b9c0aa` | TRUE |
| `results/gate4c_formal_decision.json` | `5e9d19bd8b3693510df80196e539a61464ecf403e6b612788e9e2d0806653600` | `5e9d19bd8b3693510df80196e539a61464ecf403e6b612788e9e2d0806653600` | TRUE |
| `results/gate5a_formal_decision.json` | `78a1772ae34ee67fe84da1729917fe28d827c512e65a9f4cfa1f51de8eece8c7` | `78a1772ae34ee67fe84da1729917fe28d827c512e65a9f4cfa1f51de8eece8c7` | TRUE |
| `reports/gate4b_multiseed_paired_deltas.csv` | `40da6068c4deabce2eca223bb79d2584be283a0920699773463da8fda82cb2dc` | `40da6068c4deabce2eca223bb79d2584be283a0920699773463da8fda82cb2dc` | TRUE |
| `reports/gate4b_multiseed_dataset_summary.csv` | `16a3bf34ef050c4305316e8b6e9e7bb509dbe170369a4e09450986c431c9cfeb` | `16a3bf34ef050c4305316e8b6e9e7bb509dbe170369a4e09450986c431c9cfeb` | TRUE |
| `reports/gate4c_interval_metrics.csv` | `24cff2a4833e493dc81f19a8efd8b81b4c0c57fe60bdf4a9d4c4e7792dedc73d` | `24cff2a4833e493dc81f19a8efd8b81b4c0c57fe60bdf4a9d4c4e7792dedc73d` | TRUE |
| `reports/gate4c_conditioned_bin_metrics.csv` | `5102708f75a8ae1c6b68dce60405b8073525cc6432ced29d4ab900b72e31ebe9` | `5102708f75a8ae1c6b68dce60405b8073525cc6432ced29d4ab900b72e31ebe9` | TRUE |
| `reports/gate5a_distribution_results.csv` | `f3d1db235adcc414959a9a4a83e12ea7450a28794609cc8e9bb2eae36d63e595` | `f3d1db235adcc414959a9a4a83e12ea7450a28794609cc8e9bb2eae36d63e595` | TRUE |
| `reports/gate5a_pairwise_deltas.csv` | `913209971fe76e2e3ad750c7cb11bf1b83b46931e86d92a77a0aa806a6efb03d` | `913209971fe76e2e3ad750c7cb11bf1b83b46931e86d92a77a0aa806a6efb03d` | TRUE |
| `results/final_evidence_registry.json` | `ad2fbc18b069f91a167ffbb1ee58c6cd60a3bcf70ac1bdee42a05d46e95fa927` | `ad2fbc18b069f91a167ffbb1ee58c6cd60a3bcf70ac1bdee42a05d46e95fa927` | TRUE |
| `results/final_project_status.json` | `51d8d3546644a30ef86e6407d79bea177df57400aaecc9e9d7b0ad86dc17de7d` | `51d8d3546644a30ef86e6407d79bea177df57400aaecc9e9d7b0ad86dc17de7d` | TRUE |
| `results/paper_artifact_verification.json` | `0a7f996fca7bf5b10dc1522ff6c79b25792900751d8b7e69250e5c1b02ee2372` | `0a7f996fca7bf5b10dc1522ff6c79b25792900751d8b7e69250e5c1b02ee2372` | TRUE |

## Closure gates

```text
GATE5A_DIAGNOSTICS_COMPLETE=True
GATE5A_EFFICIENCY_COMPLETE=True
GATE5A_PAPER_UPDATE_COMPLETE=True
GATE5A_FINAL_EVIDENCE_REBUILD_PASS=True
GATE5A_FORMAL_INTEGRITY_PASS=True
GATE5A_POSTPROCESS_COMPLETE=True
FINAL_SYNTHESIS_PASS=True
READY_FOR_SECOND_BACKBONE_PLANNING=True
```
