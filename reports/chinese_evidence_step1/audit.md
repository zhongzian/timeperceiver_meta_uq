# 中文初稿证据核对：第一步

## 范围与结论

核对 Gate4b 四 head seed、Gate5a 分布消融、Gate5b PatchTST 概率结果。逐运行读取 metrics/config/status，复算统计并核验源 checkpoint 与固定 mask 的 SHA256。
本报告不代表全稿全部表格已审计；未修改历史结果、正式 decision 或冻结稿件。

## 必须修正的证据归属

- 多 seed 表的 Student-t 是 learned-df，不是 fixed df=8。初始 df=8 不等于固定自由度。
- 四个 seed 为概率头训练 seed；每个数据集共享一个 backbone checkpoint，不能写成四个 backbone seed。
- 固定 df=8 的 TimePerceiver/PatchTST 当前核对证据为 seed2025；不得把 learned-df 的 12 对结果移用为 fixed-df 的多 seed 结论。
- 表格差异须区分单 seed、四 seed 均值、learned/fixed 和 backbone；下表是同一指标原始文件重算值，标准差使用 ddof=1，单 seed 不估算标准差。
- 部分 resolved_config 继承 baseline experiment_id/run_dir。保留原件，以实际目录和 status 交叉追溯，详见 provenance_findings.csv。

- 两个 corrected_minimal 运行未记录 mask_application_verified 字段；input_mask_applied 为 true，资产哈希匹配，但不能把缺失字段写成已核验通过。
- 表4、表8、表10的 NLL/CRPS/MSE 已与逐运行指标核对；表4数值匹配不代表 fixed-df 标签正确。

## 中文初稿可用的分组结果

| 证据组 | 数据集 | 分布头 | n | NLL mean ± std | CRPS mean ± std | MSE mean ± std |
|---|---|---|---:|---|---|---|
| patchtst_distribution | electricity | fixed_student_t_df8 | 1 | 0.330148 (单 seed) | 0.204371 (单 seed) | 0.167809 (单 seed) |
| patchtst_distribution | electricity | gaussian | 1 | 0.413581 (单 seed) | 0.206058 (单 seed) | 0.165829 (单 seed) |
| patchtst_distribution | etth1 | fixed_student_t_df8 | 1 | 0.857116 (单 seed) | 0.335825 (单 seed) | 0.421389 (单 seed) |
| patchtst_distribution | etth1 | gaussian | 1 | 0.892433 (单 seed) | 0.338364 (单 seed) | 0.426829 (单 seed) |
| patchtst_distribution | weather | fixed_student_t_df8 | 1 | 0.277655 (单 seed) | 0.197832 (单 seed) | 0.182180 (单 seed) |
| patchtst_distribution | weather | gaussian | 1 | 2.096352 (单 seed) | 0.274037 (单 seed) | 0.232359 (单 seed) |
| timeperceiver_distribution | electricity | fixed_student_t_df8 | 1 | 0.367009 (单 seed) | 0.207797 (单 seed) | 0.168473 (单 seed) |
| timeperceiver_distribution | electricity | gaussian | 1 | 0.440598 (单 seed) | 0.209453 (单 seed) | 0.167881 (单 seed) |
| timeperceiver_distribution | electricity | laplace | 1 | 0.370388 (单 seed) | 0.208768 (单 seed) | 0.170436 (单 seed) |
| timeperceiver_distribution | electricity | learned_student_t | 1 | 0.366449 (单 seed) | 0.207786 (单 seed) | 0.168486 (单 seed) |
| timeperceiver_distribution | etth1 | fixed_student_t_df8 | 1 | 0.864949 (单 seed) | 0.336548 (单 seed) | 0.413927 (单 seed) |
| timeperceiver_distribution | etth1 | gaussian | 1 | 0.918535 (单 seed) | 0.344127 (单 seed) | 0.420332 (单 seed) |
| timeperceiver_distribution | etth1 | laplace | 1 | 0.879116 (单 seed) | 0.341785 (单 seed) | 0.424149 (单 seed) |
| timeperceiver_distribution | etth1 | learned_student_t | 1 | 0.877154 (单 seed) | 0.340263 (单 seed) | 0.419177 (单 seed) |
| timeperceiver_distribution | weather | fixed_student_t_df8 | 1 | 0.136165 (单 seed) | 0.208461 (单 seed) | 0.210994 (单 seed) |
| timeperceiver_distribution | weather | gaussian | 1 | 0.586325 (单 seed) | 0.245442 (单 seed) | 0.212237 (单 seed) |
| timeperceiver_distribution | weather | laplace | 1 | 0.204399 (单 seed) | 0.209589 (单 seed) | 0.210051 (单 seed) |
| timeperceiver_distribution | weather | learned_student_t | 1 | 0.129293 (单 seed) | 0.208276 (单 seed) | 0.210836 (单 seed) |
| timeperceiver_multiseed | electricity | gaussian | 4 | 0.439644 ± 0.002139 | 0.209365 ± 0.000178 | 0.167419 ± 0.000420 |
| timeperceiver_multiseed | electricity | learned_student_t | 4 | 0.365348 ± 0.000800 | 0.207395 ± 0.000338 | 0.167703 ± 0.000613 |
| timeperceiver_multiseed | etth1 | gaussian | 4 | 0.915014 ± 0.006643 | 0.342011 ± 0.002338 | 0.416862 ± 0.003302 |
| timeperceiver_multiseed | etth1 | learned_student_t | 4 | 0.871113 ± 0.007801 | 0.337656 ± 0.002124 | 0.415611 ± 0.002699 |
| timeperceiver_multiseed | weather | gaussian | 4 | 0.577226 ± 0.017061 | 0.244788 ± 0.003197 | 0.210306 ± 0.001462 |
| timeperceiver_multiseed | weather | learned_student_t | 4 | 0.103670 ± 0.020066 | 0.207544 ± 0.000670 | 0.209818 ± 0.000964 |

## 使用边界与下一步

mean_std.csv 与 paired_deltas.csv 来自逐运行指标，不从论文表格抄数。配对差值定义为目标头减 Gaussian。
不能将同一运行在不同阶段的复用当成新增独立样本。当前未计算显著性，不能把计数或均值优势表述为统计显著。
CRPS 延续既有实现；本轮未提升采样数或重评估。跨 backbone 的指标协议 hash 不同，不能宣称训练器完全相同。
下一步：用上述分组证据创建独立中文双栏初稿；补查其他表格与正文引用，不覆盖冻结投稿稿。

```json
{
  "stage": "Chinese revision step 1",
  "status": "PASS_WITH_DOCUMENT_CORRECTIONS",
  "scope": "Gate4b multiseed, Gate5a distribution, Gate5b probability; Table4/8/10 NLL, CRPS, MSE and head identity audit",
  "run_appearances": 42,
  "unique_runs": 36,
  "asset_and_source_files_hashed": 126,
  "inputs_unchanged": true,
  "errors": [],
  "provenance_warning_count": 76,
  "table_numeric_checks": 72,
  "training_started": false,
  "frozen_manuscript_modified": false,
  "whole_manuscript_audit_complete": false
}
```
