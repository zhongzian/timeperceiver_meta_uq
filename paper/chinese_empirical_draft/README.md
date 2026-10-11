# 中文实证研究初稿

独立研究稿，不覆盖 `paper/manuscript`、投稿目录或任何正式实验决策。

- 正文：`main.tex`；双栏，跨栏摘要和三线表，数字引用。
- PDF：`build/main.pdf`。
- 构建：在项目根目录执行 `/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python scripts/build_chinese_empirical_draft.py`。
- 数值来源：`reports/chinese_evidence_step1/mean_std.csv` 与 `paired_deltas.csv`；表格由脚本生成，不手填。
- 当前稿使用三个数据集、H=96、random 0.3 核心证据。未把规划中的六数据集、固定df多seed、更多分布头写成完成结果。

## 写作与格式依据

结构参考近年来预测论文的问题定义、模型、实验、分析组织方式，版式按用户要求独立采用双栏，不宣称是下列会议的官方模板：

- [TimePerceiver, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html)
- [PatchTST, ICLR 2023](https://openreview.net/pdf?id=Jbdc0vTOcol)

中文字体使用服务器已有 WenQuanYi Micro Hei。此轮不安装系统软件，不联网下载TeX依赖。

## 后续修订清单

1. 在已有运行上检查逐窗口预测/损失资产，制定时间块配对bootstrap、效应量及多重比较方案。不能用重叠窗口元素数冒充独立样本数。
2. 核对64样本CRPS实现和估计误差，决定统一重评估方案；需两侧同时重算，保留历史口径。
3. 方差匹配、df敏感性和固定df多seed属于新增验证，不以learned-df运行替代。
4. 后续整合缺失机制扩展、校准、adapter负结果前，分别核验其逐运行来源。本稿没有把未完成二次审计的旧表直接复制进来。
5. 完善MAE、覆盖率/区间宽度、效率表和文献覆盖；补充图表须由可追溯数据生成。
6. 作者、单位、ORCID、资助等由真实信息补充，本稿不臆造；不作公开代码或录用保证。

语言处理以自然、准确、避免重复为目标，不以规避检测器为验证标准。结论边界集中写在协议和讨论中，结果段围绕观察展开。
