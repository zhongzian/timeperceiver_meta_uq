# TimePerceiver Meta-UQ 项目专用规范

## 1. 项目定位

基础模型：

```text
TimePerceiver
```

项目暂定名称：

```text
Meta-UQ TimePerceiver
```

核心研究问题：

```text
在训练缺失机制与测试缺失机制不一致时，
如何使 TimePerceiver 快速适应新的缺失模式，
并输出经过校准的概率预测结果？
```

预定的两个主要创新方向：

1. Missingness-Aware Meta Adapter
   缺失机制感知的条件适配器与测试时自监督快速适配。

2. Missingness-Conditioned Probabilistic Calibration
   缺失模式条件化的概率预测头与保序校准。

对比学习只能作为辅助正则项，不作为唯一核心创新。

---

## 2. 项目目录

统一项目路径：

```text
/mnt/data/zza/code/timeperceiver_meta_uq
```

建议目录结构：

```text
timeperceiver_meta_uq/
├── upstream/                 # 未修改的官方代码镜像或引用
├── models/
├── modules/
│   ├── missingness_encoder.py
│   ├── conditional_adapter.py
│   ├── probabilistic_head.py
│   └── contrastive_loss.py
├── meta/
│   ├── episode_sampler.py
│   └── inner_loop.py
├── calibration/
│   ├── conformal.py
│   └── mask_conditional.py
├── data_provider/
├── configs/
│   ├── baseline/
│   ├── adapter/
│   ├── probabilistic/
│   └── combined/
├── scripts/
│   ├── reproduce/
│   ├── smoke/
│   ├── ablation/
│   └── scheduler/
├── artifacts/
│   ├── splits/
│   ├── masks/
│   └── normalization/
├── runs/
├── reports/
├── runtime/
│   └── locks/
├── tests/
└── AGENTS.md
```

不得随意重构官方仓库。先保留官方入口，再在最少文件上扩展。

---

## 3. 固定研究阶段

### 阶段 A：官方基线复现

优先选择：

```text
ETTh1
Weather
Electricity
```

第一轮只运行：

* 单 seed；
* 官方默认或论文主要预测长度；
* 官方数据划分；
* 官方损失与评价方法。

必须依次完成：

1. 数据加载 smoke test；
2. forward；
3. backward；
4. optimizer step；
5. checkpoint 保存；
6. checkpoint 加载；
7. 验证和测试；
8. metrics.json 写入。

完成判据：

* 至少一个数据集得到可与论文比较的正式结果；
* 三个数据集均完成 smoke test；
* 原始基线可通过统一配置运行；
* 与论文值的偏差已记录。

### 阶段 B：建立缺失评估协议

在创新前固定：

* train/val/test split；
* normalization statistics；
* test mask；
* 指标实现；
* mask seed；
* evaluation script。

首批缺失类型：

```text
random
block
channel
mixed
```

首批 missing rate：

```text
0.1
0.3
0.5
0.7
```

测试 mask 必须预生成并保存：

```text
artifacts/masks/{dataset}/{mask_type}/rate_{rate}/seed_{seed}.npz
```

基线和所有改进必须读取同一份测试 mask。

### 阶段 C：创新一快速验证

创新一拆成两个递进版本：

```text
C1: Missingness-Conditioned Adapter
C2: Meta-Learned Missingness-Conditioned Adapter
```

先验证 C1，再验证 C2。

不得直接跳到复杂 MAML。若普通条件适配器没有收益，先分析表示、条件变量和插入位置，不能用元学习掩盖基础模块无效。

Missingness Encoder 首批输入：

* 总体缺失率；
* 通道级缺失率；
* 最大连续缺失长度；
* 缺失块数量；
* 距离最近观测的时间；
* 输入长度；
* 预测长度；
* mask 类型 embedding。

Adapter 首选插入位置：

1. latent cross-attention 后；
2. latent self-attention 后；
3. query cross-attention 前或后。

第一轮每次只测试一个插入位置。

首选条件化机制：

```text
FiLM 或 Bottleneck Adapter
```

暂不同时叠加 LoRA、Hypernetwork 和多个 adapter 结构。

元学习优先级：

```text
ANIL > First-Order MAML > Meta-SGD
```

inner loop 默认只更新：

* adapter；
* missingness encoder 的末端映射；
* normalization affine parameters。

禁止使用测试未来真值进行适配。

测试时适配只能通过：

1. 从测试历史中选择已观测位置；
2. 人工遮挡其中一部分；
3. 计算自监督重建损失；
4. 更新 adapter；
5. 恢复真实测试 mask；
6. 预测未来。

### 阶段 D：创新二快速验证

创新二拆成：

```text
D1: Probabilistic Forecasting Head
D2: Missingness-Conditional Conformal Calibration
```

概率头第一选择：

```text
Student-t distribution head
```

备选：

```text
Quantile regression head
```

第一版不得同时实现 Student-t、Gaussian mixture、normalizing flow 和 diffusion。

Student-t head 输出：

```text
mu
scale
degrees_of_freedom
```

必须做稳定约束：

```text
scale > epsilon
degrees_of_freedom > 2
```

评价至少包括：

* MSE；
* MAE；
* NLL；
* CRPS 或可靠近似；
* prediction interval coverage；
* interval width；
* calibration error。

Conformal calibration 必须只使用 validation/calibration set，禁止使用 test labels 调参。

缺失条件校准首选：

* 按 missingness embedding 的近邻加权；
* 或按 missing rate 与 mask type 分组。

不得将测试集整体残差用于事后校准。

### 阶段 E：联合模型

只有 C 或 D 至少一个在公平单模块实验中显示有意义改善后，才运行：

```text
Meta Adapter + Probabilistic Head + Conditional Calibration
```

不得在两个单模块都无效时直接宣称联合模型有效。

---

## 4. 可选对比学习模块

对比学习为第三优先级，只在前两个核心方向稳定后实施。

推荐形式：

```text
Missingness-Invariant Latent Consistency
```

同一窗口构造两个不同缺失视图：

```text
view_a = mask_a(x)
view_b = mask_b(x)
```

对齐内容：

* latent bottleneck representation；
* 相同 query position 的预测分布。

优先损失：

```text
VICReg
Barlow Twins
Distribution consistency
```

普通 InfoNCE 不作为默认选择，以降低时间序列 false negative 问题。

模块必须可关闭：

```yaml
model:
  use_contrastive_regularization: false
```

对比损失权重必须单独搜索并记录，不得与其他创新同时首次引入。

---

## 5. 数据泄漏检查

每次实现新的 mask、适配或校准流程，必须检查：

* 输入窗口是否包含预测区间；
* normalization 是否使用 test statistics；
* test-time adaptation 是否使用未来标签；
* calibration set 是否与 test set 重叠；
* 随机 mask 是否意外遮挡目标之外的位置；
* padding mask 是否被误当作真实缺失；
* query 时间位置是否泄露真实目标值；
* 测试 mask 是否在不同模型间重新采样。

任何泄漏风险未排除前，不得运行正式实验或报告提升。

---

## 6. 基线模式

至少保留以下模式：

```text
baseline
adapter_only
meta_adapter
probabilistic_only
probabilistic_calibrated
adapter_probabilistic
full_model
```

配置示例：

```yaml
model:
  use_missingness_encoder: false
  use_conditional_adapter: false
  use_meta_adaptation: false
  use_probabilistic_head: false
  use_conformal_calibration: false
  use_contrastive_regularization: false
```

`baseline` 模式下所有新模块必须关闭，并恢复官方损失、输出和数据流程。

---

## 7. 第一轮筛选标准

单 seed 仅用于筛选，不用于正式结论。

建议将以下情况视为“值得继续”：

* 多个 missing rate 下方向一致；
* 至少两个数据集受益；
* 没有明显牺牲完整输入场景；
* 提升并非只来自参数量显著增加；
* 训练稳定；
* 没有校准或数据泄漏问题；
* 额外推理成本处于可接受范围。

以下情况暂停扩展：

* 只在单个 seed 或单个 missing rate 提升；
* 只改善训练集或验证集；
* 提升来自不同测试 mask；
* 提升来自更长训练预算；
* 点预测提升但概率指标明显恶化；
* 参数量和计算量增长远大于收益；
* 测试时适配依赖未来真值。

不得预设固定百分比作为录用保证，但必须报告绝对变化、相对变化和计算代价。

---

## 8. 正式实验条件

进入正式实验前必须满足：

* 基线复现可信；
* 固定评估资产完成；
* 至少一个创新模块有稳定单 seed 信号；
* 模块可独立关闭；
* 没有测试泄漏；
* 结果可由 experiment ID 完整还原。

正式实验至少包括：

* 3–5 个随机 seed；
* 3 个以上数据集；
* 4 类缺失模式；
* 多个 missing rate；
* 多个预测长度；
* 完整消融；
* 参数量、速度和显存比较；
* 统计显著性检验；
* 失败案例；
* 缺失分布迁移实验。

---

## 9. 结果表

结果表至少增加以下字段：

```text
forecast_mse
forecast_mae
imputation_mse
nll
crps
coverage_80
coverage_90
interval_width_80
interval_width_90
ece_or_calibration_error
adaptation_steps
adaptation_time
parameter_count
peak_gpu_memory
train_time
inference_time
```

无法计算的指标填 `NA`，不得估算或虚构。

---

## 10. 研究结论边界

不得使用以下无证据表述：

* 保证达到 SCI 二区；
* 一定优于所有方法；
* 已解决 MNAR；
* 完全适用于真实工业系统；
* 可保证商业收入。

允许的表述应基于实验，例如：

* 在固定协议下取得稳定改善；
* 对未见缺失机制表现出更好的鲁棒性；
* 预测区间覆盖率更接近目标水平；
* 在相同训练预算下改善精度；
* 在一定计算开销下获得更优精度—成本权衡。

是否继续投稿或产品化，必须根据完整结果判断。


# TimePerceiver 项目口头要求补充条款

## 1. Codex 执行方式

本项目所有环境搭建、代码创建、代码修改、测试、调试、实验运行、调度脚本和结果汇总，默认由 Codex 完成。

每个新阶段开始前，Codex 必须依次读取：

1. `/mnt/data/zza/code/AGENTS.md`
2. `/mnt/data/zza/code/timeperceiver_meta_uq/AGENTS.md`
3. 当前工作目录及其父子目录中适用的 `AGENTS.override.md`

当前任务指令与 AGENTS 文件冲突时，执行更具体、更新、离当前工作目录更近的要求。

每阶段开始时输出：

* 推荐推理模式；
* 本阶段目标；
* 完成判据；
* 将执行的主要操作。

能够安全一次完成的相关步骤，应合并为一个可重复执行的自包含脚本或一条总命令。不要把一个完整操作拆分成大量需要用户反复复制的零碎命令。

只有遇到权限、网络登录、受限数据授权或必须由用户完成的学校系统操作时，才将操作交给用户。此时必须给出一段完整、可复制、可直接执行的命令，并说明执行位置和预期结果。

---

## 2. 固定开发环境

开发方式：

* 本地 VS Code；
* Remote-SSH 连接学校服务器；
* 通过 VS Code Codex 插件执行任务。

实验根目录：

```text
/mnt/data/zza/code
```

项目目录：

```text
/mnt/data/zza/code/timeperceiver_meta_uq
```

项目必须创建独立 Conda 环境，禁止污染 `base` 和其他项目环境。

服务器驱动显示 CUDA 12.2，但不得仅根据该版本强制安装同版本 CUDA Toolkit。必须检查：

```bash
nvidia-smi
nvcc --version
conda --version
python --version
```

并根据 TimePerceiver 官方依赖、PyTorch 官方兼容矩阵和服务器驱动选择环境。

GPU 逻辑编号与物理编号存在反向映射：

* 逻辑 GPU 0 对应物理 GPU 1；
* 逻辑 GPU 1 对应物理 GPU 0。

启动任务前必须打印并记录：

* GPU logical index；
* GPU UUID；
* GPU name；
* total memory；
* used memory；
* free memory；
* CUDA_VISIBLE_DEVICES；
* 当前 Conda 环境；
* PyTorch 和 CUDA runtime 版本。

优先使用 GPU UUID 或经过核验的稳定映射，不能盲信编号。

---

## 3. 最快复现原则

复现阶段的目标是尽快获得可信且可比较的 TimePerceiver 基线，而不是提前建设完整论文工程。

第一阶段只复现 2–3 个关键数据集，优先：

1. ETTh1；
2. Weather；
3. Electricity。

第一轮只使用一个固定 seed。

执行顺序：

1. 获取并记录官方仓库 commit；
2. 创建独立环境；
3. 下载或准备必要数据；
4. 运行极小 smoke test；
5. 确认 forward、backward、optimizer、checkpoint 和 evaluation 链路；
6. 按官方协议运行正式单 seed 基线；
7. 将结果与论文值进行比较；
8. 基线可信后立即进入创新开发。

smoke test 允许：

* 少量 batch；
* 1 个 epoch；
* 单个数据集；
* 独立 smoke experiment ID。

smoke test 仅用于发现语法、路径、维度、保存和加载问题，不得作为论文复现结果。

正式复现不得为节省显存或时间而静默改变：

* epoch；
* batch size；
* 模型规模；
* 输入长度；
* 预测长度；
* 数据划分；
* optimizer；
* scheduler；
* early stopping；
* checkpoint 选择规则；
* 评价指标。

---

## 4. 创新验证顺序

基线完成后立即实施创新，不提前做完整多 seed 和大规模消融。

首个创新方向：

```text
Missingness-Conditioned Adapter
缺失机制条件适配器
```

其快速验证顺序：

1. 普通缺失机制编码器；
2. FiLM 或瓶颈适配器；
3. 单一插入位置；
4. 单数据集、单 seed；
5. 与完全相同协议的官方基线比较。

如果普通条件适配器没有信号，不直接叠加复杂元学习。先分析：

* mask 特征是否正确；
* 缺失值处理是否正确；
* 插入位置是否合理；
* adapter 是否真正接收梯度；
* loss 权重是否合理；
* 参数量是否造成不公平比较；
* 数据缺失机制是否能够体现模块价值。

普通适配器有效后，再实施：

```text
Meta-Learned Missingness Adapter
元学习缺失机制适配器
```

第二创新方向：

```text
Student-t Probabilistic Head
学生 t 分布概率输出头
```

以及：

```text
Missingness-Conditional Conformal Calibration
缺失机制条件保序校准
```

第一轮创新比较只允许改变一个主要因素，其余训练和评估设置与基线完全一致。

单 seed 只用于筛选方向。出现稳定且有意义的收益后，才扩展：

* 多 seed；
* 更多数据集；
* 更多 missing rate；
* 更多 mask 类型；
* 多预测长度；
* 完整消融；
* 强基线；
* 复杂度分析；
* 统计显著性检验。

如果创新无效，应持续迭代最小可验证修改，而不是用大量实验掩盖问题。

---

## 5. 多实验统一调度

存在多个实验时，优先创建统一总调度器，不手工逐个启动。

调度器必须支持：

* nohup 后台运行；
* 非阻塞轮询；
* pending 队列；
* running 队列；
* succeeded、retrying、failed 和 waiting_resource 状态；
* 最大并发 3；
* 根据实时 GPU 和显存动态补位；
* 实验结束后立即检查并启动下一项；
* 独立日志、PID、配置、输出目录和状态文件；
* 调度器重启后恢复队列；
* checkpoint 完整时支持断点续训；
* status、stop、resume 命令；
* 调度器退出后不遗留僵尸进程。

并发规则：

* 只够运行 1 个实验时，只运行 1 个；
* 够运行 2 个时补到 2 个；
* 够运行 3 个时补到 3 个；
* 其他任务继续等待；
* 后续出现更多空闲显存时立即补位；
* 已正常运行的实验不因新实验而被主动中断。

实验启动必须使用 `flock` 或等价文件锁，原子完成：

1. 检查 GPU；
2. 选择 GPU；
3. 登记显存预留；
4. 写任务状态；
5. 启动进程；
6. 记录 PID；
7. 释放启动锁。

GPU 锁只防止本项目自己的任务同时抢占同一资源，不能妨碍其他合法用户。

---

## 6. 显存和 OOM 规则

正式实验不得通过缩小模型、减少 epoch、缩短预测长度或改变数据协议来静默适配显存。

调度器必须根据以下信息决定是否启动：

* 当前空闲显存；
* smoke test 或 profiling 得到的峰值；
* 同配置历史峰值；
* GPU 总显存；
* 动态显存安全余量。

显存安全余量只决定任务是否等待，不改变实验配置。

不得通过以下方式囤积显存：

* 无用 Tensor；
* 空训练进程；
* 故意占满 GPU 的程序；
* 杀死或阻碍其他用户任务。

正在运行的实验应正常持有自身计算所需显存，并保存 checkpoint，以降低外部中断造成的损失。

OOM 任务必须记录：

* oom_retry_count；
* last_oom_time；
* required_memory_estimate；
* next_eligible_time；
* 启动时空闲显存；
* OOM 前显存占用或历史峰值。

OOM 重试采用递增等待或指数退避。只有出现更多空闲显存或更合适 GPU 时才重新启动。达到较多重试次数后进入 `WAITING_RESOURCE`，降低检查频率但继续等待。

OOM 后只允许清理该实验自身确认无用的父子进程和 CUDA context，不得杀死其他用户进程。

---

## 7. 错误分类和重试

任务必须区分：

```text
SUCCESS
OOM
WAITING_RESOURCE
DATA_ERROR
CONFIG_ERROR
CODE_ERROR
NAN_OR_DIVERGENCE
EXTERNAL_KILL
UNKNOWN_ERROR
```

处理方式：

* SUCCESS：归档结果并更新总表；
* OOM：清理自身残留，更新显存估计，退避后回队列；
* WAITING_RESOURCE：低频等待资源；
* DATA_ERROR：停止自动重试，检查数据；
* CONFIG_ERROR：停止自动重试，修复后使用新 experiment ID；
* CODE_ERROR：停止自动重试，修复并重新 smoke test；
* NAN_OR_DIVERGENCE：保存状态，诊断数值问题，不无限重跑；
* EXTERNAL_KILL：确认 checkpoint 完整时恢复；
* UNKNOWN_ERROR：保存错误并进行有限诊断，无法确定后暂停。

只有 OOM、资源不足和确认的临时外部故障可以长期等待。确定性错误不得无限回队列。

---

## 8. 公平比较和固定评估资产

基线和创新快速比较必须固定：

* train/validation/test split；
* normalization statistics；
* test mask；
* mask seed；
* evaluation script；
* metrics implementation；
* checkpoint selection；
* data preprocessing version。

测试 mask 保存为：

```text
artifacts/masks/{dataset}/{mask_type}/rate_{rate}/seed_{seed}.npz
```

同一轮比较必须读取同一测试 mask，不得在评估时重新随机生成。

第一轮创新比较保持：

* 相同 epoch；
* 相同 batch size；
* 相同 seed；
* 相同 optimizer；
* 相同 scheduler；
* 相同 early stopping；
* 相同输入长度；
* 相同预测长度；
* 相同数据；
* 相同测试 mask；
* 相同评价代码；
* 相同 checkpoint 选择规则。

必须同时报告：

* 新增参数量；
* 总参数量；
* 单 epoch 时间；
* 总训练时间；
* 推理时间；
* 峰值显存；
* FLOPs 或可靠的计算量近似。

---

## 9. 代码和实验可追溯性

每个实验必须有唯一 ID：

```text
{model}__{dataset}__{mask_type}__mr{rate}__seed{seed}__{timestamp}
```

每个正式实验必须保存：

```text
runs/<experiment_id>/
├── resolved_config.yaml
├── command.sh
├── env.txt
├── git_state.txt
├── code.diff
├── train.log
├── metrics.json
├── status.json
├── resource_usage.json
└── checkpoints/
```

任何关键配置变化必须在实验启动前记录：

```text
原值 → 新值 → 修改原因
```

不得覆盖旧实验。

每个创新模块必须有独立开关。所有创新关闭时，必须恢复官方 TimePerceiver 行为。

未经用户允许：

* 不自动 commit；
* 不 push；
* 不删除分支；
* 不改写 Git 历史；
* 不覆盖用户未提交修改。

---

## 10. 结果判断和论文边界

项目当前优先级是实验质量，不是论文写作。

必须将当前结果与原论文报告值进行比较，并说明：

* 是否完全相同协议；
* 绝对差异；
* 相对差异；
* seed；
* 配置；
* 偏差原因；
* 可信程度。

不得仅根据单次最好结果作出结论。

单 seed 用于筛选创新；正式结论必须补充多 seed、统计检验和完整消融。

实验结果未达到预期时：

1. 检查实现；
2. 检查公平性；
3. 检查数据泄漏；
4. 分析训练曲线和误差；
5. 分析参数量、速度和显存；
6. 判断模块无效、优化失败还是数据不匹配；
7. 提出下一版最小修改；
8. 继续迭代。

在证据不足时，不得声称：

* 达到 SCI 二区水平；
* 保证二区或三区录用；
* 一定优于所有模型；
* 已解决所有缺失机制；
* 能够保证商业收入。

只有当实验质量、统计稳定性、方法创新和强基线比较达到足够水准后，才开始论文写作和期刊选择。

## 实验全过程记录（2026-09-10 新增）

遵守全局 AGENTS.md 的“实验全过程简明记录”规则。本项目唯一记录文件为：
`/mnt/data/zza/code/badandgood/timeperceiver_meta_uq.md`。

全过程中与文档不一致的所有实际改动均须及时记录：原要求、修改点、原因、好的地方、坏的地方、验证证据及下一步。保持简短，阶段结束同步更新，保留负结果和更正历史，不替代正式实验产物。
