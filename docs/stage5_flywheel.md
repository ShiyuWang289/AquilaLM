# 阶段5：反馈闭环调度器 — 原理框架

> 面试目标：被问"你的数据飞轮是怎么设计反馈闭环的"时，能讲清信号来源→权重矩阵→参数调整的完整链路。

---

## 一、模块定位

```
阶段4(curriculum.jsonl) → 阶段5(flywheel.py) → 阶段6(训练+评测) → 阶段2(重新合成)
       ↑                                                              │
       └────────────────── 反馈信号 ←──────────────────────────────────┘
```

核心任务：**让下游训练评估结果反向驱动上游数据策略参数**，形成"合成→评估→训练→评测→调整合成"的自进化循环。

阶段5不需要等阶段6训练完成就能搭建——纯逻辑代码 + mock 信号先跑通，预留接口给真实信号。

---

## 二、三大组件

### 1. SignalNormalizer：信号归一化器

将 8 个维度的原始评估指标转化为统一的 0~1 偏差分数。

| 信号 | 语义 | 方向 | 基线 | 权重 |
|------|------|------|------|------|
| code_accuracy | 代码任务精度 | higher_better | 0.60 | 0.25 |
| reasoning_accuracy | 推理任务精度 | higher_better | 0.55 | 0.15 |
| diversity_score | 指令多样性 | higher_better | 0.70 | 0.15 |
| loss_volatility | 训练 Loss 波动 | lower_better | 0.10 | 0.10 |
| dialogue_quality | 对话质量 | higher_better | 0.60 | 0.10 |
| qa_accuracy | 知识问答精度 | higher_better | 0.65 | 0.10 |
| effective_ratio | 有效数据占比 | higher_better | 0.85 | 0.10 |
| ifd_gap | SI/EI难度差距 | higher_better | 0.20 | 0.05 |

### 2. DecisionEngine：决策引擎

核心公式：
```
Δparam_j = Σ_i (signal_deviation_i × weight_ij)
```

权重矩阵设计原则：
- 每个参数最多受 2~3 个信号影响（防过度耦合）
- 主要信号权重 ≥ 0.5（让因果关系可解释）

**面试时讲一个具体例子**：

> "如果对话质量跌到 0.38（基线 0.60，偏差 0.37），DecisionEngine 检测到 `dialogue_quality` 的偏差超过触发阈值 0.05，自动触发 `multi_turn_ratio` 调高 27%——因为权重矩阵中 `dialogue_quality` 对多轮对话进化比例的影响权重是 0.8。"

### 3. ParameterAdjuster：参数调整器

支持三种参数类型：
- **ratio**（0~1 小数）：task_type 配比
- **count**（整数）：seed 数量、进化数量
- **threshold**（阈值）：PPL/Jaccard 阈值

所有调整在 config.yaml 的上下限内 clamp，单轮调整不超过 30%，自动备份原配置。

---

## 三、12 个可反向驱动的参数

| 阶段 | 参数 | 受什么信号驱动 |
|------|------|---------------|
| 种子筛选 | `seed_max_ppl` | 有效数据占比 + Loss 波动 |
| 种子筛选 | `max_seeds` | 多样性 + 有效数据占比 |
| Self-Instruct | `si_task_types` 配比 | 对应任务精度 + 多样性 |
| Self-Instruct | `temperature_creative` | 多样性 + 有效数据占比 |
| Evol-Instruct | `ei_max_evolve` | IFD 差距 + 有效数据占比 |
| Evol-Instruct | `evolution_types` 配比 | 推理/对话精度 + 代码精度 |
| 后过滤 | `postfilter_jaccard_threshold` | 多样性 + 有效数据占比 |
| 后过滤 | `postfilter_consistency_threshold` | 有效数据占比 + Loss 波动 |
| 规则过滤 | `min_text_length` | 有效数据占比 + Loss 波动 |
| 去重 | `jaccard_threshold` | 多样性 + 有效数据占比 |
| N-gram | `ppl_threshold` | Loss 波动 + 有效数据占比 |

---

## 四、四种模拟场景

因为阶段6训练还没做，阶段5用 mock 信号模拟四种典型失效模式：

| 场景 | 特征 | 触发的关键调整 |
|------|------|---------------|
| **baseline** | 略微偏差，多数正常 | 几乎全部 hold |
| **drift** | 多样性坍塌 + 对话崩 | ↑温度 + ↑多轮对话进化 + ↓去重阈值 |
| **hard** | 代码/推理崩，简单任务正常 | ↑代码/推理采样比例 + ↑进化量 |
| **noise** | Loss 剧烈震荡，一半数据无效 | ↑种子PPL放宽 + ↑一致性阈值 + ↓PPL阈值收紧 |

### 真实信号注入

即使没有阶段6训练，阶段5仍能从已有产物中提取两个真实信号：
- **ifd_gap**：从 scored.jsonl 的 SI/EI IFD 差距 → 当前 = 0.185
- **effective_ratio**：synthesized/cleaned 的比率 → 当前 = 0.33

---

## 五、代码架构

```
stages/stage5_flywheel.py

├── FlywheelConfig         # 配置加载（config.yaml flywheel 节）
├── SignalNormalizer       # 8维信号 + 归一化 + 4种mock场景
│   ├── normalize()        # raw → 0~1 偏差分数
│   └── apply_mock_scenario()  # baseline/drift/hard/noise
├── DecisionEngine         # 权重矩阵（13参数 × 8信号）
│   ├── decide()           # Σ(deviation × weight) → 调整量
│   └── WEIGHT_MATRIX      # 因果关系矩阵
├── ParameterAdjuster      # 调整量 → 参数值 + config更新
│   ├── apply()            # ratio/count/threshold 三类型处理
│   ├── _update_config()   # 自动备份 + yaml 写入
│   └── revert()           # 回退到最新备份
└── main()                 # 串联 + mock场景支持 + dry-run
```

---

## 六、面试速记卡

| 问题 | 回答要点 |
|------|----------|
| 反馈闭环怎么设计的？ | 8维信号归一化 → 13参数权重矩阵 → 调整量 clamp → config更新+备份 |
| 为什么用权重矩阵而不是 if-else？ | 每个参数受 2-3 个信号影响，if-else 无法处理信号叠加 |
| 调整量怎么 clamp？ | 单轮 ≤30%，上下限硬约束，防过调震荡 |
| 没有训练怎么办？ | 4种 mock 场景验证逻辑 + 2个从已有产物提取的真实信号 |
| 和 SPIN/Arena Learning 的区别？ | SPIN=自博弈不需要手动调参，阶段5=半自动策略级调整。面试时说未来方向是 SPIN 全自动 |
| 一次闭环迭代的具体流程？ | 训练评测→信号归一化→权重决策→参数更新→重新合成数据→重新训练→再评测 |
