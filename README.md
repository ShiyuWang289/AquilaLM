# AquilaLM — 数据闭环驱动的轻量级 LLM 训练栈

> "模型能力的上限由数据质量决定。"

AquilaLM 是一个以**数据闭环**为核心驱动的轻量级大模型训练栈，聚焦于后训练（Post-Training）阶段的数据工程与对齐。通过构建"数据清洗→指令合成→质量评估→课程学习→下游评估反馈"的自进化循环，用数据质量的持续提升带动模型能力的增长。

## 架构全景

```
                 ┌────────────────────────────┐
                 │   阶段5：下游评估与反馈      │
                 │   · 模型评估分数            │
                 │   · 反向驱动策略参数         │
                 └──────────────┬─────────────┘
                                │ (反向驱动)
     ┌──────────────────────────┼──────────────────────────┐
     │   阶段4：课程学习排序    │                          │
     │   · IFD 从易到难排序     │                          │
     └──────────────┬───────────┘                          │
                    │                                       │
     ┌──────────────┼──────────────────────────────────────┐
     │   阶段3：多维质量评估                                 │
     │   · PPL 流畅度 + IFD 指令难度 + Embedding 多样性     │
     └──────────────┬──────────────────────────────────────┘
                    │
     ┌──────────────┼──────────────────────────┐
     │   阶段2：指令合成                        │
     │   · Self-Instruct / Evol-Instruct       │
     │   · DPO 偏好数据自动构造                  │
     └──────────────┬──────────────────────────┘
                    │
     ┌──────────────┼──────────────┐
     │   阶段1：数据清洗             │
     │   · 规则过滤 → N-gram PPL   │
     │   · MinHash + LSH 去重      │
     └─────────────────────────────┘
```

## 项目结构

```
AquilaLM/
├── README.md
├── config.yaml                 # 全局配置（所有阈值集中管理）
├── requirements.txt
├── .gitignore
├── stages/                     # 各阶段独立脚本
│   ├── stage1_clean_pipeline.py    # ✅ 数据清洗
│   ├── stage2_instruction_synth.py # ✅ 指令合成
│   ├── stage3_quality_eval.py      # ⬜ 质量评估
│   ├── stage4_curriculum.py        # ⬜ 课程学习
│   └── stage5_flywheel.py          # ⬜ 反馈闭环
├── utils/                      # 共享工具库
│   ├── io.py                       # JSONL 读写
│   ├── profile.py                  # 数据探查
│   └── health.py                   # 健康度对比
├── docs/                       # 知识文档
│   ├── roadmap.md                  # 项目仪表盘（进度/边界/改进计划）
│   ├── architecture.md             # 架构全景
│   ├── stage1_cleaning.md          # 阶段1原理+实验报告
│   └── stage2_instruction_synth.md # 阶段2原理+面试速记卡
├── experiments/                # 实验记录
│   ├── stage1_grid_search.md       # 2×2网格搜索
│   └── stage2_synth_report.md      # 指令合成POC报告
└── data/samples/               # 小样本数据（演示用）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（下载 C4 中文语料或使用 sample）
# 完整数据下载见 data/README.md

# 3. 数据探查（先看分布再定阈值）
python stages/stage1_clean_pipeline.py --profile-only

# 4. 运行阶段1：数据清洗
python stages/stage1_clean_pipeline.py

# 5. 运行阶段2：指令合成（需设置 API Key）
export DEEPSEEK_API_KEY=sk-your-key-here
python stages/stage2_instruction_synth.py
```

## 实验证据

### 阶段1：数据清洗

在 C4 3679 条中文网页语料上完成 2×2 网格搜索，最终参数 PPL=30, Jaccard=0.60：

| 指标 | 清洗前 | 清洗后 | 变化 |
|------|--------|--------|------|
| 样本数 | 3679 | 3082 | -16% |
| 长度标准差 | 3425 | 1039 | **-70%** |
| 长度变异系数 | 2.86 | 1.14 | **-60%** |
| Bigram 多样性 | 0.193 | 0.263 | **+36%** |

### 阶段2：指令合成

基于 DeepSeek API 的 Self-Instruct + Evol-Instruct 两级合成，从 80 条种子生成了 505 条指令 + 40 对 DPO 偏好数据：

| 模块 | 输入 | 产出 | API 调用 | 费用 |
|------|------|------|----------|------|
| Self-Instruct | 80 种子 | 375 条指令 | 387 次 | ¥0.50 |
| Evol-Instruct | 150 候选 | 130 条进化指令 | ~150 次 | ~¥1.50 |
| DPO 偏好构造 | 40 对目标 | 40 对 | 63 次 | ~¥0.20 |
| **合计** | | **505 条 + 40 对** | | **¥2.20** |

类型覆盖：代码(83) / 推理(80) / 文本生成(73) / 对话(72) / 知识问答(63)。DPO 采用双策略——数学答案验证（正则提取，成功率 95%）+ LLM-as-Judge 兜底（解决代码执行判定成功率 5% 问题）。

## 技术栈

`PyTorch` `Transformers` `Jieba` `DataSketch` `Sentence-Transformers` `YaRN` `GQA` `FlashAttention` `DPO` `PPO`

## 关联项目

- [Law-Expert-7B](https://github.com/irisyu/Law-Expert-7B) — 领域化大模型工程实践（SFT→DPO→GRPO→部署）
- MiniMind — 轻量级 LLM 训练框架（Pretrain→SFT→RLHF）
