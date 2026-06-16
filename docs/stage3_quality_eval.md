# 阶段3：三维质量评估 — 原理框架

> 面试目标：被问"你的数据质量评估是怎么做的"时，能讲清 PPL/IFD/Embedding 三者的计算原理、协同机制、为什么三者互补。

---

## 一、模块定位

```
阶段2(synthesized.jsonl) → 阶段3(quality_eval.py) → scored.jsonl → 阶段4(课程学习排序)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
               PPL 流畅度     IFD 指令难度    Embedding 多样性
               (通不通顺)     (难不难)       (多不多样)
```

核心任务：对阶段2合成的 505 条指令数据，从**流畅度、难度、多样性**三个维度打分，为阶段4的课程学习排序提供依据。

---

## 二、维度1：PPL 流畅度

### 一句话

用 N-gram 语言模型计算文本困惑度，PPL 越低 = 语言越通顺。

### 计算方式

复用阶段1的 `NgramModel`（手写 trigram 回退模型，jieba 分词）：

```
PPL = exp(-1/N × Σ log P(w_i | context_i))
```

对每条数据的 `instruction + output` 拼接文本计算 PPL。

### 为什么阶段3还要再算一次 PPL

阶段1的 PPL 用于筛选脏数据（heuristic 过滤），阶段3的 PPL 用于检测**合成数据**的流畅度：
- LLM 生成的回答可能出现语言不通顺
- 作为"基础门槛"，PPL 异常高的数据不计入后续课程学习

### 参数

| 参数 | 值 | 理由 |
|------|-----|------|
| n-gram 阶数 | 3 | trigram 的计算成本与区分力平衡点 |
| 分词方式 | jieba | 中文分词 jieba 比按字粒度更准确 |

### 面试追问："PPL 用自训练数据，结果偏低，你怎么看？"

> "确实，用待评估数据自训练 N-gram 模型会导致 PPL 整体偏低（mean=3.9）——因为模型记住了自己的分布。但作为相对排序指标，同分布内比较仍然完全有效：PPL=6.0 的记录确实比 PPL=2.2 的记录更不通顺。生产环境改进方向是用外部干净语料（如维基百科）独立训练哨兵模型。"

---

## 三、维度2：IFD 指令难度（核心亮点）

### 一句话

IFD (Instruction-Following Difficulty) = loss(answer | instruction) / loss(answer)。

数值越小 = 指令对答案的引导力越强 = 越"简单"。数值越大 = 指令无法有效引导答案 = 越"困难"。

### 核心公式

```
IFD = PPL(answer | instruction) 的等效 loss / PPL(answer) 的等效 loss

具体：
loss_cond = CrossEntropy(answer tokens, 给定 instruction)
loss_uncond = CrossEntropy(answer tokens, 无 instruction)
IFD = loss_cond / loss_uncond
```

### 直觉解释

| IFD 值 | 含义 | 例子 |
|--------|------|------|
| < 0.5 | 指令提供了极强引导，答案高度确定 | "将以下文字翻译成英文：你好" |
| 0.5~1.0 | 指令有引导作用，但答案有一定灵活性 | "分析当前电池技术的主要挑战" |
| ≈ 1.0 | 指令几乎无法引导答案，答案本身很通用 | "请写一篇文章"（写什么？） |
| > 1.0 | 指令反而干扰了答案预测 | 指令和答案领域不匹配 |

### 实现方式

用 GPT-2 124M 因果语言模型计算 token-level 交叉熵 loss：

1. **无条件 loss**：只喂 answer 给模型 → 计算自回归 loss
2. **条件 loss**：喂 `instruction + answer`，用 label masking 把 instruction 部分置 -100，只算 answer 部分的 loss
3. **IFD = loss_cond / loss_uncond**

### 为什么用 GPT-2 而不是更强模型

三个理由（面试时按序抛出）：

> **第一，IFD 只需要相对排序**。GPT-2 已经足够区分"指令对答案有没有引导力"，更强的模型只是让 loss 数值整体更小，排序结果高度相关——不值得为排序多花 50 倍推理成本。

> **第二，模型越小越诚实**。大模型见过训练数据，loss 可能受记忆效应影响；GPT-2 少见过互联网语料，loss 更直接反映文本本身的预测难度。

> **第三，工程成本**。124M 参数在 RTX 4060 上 20 秒跑完 505 条，同任务换 LLaMA-7B 需要 10 分钟——不匹配轻量评估管线。

### 面试追问："IFD 的理论上限在哪？"

> "IFD 本质上要求 loss(answer|instruction) ≤ loss(answer)，因为 instruction 提供了额外上下文，理应降低不确定性。IFD > 1 的情况说明 instruction 和 answer 之间存在不一致——要么 instruction 没被遵循，要么 answer 答非所问。这正是 IFD 能捕捉到'指令-答案对齐性'的原因。"

---

## 四、维度3：Embedding 多样性

### 一句话

把所有指令用 sentence encoder 编码为向量，计算两两余弦相似度矩阵——均值越低 = 数据越多样。

### 计算方式

```
1. 句子编码：paraphrase-multilingual-MiniLM-L12-v2 → 505 × 384 矩阵
2. 余弦相似度矩阵：cosine_similarity(embeddings)
3. 取上三角（不含对角线）→ 计算均值
4. Diversity Score = 1 - mean_similarity
```

### 为什么是"宏观监控"指标

PPL 和 IFD 都是单条数据的分数（服务于课程学习排序），而 Embedding 多样性是**数据集级**的健康指标：
- 多样性高（如 0.73） → 合成策略有效，数据没有模板化
- 多样性持续下降 → 触发合成策略调整（提高温度、增加种子多样性）

### 面试追问："为什么不直接做 K-Means 聚类看类别分布？"

> "聚类需要预设 K 值，而在合成数据场景下我们不知道指令应该落在几类。余弦相似度矩阵是无监督、无参数的全局多样性指标——不需要知道'有几类'就能判断'像不像'。"

---

## 五、三维协同机制

三者是互补的，不是独立的：

```
PPL  → 基础门槛："通不通"（过滤底限）
IFD  → 难度分级："难不难"（课程学习排序依据）
Diversity → 全局监控："多不多样"（防模式坍塌）
```

| 场景 | PPL | IFD | 判断 |
|------|-----|-----|------|
| 流畅但不相关 | 低 ✓ | 高 ✗ | PPL 低说明答案本身通顺，但 IFD 高说明答案和指令没关系 → 丢弃 |
| 有瑕疵但有效 | 高 ✗ | 低 ✓ | 例如代码片段有格式问题但逻辑正确 → 保留，标记 |
| 既不通又不难 | 高 ✗ | 高 ✗ | 纯垃圾，丢弃 |

**面试时一句话总结**："PPL 能告诉你答案写得像不像人话，IFD 能告诉你答案是不是在回答这个问题，Diversity 能告诉你是不是所有问题都是一个模子刻出来的。"

---

## 六、代码架构

```
stages/stage3_quality_eval.py

├── EvalConfig       # 配置加载（config.yaml quality_eval 节）
├── PPLScorer        # 复用阶段1 NgramModel，训练 + 评分
│   └── NgramModel   # 从 stage1_clean_pipeline 导入
├── IFDScorer        # GPT-2 124M GPU/CPU 推理
│   ├── load()       # AutoModelForCausalLM + tokenizer
│   └── score()      # 无条件 loss + 条件 loss（label masking）
├── DiversityScorer  # sentence-transformers 编码 + 余弦相似度
│   └── score()      # 返回全局 diversity_score
└── main()           # 串联 + 保存 scored.jsonl
```

---

## 七、面试速记卡

| 问题 | 回答要点 |
|------|----------|
| 三维评估怎么协同？ | PPL=基础门槛，IFD=难度分级，Diversity=全局监控 |
| IFD 公式？ | loss(answer\|instr) / loss(answer)，值越小越简单 |
| 为什么用 GPT-2？ | 1)相对排序够用 2)越小学越诚实 3)RTX 4060 20秒 |
| IFD>1 说明什么？ | instruction 和 answer 不相关，或 answer 答非所问 |
| Diversity 怎么算？ | sentence embedding → 余弦相似度矩阵 → 1-均值 |
| 三个维度能互相替代吗？ | 不能。PPL 测通顺，IFD 测因果，Diversity 测全局分布 |
