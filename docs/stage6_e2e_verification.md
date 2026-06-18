# 阶段6：端到端集成验证 — 原理框架

> 面试目标：被问"你的数据飞轮有端到端验证吗"时，能讲清对照实验设计和实验结果。

---

## 一、模块定位

```
阶段4/5(课程序列+反馈闭环) → 阶段6(GPU训练+评测) → 生成评估信号 → 回到阶段5

                前五阶段数据管线 ←─→ 阶段6 训练验证
                     ↓                    ↓
              数据准备好了           数据到底行不行？
```

核心任务：用前五阶段产出的课程学习数据，在 MiniMind 框架上实际训练模型，通过对照实验**量化证明**数据飞轮每个环节带来的增益。

---

## 二、对照实验设计

### 2.1 核心假设

- **H1**：课程学习（分带混洗/β退火）比随机 shuffle 有更低的 eval loss
- **H2**：β退火在中等难度层比分带混洗有更好的精度
- **H3**：清洗后的数据训练的模型，比用未清洗数据训练的模型 loss 更低

### 2.2 三组对照

| 实验组 | 数据源 | 排序方式 | 验证什么 |
|--------|--------|----------|----------|
| A. Baseline | curriculum_band.jsonl | **random shuffle** | 基准线：没有课程学习 |
| B. Band-Shuffle | curriculum_band.jsonl | 按 band1→2→3 顺序 | 阶段4策略A效果 |
| C. Beta-Annealing | curriculum_beta.jsonl | 按 sampling_weight 降序 | 阶段4策略B效果 |

### 2.3 数据格式转换

```
AquilaLM 格式:           MiniMind SFT 格式:
{"instruction": "...",   {"conversations": [
 "output": "...",          {"role": "user", "content": "..."},
 "ifd_score": 0.345}       {"role": "assistant", "content": "..."}
                         ]}
```

### 2.4 控制变量

| 变量 | 固定值 | 理由 |
|------|--------|------|
| 预训练起点 | MiniMind2 104M checkpoint (hidden=768, 16 layers) | 同一起点 |
| Epoch | 2 | 足够看 loss 趋势，不过度训练 |
| Batch size | 16 | RTX 4060 8GB 适配 |
| Learning rate | 1e-6 | MiniMind 默认，保守微调 |
| Max seq len | 340 | MiniMind 默认，94%数据在此范围内 |
| dtype | bfloat16 | RTX 4060 支持 |

---

## 三、评估指标体系

### 3.1 训练期间

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| SFT loss | 每 100 step 打印 | 越低越好，看收敛速度 |
| logits loss | loss - aux loss | 排除 MoE 辅助 loss |
| LR schedule | cosine decay | 确认学习率一致 |

### 3.2 测试集评估

用阶段2原始的 505 条合成数据中的固定 50 条作为 hold-out eval set：

| 指标 | 计算方式 | 组间对比 |
|------|----------|----------|
| Eval loss | 所有 eval 样本的 mean loss | A vs B vs C |
| Per-band loss | 按 IFD 分3层的 mean loss | 看难度层差异 |
| Per-type loss | 按 task_type 分类的 mean loss | 看任务级差异 |

### 3.3 数据增益验证（可选，清洗前 vs 后）

```
MiniMind 默认 SFT 数据 (pretrain_hq.jsonl) → 训练 → loss_old
AquilaLM 课程学习数据 → 训练 → loss_new
Δ = loss_old - loss_new → 数据飞轮的增益量化
```

---

## 四、训练脚本设计

### 4.1 复用 MiniMind 训练框架

```
self_minimind_for_reviewing/
├── model/model.py      → 模型架构（MokioMind）
├── model/MokioModel.py  → HuggingFace wrapper
├── dataset/lm_dataset.py → SFT 数据集类
└── trainer/train_full_sft.py → SFT 训练脚本（复用核心循环）
```

**不修改训练代码**，只提供不同格式的输入数据 + 不同命令行参数。保持控制变量纯净。

### 4.2 阶段6脚本

```python
stages/stage6_e2e.py

├── data_converter()      # AquilaLM → MiniMind conversations 格式
├── run_experiment()      # 调 train_full_sft.py，传参输出 loss 日志
├── eval_wrapper()        # 加载 SFT checkpoint → 跑 eval loss
└── report()              # 汇总三组结果 + 生成对比表
```

### 4.3 执行流程

```bash
# 1. 数据转换（AquilaLM → MiniMind 格式）
python stages/stage6_e2e.py --convert-only

# 2. 三组对照实验（batch脚本）
python stages/stage6_e2e.py --experiment baseline
python stages/stage6_e2e.py --experiment band
python stages/stage6_e2e.py --experiment beta

# 3. 评估
python stages/stage6_e2e.py --eval-all

# 4. 输出报告
python stages/stage6_e2e.py --report
```

---

## 五、期望结果

### 理想情况下

| 假设 | 期望结果 | 面试解读 |
|------|----------|----------|
| H1: 课程学习 > 随机 | B/C 的 eval loss < A | "课程学习确实比随机 shuffle 好" |
| H2: β退火 > 分带 | C 的 band2 loss < B 的 band2 | "β退火在中等难度避免硬切换震荡" |
| H3: 清洗增益 | AquilaLM SFT loss < 原始数据 SFT loss | "数据飞轮全链路有效" |

### 坦诚地

505 条指令数据量较小，可能看不到统计显著的差异。但这本身也是一个诚实的结果——"数据量小时，课程学习效果不显著"同样是面试时的工程洞见。

---

## 六、已知局限与诚实说

- [ ] **505 条数据偏少**：足够验证流程正确性，但未必产生统计显著的组间差异
- [ ] **单次实验噪音**：每组仅一次 run，受随机种子影响。生产环境应每组 3 次取均值
- [ ] **模型偏小**：104M 参数模型，IFD 难度差距在大模型上会体现得更明显
- [ ] **无 benchmark 评测**：纯 loss 对比，没有 CMMLU/CEval 等下游任务评测

---

## 七、面试速记卡

| 问题 | 回答要点 |
|------|----------|
| 端到端验证做了什么？ | 三组对照（随机 vs band vs β），控制变量，同起点同参数 |
| 怎么证明课程学习有效？ | 对比 A/B/C 的 eval loss 和 per-band loss |
| 为什么不用更多数据？ | "505条足够验证流程。量级增大是工程问题，方向性验证是科学问题" |
| 如果三组没差异？ | "505 条数据量小，加上 104M 模型天花板低。但流程已跑通，换上更大模型和 10x 数据即可看到差异——这就是为什么大厂需要数据飞轮" |
