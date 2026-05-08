# AquilaLM 项目作品集附件（该文件用于简历投递）

## 1. 项目概述
AquilaLM 是一个面向 **大语言模型训练全链路** 的开源项目，覆盖从预训练（Pretrain）到监督微调（SFT）、偏好优化（DPO）以及强化学习优化（PPO/GRPO）的完整流程。项目强调“可读、可改、可扩展”，适合作为 LLM 训练工程能力与算法理解能力的综合展示。

- 项目定位：轻量但完整的 LLM 训练栈
- 技术目标：在可控复杂度下复现现代 LLM 训练核心组件
- 适用场景：教学研究、方案验证、个人作品集展示

---

## 2. 核心能力一览
- **端到端训练流程**：Pretrain → SFT → DPO → PPO / GRPO
- **模型结构能力**：Decoder-Only Transformer、GQA、RoPE（含 YaRN 扩展）、可选 MoE
- **参数高效微调**：LoRA 低秩适配器注入/保存/加载
- **数据处理能力**：Pretrain / SFT / DPO / RLAIF 多种数据管线与 loss mask 设计
- **训练工程能力**：AMP 混合精度、梯度累积、梯度裁剪、DDP 分布式、断点续训
- **实验管理能力**：Checkpoint 管理、可选 swanlab 记录训练过程

---

## 3. 仓库结构与模块职责

```text
AquilaLM/
├── model/
│   ├── model.py                 # 模型配置、注意力、RoPE、MoE、CausalLM
│   └── model_lora.py            # LoRA 注入与权重存取
├── dataset/
│   └── lm_dataset.py            # Pretrain/SFT/DPO/RLAIF 数据集实现
├── trainer/
│   ├── train_pretrain.py        # 预训练
│   ├── train_full_sft.py        # 全量监督微调
│   ├── train_dpo.py             # 直接偏好优化
│   ├── train_ppo.py             # PPO 强化学习优化
│   ├── train_grpo.py            # GRPO 强化学习优化
│   └── trainer_utils.py         # 分布式、学习率、checkpoint、模型初始化等工具
├── docs/
│   ├── README.md
│   └── GETTING_STARTED.md
├── main.py
├── pyproject.toml
└── requirements.txt
```

---

## 4. 模型与算法实现亮点

### 4.1 主体模型（`model/model.py`）
- 配置层：统一管理模型超参数与推理/训练开关
- Backbone 层：Decoder-Only 主体结构，支持 KV Cache
- CausalLM 层：封装 logits 计算、训练损失与生成接口

### 4.2 注意力与位置编码
- **GQA（Grouped-Query Attention）**：Query 头与 KV 头解耦，提高效率
- **RoPE**：旋转位置编码
- **YaRN 扩展**：长上下文推理时的 RoPE 缩放策略
- **Flash Attention 路径**：在可用条件下走高效实现

### 4.3 FFN 与 MoE
- 标准门控 FFN（SwiGLU 风格）
- 可选 **MoE 路由专家结构**，包含辅助负载均衡损失

### 4.4 LoRA（`model/model_lora.py`）
- 对线性层进行低秩增量注入：`W0x + BAx`
- 支持独立保存/加载 LoRA 权重，便于轻量化迁移与快速切换

---

## 5. 数据管线设计（`dataset/lm_dataset.py`）

项目内置四类核心数据集实现：

1. **PretrainDataset**
   - 输入：原始文本
   - 目标：标准 next-token 预测
   - 处理：BOS/EOS 拼接、padding、ignore_index 处理

2. **SFTDataset**
   - 输入：多轮对话
   - 目标：仅监督 assistant 回复部分
   - 关键：通过 token span 生成稀疏标签（非回复区域置 -100）

3. **DPODataset**
   - 输入：`chosen/rejected` 偏好对
   - 目标：为 DPO 训练提供对比样本与 loss mask

4. **RLAIFDataset**
   - 输入：对话与参考答案
   - 目标：为 PPO/GRPO 在线 rollout 提供 prompt 与评价上下文

---

## 6. 训练流程与脚本

### 6.1 预训练 Pretrain
```bash
python trainer/train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl
```

### 6.2 全量监督微调 SFT
```bash
python trainer/train_full_sft.py --data_path ../dataset/sft_mini_512.jsonl
```

### 6.3 直接偏好优化 DPO
```bash
python trainer/train_dpo.py --data_path ../dataset/dpo.jsonl
```

### 6.4 强化学习优化 PPO / GRPO
```bash
python trainer/train_ppo.py --data_path ../dataset/rl.jsonl
python trainer/train_grpo.py --data_path ../dataset/rl.jsonl
```

---

## 7. 强化学习阶段能力展示

### PPO（`train_ppo.py`）
- 采用 Actor / Old Actor / Critic / Reference / Reward Model 五模型协作
- 包含 PPO-Clip、价值函数损失、KL 约束
- 支持基于格式与内容的奖励组合

### GRPO（`train_grpo.py`）
- 无 Critic 架构，通过组内多采样奖励标准化估计优势
- 计算更简化，参数更少，适合高性价比偏好优化实验

---

## 8. 训练工程与稳定性设计
- DDP 分布式训练初始化与主进程日志控制
- AMP 混合精度（bfloat16/float16）
- 梯度累积 + 梯度裁剪
- Cosine 学习率调度
- Checkpoint 与 Resume 恢复（支持训练中断续跑）
- 统一工具函数封装，降低训练脚本复杂度

---

## 9. 环境与依赖
- Python `>=3.11`（`pyproject.toml`）
- 关键依赖：`torch`、`transformers`、`numpy`、`pandas`

安装示例：
```bash
pip install -r requirements.txt
```

---

## 10. 作为作品集可体现的能力
- LLM 训练全流程工程落地能力（预训练到 RL 对齐）
- Transformer 核心模块与训练目标函数的实现理解
- 多范式优化方法（监督学习 + 偏好学习 + 强化学习）
- 训练稳定性与可复现实验工程实践
- 可扩展架构设计（MoE、LoRA、Reward Model 接入）

---

## 11. 相关文档
- 项目主页说明：`README.md`
- 文档导览：`docs/README.md`
- 入门说明：`docs/GETTING_STARTED.md`

---

## 12. 总结
AquilaLM 是一个覆盖 **Pretrain/SFT/DPO/PPO/GRPO** 的轻量化 LLM 训练项目，具备从模型结构实现、数据管线设计到分布式训练与偏好优化的完整工程闭环，能够直接作为大模型训练方向的作品集项目展示。
