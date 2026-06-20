#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquilaLM 三维质量评估
=====================
PPL 流畅度 + IFD 指令难度 + Embedding 多样性

使用方法：
    python stage3_quality_eval.py              # 全流程
    python stage3_quality_eval.py --stage ppl  # 仅 PPL
    python stage3_quality_eval.py --stage ifd  # 仅 IFD
    python stage3_quality_eval.py --no-gpu     # 强制 CPU

输入：data/synthesized.jsonl (505条)
输出：data/scored.jsonl (505条 + ppl/ifd_score/diversity_score)

计算原理、协同机制、以及为什么这三者能覆盖质量的主要方面。
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_jsonl, save_jsonl

# 复用阶段1的 NgramModel
sys.path.insert(0, str(Path(__file__).parent))
from stage1_clean_pipeline import NgramModel


# ============================================================
# 0. 配置与日志
# ============================================================

@dataclass
class EvalConfig:
    """质量评估配置"""
    data_input: str = "data/synthesized.jsonl"
    data_output: str = "data/scored.jsonl"
    log_dir: str = "logs"

    # PPL 评分
    ppl_n: int = 3
    ppl_tokenizer: str = "jieba"

    # IFD 评分
    ifd_model_name: str = "uer/gpt2-chinese-cluecorpussmall"  # 中文 GPT-2
    ifd_batch_size: int = 4
    ifd_max_length: int = 512

    # Embedding 多样性
    embed_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embed_batch_size: int = 32

    # 设备
    device: str = "auto"  # auto | cuda | cpu

    # 输出字段
    output_ppl: str = "ppl"
    output_ifd: str = "ifd_score"
    output_embed_vec: str = "embedding"  # 仅在内存中使用，不写入文件


def load_config(config_path: str = "config.yaml") -> EvalConfig:
    """从 config.yaml 加载配置，命令行参数覆盖"""
    cfg = EvalConfig()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        qc = raw.get("quality_eval", {})
        if qc:
            cfg.ppl_n = qc.get("ppl_n", cfg.ppl_n)
            cfg.ppl_tokenizer = qc.get("ppl_tokenizer", cfg.ppl_tokenizer)
            cfg.ifd_model_name = qc.get("ifd_model_name", cfg.ifd_model_name)
            cfg.ifd_batch_size = qc.get("ifd_batch_size", cfg.ifd_batch_size)
            cfg.ifd_max_length = qc.get("ifd_max_length", cfg.ifd_max_length)
            cfg.embed_model_name = qc.get("embed_model_name", cfg.embed_model_name)
            cfg.embed_batch_size = qc.get("embed_batch_size", cfg.embed_batch_size)
            cfg.device = qc.get("device", cfg.device)
            paths = raw.get("paths", {})
            cfg.data_output = qc.get("output", cfg.data_output)
            if qc.get("input"):
                cfg.data_input = qc["input"]
    return cfg


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("AquilaLM-QE")
    logger.setLevel(logging.DEBUG)
    # 文件 handler
    fh = logging.FileHandler(os.path.join(log_dir, "quality_eval.log"),
                             encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


# ============================================================
# 1. PPL 流畅度评分器
# ============================================================

class PPLScorer:
    """
    用阶段1的 NgramModel 对指令+回答拼接文本计算 PPL。
    原理：PPL 越低 = 语言越流畅、越符合自然语言分布。

    与阶段1的 PPL 不同：阶段1是对单条语料打分（筛垃圾），
    阶段3是对指令+回答合成数据打分（测流畅度），但底层模型可复用。
    """

    def __init__(self, n: int = 3, tokenizer_type: str = "jieba"):
        self.n = n
        self.tokenizer_type = tokenizer_type
        self.model: Optional[NgramModel] = None

    def train(self, texts: List[str], logger: logging.Logger) -> None:
        """在待评估数据上训练 N-gram 模型（自训练）"""
        self.model = NgramModel(n=self.n, tokenizer_type=self.tokenizer_type)
        self.model.train(texts)
        # 注：已知局限——用待评估数据自训练，PPL 值整体偏低。
        # 生产环境应用外部干净语料独立训练。
        # 但这里作为相对排序指标（同分布内比较）仍然有效。

    def score(self, data: List[Dict], logger: logging.Logger) -> List[Dict]:
        """对每条数据计算 PPL，返回带 ppl 字段的新列表"""
        if self.model is None:
            raise RuntimeError("PPLScorer 未训练，请先调用 train()")

        logger.info("  [PPL] 开始评分...")
        results = []
        ppl_values = []

        for item in tqdm(data, desc="PPL 评分"):
            # 拼接 instruction + output 作为待评文本
            instr = item.get("instruction", "")
            output = item.get("output", "")
            text = f"{instr}\n{output}" if instr and output else (instr or output)

            ppl = self.model.perplexity(text)
            ppl_values.append(ppl)

            item_copy = dict(item)
            item_copy["ppl"] = round(ppl, 2)
            results.append(item_copy)

        finite = [p for p in ppl_values if p != float('inf')]
        logger.info(f"  [PPL] 完成: mean={np.mean(finite):.1f}, "
                    f"median={np.median(finite):.1f}, "
                    f"min={min(finite):.1f}, max={max(finite):.1f}")
        return results


# ============================================================
# 2. IFD 指令难度评分器（核心亮点）
# ============================================================

class IFDScorer:
    """
    用小型因果语言模型计算 IFD (Instruction-Following Difficulty)。

    IFD = loss(answer | instruction) / loss(answer)

    - loss(answer | instruction)：将 instruction+answer 拼接后，
      计算模型在 answer 部分上的交叉熵 loss
    - loss(answer)：仅喂 answer 给模型，计算 loss

    直觉：
    - IFD < 1：指令提供了有效引导，模型看到指令后预测答案更确定 → 简单指令
    - IFD ≈ 1：指令对答案预测帮助很小 → 困难指令
    - IFD > 1：指令反而干扰了答案预测（罕见，通常说明指令和答案不一致）

    - 为什么用 GPT-2 而不用更强的模型？
      答：IFD 只需要相对排序，GPT-2 已能反映"指令对答案的引导力"。
      更大的模型反而增加推理成本，且排序结果高度相关。
    - IFD 的计算为什么不直接用 PPL 做？
      答：PPL 测的是"文本本身是否通顺"，IFD 测的是"指令和答案的因果关系"。
      一个不通顺的答案 PPL 很高，但指令可能很好——PPL 单独无法区分。
    """

    def __init__(self, model_name: str = "uer/gpt2-chinese-cluecorpussmall",
                 device: str = "cuda", batch_size: int = 4,
                 max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def load(self, logger: logging.Logger) -> None:
        """加载 GPT-2 模型和分词器"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"  [IFD] 加载模型 {self.model_name} (device={self.device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # GPT-2 没有 pad_token，用 eos_token 代替
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"  [IFD] 模型加载完成: {params_m:.0f}M 参数")

    def _compute_loss(self, texts: List[str],
                      logger: logging.Logger) -> List[float]:
        """
        计算每段文本的交叉熵 loss。
        使用模型自回归 loss，对 batch 中每个样本独立计算。
        """
        import torch

        losses = []
        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc="  IFD loss 计算"):
            batch_texts = texts[i:i + self.batch_size]

            # tokenize
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.model(**enc, labels=enc["input_ids"])
                # outputs.loss 是 batch 平均 loss
                batch_loss = outputs.loss.item()

            losses.append(batch_loss)

        return losses

    def _compute_conditional_loss(self, instructions: List[str],
                                  answers: List[str],
                                  logger: logging.Logger) -> List[float]:
        """
        计算 loss(answer | instruction)，逐条返回（长度与输入一致）。

        方法：将 instruction + answer 拼接，用 label masking 令模型
        只在 answer token 上计算 loss（instruction 部分 label = -100）。
        """
        import torch

        losses = []
        for i in tqdm(range(0, len(instructions), self.batch_size),
                      desc="  IFD 条件 loss"):
            batch_inst = instructions[i:i + self.batch_size]
            batch_ans = answers[i:i + self.batch_size]

            for inst, ans in zip(batch_inst, batch_ans):
                full_text = inst + "\n" + ans

                enc = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = enc["input_ids"].to(self.device)

                # tokenize 单独的 instruction 以确定 answer 起点
                inst_enc = self.tokenizer(
                    inst,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                inst_len = inst_enc["input_ids"].shape[1]

                # label: instruction 部分 = -100，answer 部分保留原 token id
                labels = input_ids.clone()
                labels[:, :inst_len] = -100

                with torch.no_grad():
                    outputs = self.model(input_ids, labels=labels)
                    losses.append(outputs.loss.item())

        return losses

    def score(self, data: List[Dict], logger: logging.Logger) -> List[Dict]:
        """
        对每条数据计算 IFD 分数。

        返回带 ifd_score 字段的数据列表。
        """
        if self.model is None:
            raise RuntimeError("IFDScorer 未加载，请先调用 load()")

        logger.info("  [IFD] 开始评分...")

        instructions = [d.get("instruction", "") for d in data]
        answers = [d.get("output", "") for d in data]

        # 1. 计算无条件 loss：P(answer)
        logger.info("  [IFD] 第1步：计算无条件 loss P(answer)...")
        uncond_losses = []
        for i in tqdm(range(0, len(answers), self.batch_size),
                      desc="  IFD 无条件 loss"):
            batch = answers[i:i + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            import torch
            with torch.no_grad():
                # 逐样本精确计算 loss（而不是 batch 平均）
                for j in range(enc["input_ids"].shape[0]):
                    single_ids = enc["input_ids"][j:j+1]
                    single_labels = single_ids.clone()
                    outputs = self.model(single_ids, labels=single_labels)
                    uncond_losses.append(outputs.loss.item())

        # 2. 计算条件 loss：P(answer | instruction)
        logger.info("  [IFD] 第2步：计算条件 loss P(answer|instruction)...")
        cond_losses = self._compute_conditional_loss(instructions, answers, logger)

        # 3. 计算 IFD = loss_cond / loss_uncond
        results = []
        ifd_values = []

        for idx, item in enumerate(data):
            l_cond = cond_losses[idx] if idx < len(cond_losses) else float('inf')
            l_uncond = uncond_losses[idx] if idx < len(uncond_losses) else float('inf')

            if l_uncond > 1e-8:
                ifd = l_cond / l_uncond
            else:
                ifd = float('inf')

            ifd_values.append(ifd)
            item_copy = dict(item)
            item_copy["ifd_score"] = round(ifd, 4)
            item_copy["loss_cond"] = round(l_cond, 4)
            item_copy["loss_uncond"] = round(l_uncond, 4)
            results.append(item_copy)

        finite = [v for v in ifd_values if v != float('inf') and not math.isnan(v)]
        logger.info(f"  [IFD] 完成: mean={np.mean(finite):.4f}, "
                    f"median={np.median(finite):.4f}, "
                    f"min={min(finite):.4f}, max={max(finite):.4f}")
        logger.info(f"  [IFD] IFD<1 (简单): {sum(1 for v in finite if v < 1)}条, "
                    f"IFD≥1 (困难): {sum(1 for v in finite if v >= 1)}条")

        return results


# ============================================================
# 3. Embedding 多样性评分
# ============================================================

class DiversityScorer:
    """
    用 sentence-transformers 将指令编码为向量，计算数据集整体多样性。

    核心指标：两两余弦相似度的均值。均值越低 = 数据越多样。

    这是**宏观监控指标**，不针对单条数据打分。当多样性下降时，
    说明合成数据可能出现了模板化/模式坍塌，需要调整合成策略。
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def load(self, logger: logging.Logger) -> None:
        """加载 sentence-transformers 模型"""
        from sentence_transformers import SentenceTransformer

        logger.info(f"  [DIV] 加载 Embedding 模型 {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        logger.info("  [DIV] 模型加载完成")

    def score(self, data: List[Dict], logger: logging.Logger) -> Dict:
        """
        计算数据集整体多样性。

        返回 dict:
        - diversity_score (float): 1 - mean_pairwise_cosine_sim
        - mean_similarity (float): 平均余弦相似度
        - median_similarity (float): 中位余弦相似度
        - std_similarity (float): 相似度标准差
        - embedding_dim (int): 向量维度
        """
        if self.model is None:
            raise RuntimeError("DiversityScorer 未加载，请先调用 load()")

        logger.info("  [DIV] 开始编码指令...")
        instructions = [d.get("instruction", "") for d in data]

        # 批量编码
        embeddings = self.model.encode(
            instructions,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

        logger.info(f"  [DIV] 编码完成: {embeddings.shape[0]} x {embeddings.shape[1]}")

        # 计算两两余弦相似度（采样策略，避免 O(n²)）
        # 505条 × 505 = 255k 对，完全可算全量
        logger.info("  [DIV] 计算余弦相似度矩阵...")
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        # 取上三角（不含对角线）
        n = sim_matrix.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_indices]

        mean_sim = float(np.mean(pairwise_sims))
        median_sim = float(np.median(pairwise_sims))
        std_sim = float(np.std(pairwise_sims))
        diversity = round(1.0 - mean_sim, 4)

        logger.info(f"  [DIV] 完成:")
        logger.info(f"    mean_similarity = {mean_sim:.4f}")
        logger.info(f"    median_similarity = {median_sim:.4f}")
        logger.info(f"    std_similarity = {std_sim:.4f}")
        logger.info(f"    diversity_score = {diversity:.4f} (1 - mean_sim)")

        return {
            "diversity_score": diversity,
            "mean_similarity": round(mean_sim, 4),
            "median_similarity": round(median_sim, 4),
            "std_similarity": round(std_sim, 4),
            "embedding_dim": embeddings.shape[1],
        }


# ============================================================
# 4. 主流程
# ============================================================

def get_device(cfg: EvalConfig, logger: logging.Logger) -> str:
    """确定计算设备"""
    if cfg.device != "auto":
        return cfg.device

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            logger.info(f"检测到 GPU: {name}，使用 CUDA")
        else:
            device = "cpu"
            logger.info("未检测到 GPU，使用 CPU")
    except ImportError:
        device = "cpu"
        logger.info("PyTorch 未安装，使用 CPU")
    return device


def run_ppl(data: List[Dict], cfg: EvalConfig, logger: logging.Logger) -> List[Dict]:
    """运行 PPL 评分"""
    logger.info("=" * 50)
    logger.info("[维度1] PPL 流畅度评分")
    logger.info("=" * 50)

    # 拼接 instruction + output 用于 N-gram 训练
    texts = []
    for d in data:
        instr = d.get("instruction", "")
        out = d.get("output", "")
        texts.append(f"{instr}\n{out}" if instr and out else (instr or out))

    scorer = PPLScorer(n=cfg.ppl_n, tokenizer_type=cfg.ppl_tokenizer)
    scorer.train(texts, logger)
    results = scorer.score(data, logger)
    return results


def run_ifd(data: List[Dict], cfg: EvalConfig, device: str,
            logger: logging.Logger) -> List[Dict]:
    """运行 IFD 评分"""
    logger.info("=" * 50)
    logger.info("[维度2] IFD 指令难度评分")
    logger.info("=" * 50)

    scorer = IFDScorer(
        model_name=cfg.ifd_model_name,
        device=device,
        batch_size=cfg.ifd_batch_size,
        max_length=cfg.ifd_max_length,
    )
    scorer.load(logger)
    results = scorer.score(data, logger)

    # 释放 GPU 内存
    del scorer
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_diversity(data: List[Dict], cfg: EvalConfig,
                  logger: logging.Logger) -> Dict:
    """运行 Embedding 多样性评分"""
    logger.info("=" * 50)
    logger.info("[维度3] Embedding 多样性评分")
    logger.info("=" * 50)

    scorer = DiversityScorer(
        model_name=cfg.embed_model_name,
        batch_size=cfg.embed_batch_size,
    )
    scorer.load(logger)
    results = scorer.score(data, logger)
    return results


def main():
    parser = argparse.ArgumentParser(description="AquilaLM 三维质量评估")
    parser.add_argument("--stage", choices=["ppl", "ifd", "diversity", "all"],
                        default="all", help="运行哪个阶段")
    parser.add_argument("--no-gpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--input", type=str, help="输入文件（覆盖 config）")
    parser.add_argument("--output", type=str, help="输出文件（覆盖 config）")
    parser.add_argument("--profile-only", action="store_true",
                        help="仅看数据分布，不评分")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config()
    if args.input:
        cfg.data_input = args.input
    if args.output:
        cfg.data_output = args.output

    logger = setup_logging(cfg.log_dir)
    logger.info("AquilaLM 阶段3：三维质量评估")
    logger.info(f"  输入: {cfg.data_input}")
    logger.info(f"  输出: {cfg.data_output}")

    # 加载数据
    data = load_jsonl(cfg.data_input)
    logger.info(f"  加载 {len(data)} 条数据")

    if args.profile_only:
        # 简单探查
        from collections import Counter
        sources = Counter(d.get("source", "?") for d in data)
        types = Counter(d.get("task_type", "?") for d in data)
        logger.info(f"  来源分布: {dict(sources)}")
        logger.info(f"  类型分布: {dict(types)}")
        ilens = [len(d.get("instruction", "")) for d in data]
        olens = [len(d.get("output", "")) for d in data]
        logger.info(f"  instruction 长度: min={min(ilens)}, max={max(ilens)}, "
                    f"mean={np.mean(ilens):.0f}")
        logger.info(f"  output 长度: min={min(olens)}, max={max(olens)}, "
                    f"mean={np.mean(olens):.0f}")
        return

    # 确定设备
    device = get_device(cfg, logger) if not args.no_gpu else "cpu"

    results = data

    # 维度1：PPL 流畅度
    if args.stage in ("ppl", "all"):
        results = run_ppl(results, cfg, logger)

    # 维度2：IFD 指令难度
    if args.stage in ("ifd", "all"):
        # IFD 在已有 ppl 字段的基础上叠加
        results = run_ifd(results, cfg, device, logger)

    # 维度3：Embedding 多样性
    if args.stage in ("diversity", "all"):
        div_result = run_diversity(results, cfg, logger)
        # 全局多样性指标只打日志，不写入单条
        # （多样性是数据集级指标，不是单条分数）
        logger.info(f"\n  全局多样性: {div_result['diversity_score']} "
                    f"(1 - mean_similarity={div_result['mean_similarity']})")

    # 保存结果
    save_jsonl(results, cfg.data_output)
    logger.info(f"\n输出已保存到 {cfg.data_output} ({len(results)} 条)")

    # 输出摘要
    logger.info("\n" + "=" * 50)
    logger.info("质量评估摘要")
    logger.info("=" * 50)
    if any("ppl" in r for r in results):
        ppls = [r.get("ppl", float('inf')) for r in results]
        finite = [p for p in ppls if p != float('inf')]
        if finite:
            logger.info(f"  PPL: mean={np.mean(finite):.1f}, "
                        f"median={np.median(finite):.1f}, "
                        f"min={min(finite):.1f}, max={max(finite):.1f}")

    if any("ifd_score" in r for r in results):
        ifds = [r.get("ifd_score", float('inf')) for r in results]
        finite = [v for v in ifds if v != float('inf') and not math.isnan(v)]
        if finite:
            logger.info(f"  IFD: mean={np.mean(finite):.3f}, "
                        f"median={np.median(finite):.3f}, "
                        f"min={min(finite):.3f}, max={max(finite):.3f}")
            easy = sum(1 for v in finite if v < 1)
            hard = sum(1 for v in finite if v >= 1)
            logger.info(f"  IFD 分布: 简单({easy}条, {easy/len(finite)*100:.1f}%) | "
                        f"困难({hard}条, {hard/len(finite)*100:.1f}%)")
        si_ifd = [v for r, v in zip(results, ifds) if r.get("source") == "self_instruct" and v != float('inf') and not math.isnan(v)]
        ei_ifd = [v for r, v in zip(results, ifds) if r.get("source") == "evol_instruct" and v != float('inf') and not math.isnan(v)]
        if si_ifd and ei_ifd:
            logger.info(f"  Self-Instruct IFD mean: {np.mean(si_ifd):.3f}")
            logger.info(f"  Evol-Instruct IFD mean: {np.mean(ei_ifd):.3f}")

    if "diversity_score" in dir() or args.stage in ("diversity", "all"):
        logger.info(f"  Diversity: {div_result['diversity_score']} "
                    f"(mean_sim={div_result['mean_similarity']})")


if __name__ == "__main__":
    main()
