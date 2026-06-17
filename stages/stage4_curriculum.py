#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquilaLM 课程学习排序器
=======================
两种策略对 IFD 打分后的数据进行课程学习排序：

  策略A: 分带混洗 (Band-Shuffle)  — 经典方案，面试讲"为什么不能纯排序"
  策略B: β退火加权采样 (Beta-Annealing) — 前沿方案，面试讲"动态难度调度"

使用方法:
    python stage4_curriculum.py                    # 全流程，输出两套方案
    python stage4_curriculum.py --strategy band    # 仅分带混洗
    python stage4_curriculum.py --strategy beta    # 仅 β 退火
    python stage4_curriculum.py --profile-only     # 数据探查

输入: data/scored.jsonl (505条 + IFD/PPL)
输出: data/curriculum_band.jsonl / data/curriculum_beta.jsonl

面试目标: 被问"你的课程学习排序是怎么做的，为什么不用纯排序"
时，能对比两种策略的优劣，解释 β 退火的数学原理。
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_jsonl, save_jsonl


# ============================================================
# 0. 配置与日志
# ============================================================

@dataclass
class CurriculumConfig:
    """课程学习配置"""
    data_input: str = "data/scored.jsonl"
    output_band: str = "data/curriculum_band.jsonl"
    output_beta: str = "data/curriculum_beta.jsonl"
    log_dir: str = "logs"

    # 质量过滤
    ppl_max: float = 5.0                    # PPL 超过此值的数据不进课程学习
    ifd_missing_strategy: str = "drop"      # IFD 缺失处理: drop | mean | median

    # 策略A: 分带混洗
    band_count: int = 3
    band_boundaries: Optional[List[float]] = None  # None = 自动分位数切割
    band_shuffle: bool = True

    # 策略B: β退火
    beta_start: float = 1.0                 # 初始 β（倾向简单）
    beta_end: float = 0.1                   # 终止 β（接近均匀）
    beta_steps: int = 5                     # 退火步数
    sampling_epochs: int = 3                # 完整遍历数据集多少轮
    temperature: float = 0.5                # 采样温度（softmax 用）


def load_config(config_path: str = "config.yaml") -> CurriculumConfig:
    """从 config.yaml 加载配置"""
    cfg = CurriculumConfig()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cc = raw.get("curriculum", {})
        if cc:
            cfg.ppl_max = cc.get("ppl_max", cfg.ppl_max)
            cfg.ifd_missing_strategy = cc.get("ifd_missing_strategy", cfg.ifd_missing_strategy)
            cfg.band_count = cc.get("band_count", cfg.band_count)
            cfg.band_boundaries = cc.get("band_boundaries", None)
            cfg.band_shuffle = cc.get("band_shuffle", True)
            cfg.beta_start = cc.get("beta_start", cfg.beta_start)
            cfg.beta_end = cc.get("beta_end", cfg.beta_end)
            cfg.beta_steps = cc.get("beta_steps", cfg.beta_steps)
            cfg.sampling_epochs = cc.get("sampling_epochs", cfg.sampling_epochs)
            cfg.temperature = cc.get("temperature", cfg.temperature)
            paths = raw.get("paths", {})
            if cc.get("output_band"):
                cfg.output_band = cc["output_band"]
            if cc.get("output_beta"):
                cfg.output_beta = cc["output_beta"]
    return cfg


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("AquilaLM-CL")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, "curriculum.log"),
                             encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


# ============================================================
# 1. 数据预处理
# ============================================================

def preprocess(data: List[Dict], cfg: CurriculumConfig,
               logger: logging.Logger) -> List[Dict]:
    """
    质量过滤 + IFD 缺失处理。

    三步:
    1. PPL > ppl_max 且 PPL 存在的 → 丢弃
    2. IFD 缺失/inf/nan → 按策略填充
    3. 返回清洁数据
    """
    logger.info("  [预处理] 质量过滤 (PPL <= %.1f)...", cfg.ppl_max)

    clean = []
    dropped_ppl = 0
    dropped_ifd = 0
    filled_ifd = 0

    valid_ifds = [d["ifd_score"] for d in data
                  if d.get("ifd_score") is not None
                  and d["ifd_score"] != float('inf')
                  and not math.isnan(d["ifd_score"])]

    ifd_mean = float(np.mean(valid_ifds)) if valid_ifds else 0.6
    ifd_median = float(np.median(valid_ifds)) if valid_ifds else 0.6

    for item in data:
        # PPL 过滤
        ppl = item.get("ppl")
        if ppl is not None and ppl > cfg.ppl_max:
            dropped_ppl += 1
            continue

        # IFD 缺失处理
        ifd = item.get("ifd_score")
        if ifd is None or ifd == float('inf') or math.isnan(ifd):
            if cfg.ifd_missing_strategy == "drop":
                dropped_ifd += 1
                continue
            elif cfg.ifd_missing_strategy == "median":
                item["ifd_score"] = round(ifd_median, 4)
                filled_ifd += 1
            else:  # mean
                item["ifd_score"] = round(ifd_mean, 4)
                filled_ifd += 1

        clean.append(dict(item))

    logger.info(f"  [预处理] 输入 {len(data)} → 输出 {len(clean)} "
                f"(PPL丢弃 {dropped_ppl}, IFD丢弃 {dropped_ifd}, "
                f"IFD填充 {filled_ifd})")
    return clean


# ============================================================
# 2. 策略A: 分带混洗
# ============================================================

class BandShuffleScheduler:
    """
    策略A: 按 IFD 分 N 个难度带，带内随机混洗，带间递增排列。

    面试解释：
    - 为什么分带？纯 IFD 排序会让相同 task_type 扎堆
    - 为什么带内混洗？防止模型记住"越往后越难"的规律
    - 为什么带间递增？保留课程学习"从易到难"的核心收益
    """

    def __init__(self, n_bands: int = 3,
                 boundaries: Optional[List[float]] = None):
        self.n_bands = n_bands
        self.boundaries = boundaries  # None = 自动等分

    def schedule(self, data: List[Dict],
                 logger: logging.Logger) -> List[Dict]:
        """
        返回带 curriculum_band 字段的数据列表，已按 band 排序。

        同时输出每带统计信息。
        """
        logger.info("=" * 50)
        logger.info("[策略A] 分带混洗 (Band-Shuffle)")
        logger.info("=" * 50)

        ifds = sorted([d["ifd_score"] for d in data])

        # 自动确定带边界（等分位数）
        if self.boundaries is None:
            boundaries = []
            for i in range(1, self.n_bands):
                p = i / self.n_bands * 100
                boundaries.append(round(float(np.percentile(ifds, p)), 3))
        else:
            boundaries = self.boundaries

        logger.info(f"  IFD 带边界: {boundaries}")
        logger.info(f"  IFD 范围: [{min(ifds):.3f}, {max(ifds):.3f}]")

        # 分带
        bands = defaultdict(list)
        for item in data:
            ifd = item["ifd_score"]
            band = 0
            for i, b in enumerate(boundaries):
                if ifd >= b:
                    band = i + 1
            bands[band].append(item)

        # 带内混洗
        rng = np.random.RandomState(42)
        for band_idx in range(self.n_bands):
            rng.shuffle(bands[band_idx])

        # 统计
        logger.info(f"\n{'Band':<6} {'IFD范围':<20} {'条数':<6} {'主要task_type'}")

        result = []
        for band_idx in range(self.n_bands):
            items = bands[band_idx]
            if not items:
                continue
            band_ifds = [d["ifd_score"] for d in items]
            lo, hi = min(band_ifds), max(band_ifds)

            # 标记 band（从 1 开始）
            for item in items:
                item["curriculum_band"] = band_idx + 1

            result.extend(items)

            # 主要任务类型
            type_counts = defaultdict(int)
            for d in items:
                type_counts[d.get("task_type", "?")] += 1
            top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:3]
            types_str = ", ".join(f"{t}:{c}" for t, c in top_types)

            logger.info(f"  Band{band_idx+1:<3} [{lo:.3f}-{hi:.3f}]        "
                        f"{len(items):<6} {types_str}")

        logger.info(f"  总条数: {len(result)}")
        return result


# ============================================================
# 3. 策略B: β退火加权采样
# ============================================================

class BetaAnnealingScheduler:
    """
    策略B: 用 IFD 分数加权采样，β 逐步退火。

    采样权重公式:
        P(样本 i) ∝ exp(-IFD_i * β / τ)

    直觉:
    - β 大 (如 1.0): 简单样本权重高 → 早期主要采样简单数据
    - β 小 (如 0.1): 权重接近均匀 → 后期所有数据被公平采样
    - τ (temperature): 控制权重分布的锐度

    面试必答:
    - "β退火的本质是什么？"
      → β 从 1.0 衰减到 0.1，采样概率从"集中在简单样本"
        逐步过渡到"均匀采样整个数据集"。相比硬分带，
        β退火让难度调度连续化，避免"突然换档"的震荡。
    - "为什么用 exponential 而不是线性权重？"
      → IFD 是非线性分布的（少数高 IFD 困难样本），
        exponential 能自适应地拉开差距，简单样本的权重
        在 β大时获得指数级优势，β小时自动平滑。
    """

    def __init__(self, beta_start: float = 1.0,
                 beta_end: float = 0.1,
                 beta_steps: int = 5,
                 epochs: int = 3,
                 temperature: float = 0.5):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epochs = epochs
        self.temperature = temperature

    def schedule(self, data: List[Dict],
                 logger: logging.Logger) -> List[Dict]:
        """
        为每条数据分配 sampling_weight，并为多个 β 退火步
        生成加权采样序列。

        返回带 sampling_weight 的数据列表（按 β=0 均匀排列）。
        """
        logger.info("=" * 50)
        logger.info("[策略B] β退火加权采样 (Beta-Annealing)")
        logger.info("=" * 50)

        ifds = np.array([d["ifd_score"] for d in data])

        # 归一化 IFD 到相似量级
        ifd_norm = ifds / np.median(ifds)

        # β 退火序列
        betas = np.linspace(self.beta_start, self.beta_end, self.beta_steps)
        logger.info(f"  β 序列: {[round(b, 3) for b in betas]}")
        logger.info(f"  temperature τ: {self.temperature}")
        logger.info(f"  epochs: {self.epochs}")

        logger.info(f"\n{'步':<6} {'β':<10} {'权重比(简单/困难)':<18} "
                    f"{'有效占比(p>0.001)':<16} {'熵'}")

        rng = np.random.RandomState(42)
        all_sequences = []

        for step, beta in enumerate(betas):
            # 计算采样权重: P ∝ exp(-IFD_norm * beta / τ)
            logits = -ifd_norm * beta / self.temperature
            logits = logits - np.max(logits)  # 数值稳定
            weights = np.exp(logits)
            weights = weights / weights.sum()

            # 统计
            top_ratio = np.max(weights) / np.min(weights[weights > 1e-8]) \
                if np.any(weights > 1e-8) else float('inf')
            active = float(np.sum(weights > 0.001) / len(weights))
            # 熵（越高越均匀）
            entropy = -np.sum(weights * np.log(weights + 1e-12))

            logger.info(f"  Step{step + 1:<3} β={beta:<7.3f} "
                        f"max/min={top_ratio:<15.1f} "
                        f"{active:.3f}          {entropy:.3f}")

            # 采样生成序列（单 epoch）
            n_samples = len(data)
            indices = rng.choice(len(data), size=n_samples, p=weights)
            for idx in indices:
                item = dict(data[idx])
                item["beta_step"] = step + 1
                item["beta"] = round(float(beta), 3)
                item["sampling_weight"] = round(float(weights[idx]), 6)
                all_sequences.append(item)

        # 去重：每条数据保留 β=0 时的采样权重（最均匀的参考权重）
        beta_zero_logits = -ifd_norm * self.beta_end / self.temperature
        beta_zero_logits = beta_zero_logits - np.max(beta_zero_logits)
        beta_zero_weights = np.exp(beta_zero_logits)
        beta_zero_weights = beta_zero_weights / beta_zero_weights.sum()

        result = []
        for idx, item in enumerate(data):
            item["sampling_weight"] = round(float(beta_zero_weights[idx]), 6)
            result.append(dict(item))

        logger.info(f"\n  生成 {len(all_sequences)} 条采样序列 "
                    f"({self.beta_steps} steps × {len(data)} samples)")

        # 按 IFD 并排对比前后 β 的权重变化
        logger.info(f"\n  β 退火效果对比 (top5简单 vs bottom5困难):")
        sorted_data = sorted(data, key=lambda d: d["ifd_score"])
        for label, items in [("最简单", sorted_data[:3]),
                              ("最困难", sorted_data[-3:])]:
            for item in items:
                logger.info(f"  [{label}] IFD={item['ifd_score']:.3f} "
                            f"weight={item.get('sampling_weight', 0):.6f} | "
                            f"instr={item['instruction'][:40]}...")

        return result


# ============================================================
# 4. 主流程
# ============================================================

def run_band(data: List[Dict], cfg: CurriculumConfig,
             logger: logging.Logger) -> List[Dict]:
    """运行策略A"""
    scheduler = BandShuffleScheduler(
        n_bands=cfg.band_count,
        boundaries=cfg.band_boundaries,
    )
    return scheduler.schedule(data, logger)


def run_beta(data: List[Dict], cfg: CurriculumConfig,
             logger: logging.Logger) -> List[Dict]:
    """运行策略B"""
    scheduler = BetaAnnealingScheduler(
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_steps=cfg.beta_steps,
        epochs=cfg.sampling_epochs,
        temperature=cfg.temperature,
    )
    return scheduler.schedule(data, logger)


def main():
    parser = argparse.ArgumentParser(description="AquilaLM 课程学习排序器")
    parser.add_argument("--strategy", choices=["band", "beta", "all"],
                        default="all", help="排序策略")
    parser.add_argument("--input", type=str, help="输入文件")
    parser.add_argument("--output", type=str, help="输出文件（覆盖 config）")
    parser.add_argument("--profile-only", action="store_true",
                        help="仅探查 IFD 分布")
    args = parser.parse_args()

    cfg = load_config()
    if args.input:
        cfg.data_input = args.input

    logger = setup_logging(cfg.log_dir)
    logger.info("AquilaLM 阶段4：课程学习排序器")
    logger.info(f"  输入: {cfg.data_input}")

    data = load_jsonl(cfg.data_input)
    logger.info(f"  加载 {len(data)} 条数据")

    if args.profile_only:
        from collections import Counter
        ifds = [d.get("ifd_score") for d in data
                if d.get("ifd_score") is not None
                and d["ifd_score"] != float('inf')
                and not math.isnan(d["ifd_score"])]
        ppls = [d.get("ppl") for d in data if d.get("ppl") is not None]
        types = Counter(d.get("task_type", "?") for d in data)
        sources = Counter(d.get("source", "?") for d in data)

        logger.info(f"  IFD: min={min(ifds):.3f}, p25={np.percentile(ifds,25):.3f}, "
                    f"p50={np.percentile(ifds,50):.3f}, "
                    f"p75={np.percentile(ifds,75):.3f}, max={max(ifds):.3f}")
        logger.info(f"  PPL: min={min(ppls):.1f}, max={max(ppls):.1f}")
        logger.info(f"  task_type: {dict(types)}")
        logger.info(f"  source: {dict(sources)}")
        logger.info(f"  IFD缺失/inf/nan: {len(data) - len(ifds)}")
        logger.info(f"  PPL>5.0: {sum(1 for p in ppls if p > 5.0)}")
        return

    # 预处理
    clean = preprocess(data, cfg, logger)

    if args.strategy in ("band", "all"):
        band_result = run_band(clean, cfg, logger)
        save_jsonl(band_result, cfg.output_band)
        logger.info(f"\n  策略A 输出: {cfg.output_band} ({len(band_result)} 条)")

    if args.strategy in ("beta", "all"):
        beta_result = run_beta(clean, cfg, logger)
        save_jsonl(beta_result, cfg.output_beta)
        logger.info(f"  策略B 输出: {cfg.output_beta} ({len(beta_result)} 条)")

    # 如果两套都运行，输出对比总结
    if args.strategy == "all":
        logger.info("\n" + "=" * 50)
        logger.info("策略对比总结")
        logger.info("=" * 50)
        logger.info("""
  ┌────────────┬──────────────────┬─────────────────────────┐
  │ 维度       │ 策略A (分带混洗)  │ 策略B (β退火加权采样)    │
  ├────────────┼──────────────────┼─────────────────────────┤
  │ 难度过渡   │ 硬切换 (带边界)  │ 软过渡 (β连续衰减)      │
  │ 采样方式   │ 同带均匀         │ 全文按权重采样           │
  │ 灾难遗忘   │ 带间切换有风险   │ 全程均匀混合，风险更低   │
  │ 适用数据量 │ 任意             │ ≥500条效果更好           │
  │ 业界定位   │ 经典基线         │ 2024主流                 │
  │ 面试价值   │ 解释"为什么不能  │ 解释β退火原理 + 对比     │
  │            │ 纯IFD排序"       │ 曲线，更有深度           │
  └────────────┴──────────────────┴─────────────────────────┘
""")


if __name__ == "__main__":
    main()
