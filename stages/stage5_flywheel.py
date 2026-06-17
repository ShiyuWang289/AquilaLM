#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquilaLM 反馈闭环调度器
=======================
将下游训练评估结果反向驱动上游数据策略参数，实现数据飞轮的"方向盘"。

核心组件:
  SignalNormalizer   — 原始评估指标 → 0~1 偏差分数
  DecisionEngine     — 信号 ⊗ 权重矩阵 → 参数调整量
  ParameterAdjuster  — 调整量 → 新参数值 + config 备份更新

使用方法:
    python stage5_flywheel.py                       # 全流程（mock信号演示）
    python stage5_flywheel.py --mock-scenario drift # 模拟"多样性坍塌"
    python stage5_flywheel.py --mock-scenario hard  # 模拟"模型只擅长简单任务"
    python stage5_flywheel.py --dry-run             # 只输出建议，不改config

面试目标: 被问"你的反馈闭环是怎么设计的"时, 能讲清信号→权重→
调整的完整决策链路, 以及哪些信号驱动哪些参数, 为什么。
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
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
class FlywheelConfig:
    """反馈闭环配置"""
    config_path: str = "config.yaml"
    report_path: str = "experiments/stage5_flywheel_report.md"
    log_dir: str = "logs"

    # 信号基线和阈值
    # 当偏差分数超过阈值时触发调整
    trigger_threshold: float = 0.05   # 偏差 < 5% 不触发调整

    # 调整幅度 clamp（防止单轮调整过猛）
    max_param_change_ratio: float = 0.30  # 单轮最多调 30%
    min_param_change_ratio: float = 0.02  # 单轮最少调 2%（低于则不调）

    # 权重矩阵（外部可覆盖）
    # 格式: {signal_name: {param_path: weight}}

    # 模拟场景
    mock_scenario: str = "baseline"  # baseline | drift | hard | noise


def load_config(config_path: str = "config.yaml") -> FlywheelConfig:
    cfg = FlywheelConfig()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        fc = raw.get("flywheel", {})
        if fc:
            cfg.trigger_threshold = fc.get("trigger_threshold", cfg.trigger_threshold)
            cfg.max_param_change_ratio = fc.get("max_param_change_ratio", cfg.max_param_change_ratio)
            cfg.min_param_change_ratio = fc.get("min_param_change_ratio", cfg.min_param_change_ratio)
            cfg.mock_scenario = fc.get("mock_scenario", cfg.mock_scenario)
    return cfg


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("AquilaLM-FW")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, "flywheel.log"),
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
# 1. 信号归一化器
# ============================================================

class SignalNormalizer:
    """
    将原始评估指标转化为 0~1 偏差分数。

    偏差分数语义:
      0.0  = 指标完美，无需调整
      0.5  = 中等偏差
      1.0  = 严重偏差，必须调整

    每组信号包含:
      - raw_value: 当前评估结果
      - baseline: 期望值/基线
      - direction: "lower_better" | "higher_better"
      - weight: 该信号在决策中的全局重要度
    """

    DEFAULT_SIGNALS = OrderedDict({
        # ---- 任务维度 ----
        "code_accuracy": {
            "label": "代码任务精度",
            "direction": "higher_better",
            "baseline": 0.60,
            "raw_value": 0.45,
            "weight": 0.25,
            "unit": "fraction",
        },
        "reasoning_accuracy": {
            "label": "推理任务精度",
            "direction": "higher_better",
            "baseline": 0.55,
            "raw_value": 0.50,
            "weight": 0.15,
            "unit": "fraction",
        },
        "qa_accuracy": {
            "label": "知识问答精度",
            "direction": "higher_better",
            "baseline": 0.65,
            "raw_value": 0.62,
            "weight": 0.10,
            "unit": "fraction",
        },
        "dialogue_quality": {
            "label": "对话质量",
            "direction": "higher_better",
            "baseline": 0.60,
            "raw_value": 0.58,
            "weight": 0.10,
            "unit": "fraction",
        },

        # ---- 数据维度 ----
        "diversity_score": {
            "label": "指令多样性",
            "direction": "higher_better",
            "baseline": 0.70,
            "raw_value": 0.73,
            "weight": 0.15,
            "unit": "fraction (1-mean_sim)",
        },
        "effective_ratio": {
            "label": "有效数据占比",
            "direction": "higher_better",
            "baseline": 0.85,
            "raw_value": 0.82,
            "weight": 0.10,
            "unit": "fraction",
        },

        # ---- 训练稳定性 ----
        "loss_volatility": {
            "label": "训练 Loss 波动率",
            "direction": "lower_better",
            "baseline": 0.10,
            "raw_value": 0.15,
            "weight": 0.10,
            "unit": "std(step_loss)/mean(step_loss)",
        },
        "ifd_gap": {
            "label": "SI/EI 难度差距",
            "direction": "higher_better",
            "baseline": 0.20,
            "raw_value": 0.18,
            "weight": 0.05,
            "unit": "ΔIFD",
        },
    })

    def __init__(self, signals: Optional[Dict] = None):
        self.signals = signals or copy.deepcopy(self.DEFAULT_SIGNALS)

    def normalize(self, logger: logging.Logger) -> Dict[str, Dict]:
        """
        对每个信号计算偏差分数。

        偏差分数公式:
          "higher_better": deviation = clamp((baseline - actual) / baseline, 0, 1)
          "lower_better":  deviation = clamp((actual - baseline) / baseline, 0, 1)
        """
        logger.info("=" * 50)
        logger.info("[信号归一化] 原始评估指标 → 偏差分数")
        logger.info("=" * 50)

        results = {}
        for key, sig in self.signals.items():
            raw = sig["raw_value"]
            base = sig["baseline"]
            direction = sig["direction"]

            if base == 0:
                deviation = 0.0
            elif direction == "higher_better":
                deviation = max(0.0, min(1.0, (base - raw) / base))
            else:  # lower_better
                deviation = max(0.0, min(1.0, (raw - base) / base))

            results[key] = {
                **sig,
                "deviation": round(deviation, 4),
                "delta_pct": round((raw - base) / base * 100, 1),
            }

            bar = "█" * int(deviation * 20)
            logger.info(
                f"  {sig['label']:<14} {raw:.2f} (基线{base:.2f}) "
                f"→ 偏差{deviation:.3f} {bar}"
            )

        return results

    def apply_mock_scenario(self, scenario: str) -> None:
        """
        注入模拟评估信号，模拟阶段6训练后可能出现的场景。

        三种典型场景:
          baseline — 略微偏差，多数指标接近基线（正常迭代）
          drift    — 多样性坍塌，dialogue/code 精度分化
          hard     — 模型只擅长简单，困难任务全崩
          noise    — 训练不稳定，loss震荡，有效数据占比低
        """
        scenarios = {
            "baseline": {
                "code_accuracy": 0.45,
                "reasoning_accuracy": 0.50,
                "qa_accuracy": 0.62,
                "dialogue_quality": 0.58,
                "diversity_score": 0.73,
                "effective_ratio": 0.82,
                "loss_volatility": 0.15,
                "ifd_gap": 0.18,
            },
            "drift": {
                "code_accuracy": 0.60,
                "reasoning_accuracy": 0.52,
                "qa_accuracy": 0.63,
                "dialogue_quality": 0.38,        # 对话崩了
                "diversity_score": 0.42,          # 多样性坍塌 (从0.73→0.42)
                "effective_ratio": 0.78,
                "loss_volatility": 0.22,
                "ifd_gap": 0.10,                  # 难度差距缩小（数据变简单了）
            },
            "hard": {
                "code_accuracy": 0.28,             # 代码/推理崩
                "reasoning_accuracy": 0.30,
                "qa_accuracy": 0.64,               # 简单任务还行
                "dialogue_quality": 0.55,
                "diversity_score": 0.68,
                "effective_ratio": 0.88,
                "loss_volatility": 0.08,
                "ifd_gap": 0.25,                   # 难度差距大（难的太难过不去）
            },
            "noise": {
                "code_accuracy": 0.52,
                "reasoning_accuracy": 0.48,
                "qa_accuracy": 0.60,
                "dialogue_quality": 0.54,
                "diversity_score": 0.65,
                "effective_ratio": 0.60,            # 一半数据没用
                "loss_volatility": 0.45,            # loss剧烈震荡
                "ifd_gap": 0.15,
            },
        }

        if scenario not in scenarios:
            return

        for key, val in scenarios[scenario].items():
            if key in self.signals:
                self.signals[key]["raw_value"] = val


# ============================================================
# 2. 决策引擎
# ============================================================

class DecisionEngine:
    """
    权重矩阵驱动的参数调整决策引擎。

    核心原理:
      Δparam_j = Σ_i (signal_deviation_i × weight_ij)

    其中 weight_ij 表示"第 i 个信号对第 j 个参数的影响力"。

    权重矩阵设计原则:
      1. 每个参数最多受 2~3 个信号影响（防过度耦合）
      2. 所有权重和归一化到 1.0（保证调整量量级可控）
      3. 主要信号权重 ≥ 0.5（让因果关系可解释）
    """

    # 权重矩阵: {param_path: {signal_key: weight}}
    # param_path 对应 config.yaml 中的路径（用 "." 分隔层级）
    WEIGHT_MATRIX: Dict[str, Dict[str, float]] = {
        # ---- Self-Instruct 参数 ----
        "instruction_synth.si_task_types.code_ratio": {
            "code_accuracy": 0.7,
            "diversity_score": 0.3,
        },
        "instruction_synth.si_task_types.reasoning_ratio": {
            "reasoning_accuracy": 0.7,
            "diversity_score": 0.3,
        },
        "instruction_synth.temperature_creative": {
            "diversity_score": 0.6,
            "effective_ratio": 0.4,
        },

        # ---- Evol-Instruct 参数 ----
        "instruction_synth.ei_max_evolve": {
            "ifd_gap": 0.5,
            "effective_ratio": 0.5,
        },
        "instruction_synth.ei_evolution_types.deep_reasoning_ratio": {
            "reasoning_accuracy": 0.6,
            "code_accuracy": 0.4,
        },
        "instruction_synth.ei_evolution_types.multi_turn_ratio": {
            "dialogue_quality": 0.8,
            "diversity_score": 0.2,
        },

        # ---- 种子筛选参数 ----
        "instruction_synth.seed_max_ppl": {
            "effective_ratio": 0.6,
            "loss_volatility": 0.4,
        },
        "instruction_synth.max_seeds": {
            "diversity_score": 0.5,
            "effective_ratio": 0.5,
        },

        # ---- 后过滤参数 ----
        "instruction_synth.postfilter_jaccard_threshold": {
            "diversity_score": 0.7,
            "effective_ratio": 0.3,
        },
        "instruction_synth.postfilter_consistency_threshold": {
            "effective_ratio": 0.5,
            "loss_volatility": 0.5,
        },

        # ---- 清洗参数 ----
        "rule_filter.min_text_length": {
            "effective_ratio": 0.7,
            "loss_volatility": 0.3,
        },
        "dedup.jaccard_threshold": {
            "diversity_score": 0.6,
            "effective_ratio": 0.4,
        },
        "ngram_ppl.ppl_threshold": {
            "loss_volatility": 0.6,
            "effective_ratio": 0.4,
        },
    }

    def __init__(self, weight_matrix: Optional[Dict] = None):
        self.weight_matrix = weight_matrix or copy.deepcopy(self.WEIGHT_MATRIX)

    def decide(self, signals: Dict[str, Dict],
               trigger_threshold: float,
               logger: logging.Logger) -> Dict[str, Dict]:
        """
        输入归一化信号，输出每个参数的调整建议。

        返回:
          {param_path: {
              "adjustment": float,      # -1 ~ +1（负=减, 正=加）
              "magnitude": float,       # 0 ~ 1（调整幅度）
              "drivers": [信号列表],     # 驱动此调整的信号及贡献
              "action": str,            # "increase" | "decrease" | "hold"
              "reason": str,            # 人类可读的调整理由
          }}
        """
        logger.info("=" * 50)
        logger.info("[决策引擎] 信号 → 参数调整量")
        logger.info("=" * 50)

        decisions = {}

        for param_path, signal_weights in self.weight_matrix.items():
            # 计算加权偏差
            total_deviation = 0.0
            total_weight = 0.0
            driver_details = []

            for sig_key, weight in signal_weights.items():
                if sig_key in signals:
                    dev = signals[sig_key]["deviation"]
                    total_deviation += dev * weight
                    total_weight += weight
                    driver_details.append(
                        f"{signals[sig_key]['label']}(dev={dev:.2f},w={weight:.1f})"
                    )

            if total_weight == 0:
                continue

            # 归一化
            normed_deviation = total_deviation / total_weight

            # 如果低于触发阈值 → hold
            if normed_deviation < trigger_threshold:
                decisions[param_path] = {
                    "adjustment": 0.0,
                    "magnitude": normed_deviation,
                    "drivers": driver_details,
                    "action": "hold",
                    "reason": f"偏差 {normed_deviation:.3f} < 阈值 {trigger_threshold}，保持不变",
                }
                continue

            # 确定调大还是调小
            # 规则: 如果主导信号的 direction 是 lower_better →
            #       偏差高说明值太大→减少；higher_better→偏差高说明值太小→增加
            dominant_signal = max(signal_weights, key=signal_weights.get)
            sig_info = signals.get(dominant_signal, {})

            if sig_info.get("direction") == "lower_better":
                # 值太高 → 减少参数
                direction = -1
            else:
                # 值太低 → 增加参数（ifd_gap 低 → 增加进化量）
                direction = +1

            # 特殊规则覆盖
            # diversity_score 低 → 降低 jaccard 阈值（去重放宽）+ 提高温度
            if param_path == "dedup.jaccard_threshold" and "diversity_score" in signal_weights:
                dev_d = signals.get("diversity_score", {}).get("deviation", 0)
                if dev_d > 0.1:
                    direction = -1  # 多样性低 → 降低jaccard阈值, 保留更多样数据
            if param_path == "instruction_synth.temperature_creative" and "diversity_score" in signal_weights:
                dev_d = signals.get("diversity_score", {}).get("deviation", 0)
                if dev_d > 0.1:
                    direction = +1  # 多样性低 → 提高温度, 增加生成多样性

            decisions[param_path] = {
                "adjustment": round(normed_deviation * direction, 4),
                "magnitude": round(normed_deviation, 4),
                "drivers": driver_details,
                "action": "increase" if direction > 0 else "decrease",
                "reason": self._explain(param_path, normed_deviation, direction,
                                        driver_details),
            }

            logger.info(
                f"  {param_path:<50} {decisions[param_path]['action']:>8} "
                f"(mag={normed_deviation:.3f})"
            )

        # 统计
        actions = {}
        for d in decisions.values():
            actions[d["action"]] = actions.get(d["action"], 0) + 1
        logger.info(f"\n  决策汇总: {dict(actions)}")
        logger.info(f"  hold (无动作): {actions.get('hold', 0)} 个参数")
        logger.info(f"  需调整: {len(decisions) - actions.get('hold', 0)} 个参数")

        return decisions

    def _explain(self, param_path: str, deviation: float,
                 direction: int, drivers: List[str]) -> str:
        """生成人类可读的调整理由"""
        action_cn = "增大" if direction > 0 else "减小"
        parts = param_path.split(".")
        param_name = parts[-1]

        readable = {
            "code_ratio": "代码指令采样比例",
            "reasoning_ratio": "推理指令采样比例",
            "deep_reasoning_ratio": "深化推理进化比例",
            "multi_turn_ratio": "多轮对话进化比例",
            "temperature_creative": "Self-Instruct 生成温度",
            "ei_max_evolve": "Evol-Instruct 最大进化量",
            "seed_max_ppl": "种子 PPL 上限",
            "max_seeds": "最大种子数",
            "postfilter_jaccard_threshold": "后过滤 Jaccard 阈值",
            "postfilter_consistency_threshold": "后过滤一致性阈值",
            "min_text_length": "规则过滤最小文本长度",
            "jaccard_threshold": "去重 Jaccard 阈值",
            "ppl_threshold": "PPL 过滤阈值",
        }

        return (f"{action_cn}{readable.get(param_name, param_name)}，"
                f"偏差 {deviation:.3f}，驱动: {', '.join(drivers)}")


# ============================================================
# 3. 参数调整器
# ============================================================

class ParameterAdjuster:
    """
    将调整量转化为新参数值，在上下限约束内 clamp。

    参数类型:
      - ratio (0~1 的小数):   调整 = clamp(val + Δ × range, 0, 1)
      - count (整数):         调整 = clamp(val + Δ × N, min, max)
      - threshold (阈值):     调整 = clamp(val × (1 + Δ), lower, upper)
    """

    PARAM_SCHEMA = {
        # (类型, 当前值, 下限, 上限)
        "instruction_synth.si_task_types.code_ratio":
            ("ratio", 0.20, 0.05, 0.40),
        "instruction_synth.si_task_types.reasoning_ratio":
            ("ratio", 0.20, 0.05, 0.40),
        "instruction_synth.temperature_creative":
            ("threshold", 0.80, 0.40, 1.20),
        "instruction_synth.ei_max_evolve":
            ("count", 150, 50, 300),
        "instruction_synth.ei_evolution_types.deep_reasoning_ratio":
            ("ratio", 0.25, 0.05, 0.50),
        "instruction_synth.ei_evolution_types.multi_turn_ratio":
            ("ratio", 0.25, 0.05, 0.50),
        "instruction_synth.seed_max_ppl":
            ("threshold", 15.0, 5.0, 30.0),
        "instruction_synth.max_seeds":
            ("count", 80, 20, 200),
        "instruction_synth.postfilter_jaccard_threshold":
            ("threshold", 0.70, 0.40, 0.90),
        "instruction_synth.postfilter_consistency_threshold":
            ("threshold", 0.70, 0.40, 0.90),
        "rule_filter.min_text_length":
            ("count", 10, 5, 50),
        "dedup.jaccard_threshold":
            ("threshold", 0.60, 0.30, 0.90),
        "ngram_ppl.ppl_threshold":
            ("threshold", 30, 10, 60),
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    def apply(self, decisions: Dict[str, Dict],
              max_change_ratio: float,
              min_change_ratio: float,
              dry_run: bool,
              logger: logging.Logger) -> Dict[str, Dict]:
        """
        将决策转化为具体参数值。

        返回: {param_path: {old, new, change_pct, clamped, action}}
        """
        logger.info("=" * 50)
        logger.info("[参数调整] 调整量 → 新参数值")
        logger.info("=" * 50)

        results = {}

        for param_path, dec in decisions.items():
            if dec["action"] == "hold":
                continue

            schema = self.PARAM_SCHEMA.get(param_path)
            if schema is None:
                logger.warning(f"  未知参数 {param_path}，跳过")
                continue

            ptype, current, lower, upper = schema
            adj = dec["adjustment"]

            # 检查调整幅度是否在 min/max 范围内
            if abs(adj) < min_change_ratio:
                logger.info(f"  {param_path}: 调整量 {adj:.3f} < min {min_change_ratio}，跳过")
                continue

            # Clamp 调整量
            adj = max(-max_change_ratio, min(max_change_ratio, adj))

            # 计算新值
            if ptype == "ratio":
                new_val = current + adj * (upper - lower) * 0.5
                new_val = max(lower, min(upper, new_val))
            elif ptype == "count":
                delta_count = int(adj * (upper - lower) * 0.5)
                new_val = max(lower, min(upper, current + delta_count))
            else:  # threshold
                new_val = current * (1.0 + adj * 0.5)
                new_val = max(lower, min(upper, new_val))

            change_pct = (new_val - current) / current * 100

            results[param_path] = {
                "old": current,
                "new": round(new_val, 4) if ptype != "count" else int(new_val),
                "change_pct": round(change_pct, 1),
                "adjusted_by": round(adj, 4),
                "action": dec["action"],
            }

            arrow = "↑" if change_pct > 0 else "↓"
            logger.info(
                f"  {param_path:<50} {current:>8} → {results[param_path]['new']:<8} "
                f"({arrow}{abs(change_pct):.1f}%) | {dec['action']}"
            )

        if not dry_run and results:
            self._update_config(results, logger)

        return results

    def _update_config(self, adjustments: Dict[str, Dict],
                       logger: logging.Logger) -> None:
        """将调整写回 config.yaml（自动备份）"""
        # 备份原配置
        backup_path = self.config_path + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        import shutil
        shutil.copy2(self.config_path, backup_path)
        logger.info(f"\n  配置已备份: {backup_path}")

        # 加载配置
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 应用调整
        for param_path, adj in adjustments.items():
            parts = param_path.split(".")
            node = config
            for p in parts[:-1]:
                if p not in node:
                    logger.warning(f"  配置路径 {param_path} 在 config 中不存在")
                    break
                node = node[p]
            else:
                last = parts[-1]
                if last in node:
                    node[last] = adj["new"]
                else:
                    logger.info(f"  配置键 {last} 在 config 中不存在，跳过写入")

        # 写回
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False)
        logger.info(f"  配置已更新: {self.config_path}")

    def revert(self, logger: logging.Logger) -> None:
        """回退到最新备份"""
        import glob
        backups = sorted(glob.glob(self.config_path + ".bak.*"))
        if not backups:
            logger.warning("无可用备份")
            return
        latest = backups[-1]
        import shutil
        shutil.copy2(latest, self.config_path)
        logger.info(f"配置已回退至: {latest}")


# ============================================================
# 4. 主流程
# ============================================================

def load_real_signals(data_dir: str, logger: logging.Logger) -> Optional[Dict]:
    """
    尝试从阶段3/4的产物中提取真实信号（而非 mock）。

    可提取的信号:
      - diversity_score: 从 stage3 日志或 scored.jsonl 的 embedding
      - effective_ratio: 从 stage2 的合成有效率
      - ifd_gap: SI/EI 的 IFD 差距
    """
    real = {}

    # IFD gap
    scored_path = os.path.join(data_dir, "scored.jsonl")
    if os.path.exists(scored_path):
        data = load_jsonl(scored_path)
        si_ifd = [d["ifd_score"] for d in data
                  if d.get("source") == "self_instruct"
                  and d.get("ifd_score") not in (float('inf'), None)
                  and not math.isnan(d.get("ifd_score", float('nan')))]
        ei_ifd = [d["ifd_score"] for d in data
                  if d.get("source") == "evol_instruct"
                  and d.get("ifd_score") not in (float('inf'), None)
                  and not math.isnan(d.get("ifd_score", float('nan')))]
        if si_ifd and ei_ifd:
            real["ifd_gap"] = round(float(np.mean(ei_ifd) - np.mean(si_ifd)), 4)

    # Effective ratio
    synth_path = os.path.join(data_dir, "synthesized.jsonl")
    cleaned_path = os.path.join(data_dir, "cleaned.jsonl")
    if os.path.exists(synth_path) and os.path.exists(cleaned_path):
        synth_data = load_jsonl(synth_path)
        cleaned_data = load_jsonl(cleaned_path)
        if cleaned_data:
            # 有效指令 / 清洗后种子数 ≈ 每条种子产生多少有效指令
            ratio = len(synth_data) / max(len(cleaned_data), 1)
            real["effective_ratio"] = round(min(ratio / 0.5, 1.0), 4)

    if real:
        logger.info(f"从现有数据提取信号: {real}")
    return real if real else None


def main():
    parser = argparse.ArgumentParser(description="AquilaLM 反馈闭环调度器")
    parser.add_argument("--mock-scenario", choices=["baseline", "drift", "hard", "noise"],
                        default=None, help="模拟评估场景（默认从config读取）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅输出建议，不修改 config.yaml")
    parser.add_argument("--revert", action="store_true",
                        help="回退 config.yaml 到最新备份")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_logging(cfg.log_dir)

    logger.info("AquilaLM 阶段5：反馈闭环调度器")
    logger.info("=" * 60)

    if args.revert:
        adjuster = ParameterAdjuster(cfg.config_path)
        adjuster.revert(logger)
        return

    # 1. 信号收集
    scenario = args.mock_scenario or cfg.mock_scenario
    logger.info(f"\n模拟场景: {scenario}")

    normalizer = SignalNormalizer()
    normalizer.apply_mock_scenario(scenario)

    # 尝试注入真实可计算信号
    real_signals = load_real_signals("data", logger)
    if real_signals:
        for key, val in real_signals.items():
            if key in normalizer.signals:
                normalizer.signals[key]["raw_value"] = val

    # 2. 信号归一化
    signals = normalizer.normalize(logger)

    # 3. 权重矩阵决策
    engine = DecisionEngine()
    decisions = engine.decide(signals, cfg.trigger_threshold, logger)

    # 4. 参数调整
    adjuster = ParameterAdjuster(cfg.config_path)
    adjustments = adjuster.apply(
        decisions,
        cfg.max_param_change_ratio,
        cfg.min_param_change_ratio,
        dry_run=args.dry_run,
        logger=logger,
    )

    # 5. 输出总结
    logger.info("\n" + "=" * 60)
    logger.info("反馈闭环总结")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("⚠ DRY-RUN 模式，未修改 config.yaml")

    # 按偏差排序信号
    ranked = sorted(signals.items(), key=lambda x: -x[1]["deviation"])
    logger.info(f"\n信号优先级（偏差从高到低）:")
    for key, sig in ranked[:5]:
        logger.info(f"  {sig['deviation']:.3f}  {sig['label']}")

    # 统计调整
    changed = [p for p, a in adjustments.items()]
    logger.info(f"\n已调整参数: {len(changed)}/{len(decisions)}")
    for p in changed:
        a = adjustments[p]
        logger.info(f"  {p}: {a['old']} → {a['new']} ({a['change_pct']:+.1f}%)")

    logger.info(f"\n下一轮建议:")
    logger.info(f"  1. 用新参数重新运行阶段2（指令合成）")
    logger.info(f"  2. 用新合成数据重新运行阶段3（质量评估）")
    logger.info(f"  3. 重新阶段4（课程学习排序）")
    logger.info(f"  4. （阶段6）用新数据重新训练 → 评测 → 形成闭环")


if __name__ == "__main__":
    main()
