#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquilaLM 数据清洗流水线
=======================
三阶段管道：规则过滤 → N-gram PPL 困惑度 → MinHash+LSH 去重

使用方法：
    python clean_pipeline.py                          # 全流程
    python clean_pipeline.py --stage rules            # 仅规则过滤
    python clean_pipeline.py --stage ppl              # 仅 PPL 评分
    python clean_pipeline.py --stage dedup            # 仅去重
    python clean_pipeline.py --profile-only           # 仅数据探查

依赖（全部已装）：
    jieba, tqdm, PyYAML, scipy, datasketch
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 添加项目根目录到 path，确保 utils/ 可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

import jieba
import yaml
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from utils.io import load_jsonl, save_jsonl, extract_texts
from utils.profile import profile_data
from utils.health import profile_health, log_health_comparison

# ============================================================
# 0. 配置与日志
# ============================================================


@dataclass
class PipelineConfig:
    """从 config.yaml 加载全部参数"""
    # 路径
    input_path: str = "data/raw_seeds.jsonl"
    output_dir: str = "data"
    log_dir: str = "logs"
    # 规则过滤
    rule_enabled: bool = True
    min_text_length: int = 10
    max_text_length: int = 10000
    max_non_chinese_ratio: float = 0.30
    max_repeat_char_ratio: float = 0.40
    max_url_count: int = 3
    min_effective_ratio: float = 0.50
    rule_output: str = "data/passed_rules.jsonl"
    # N-gram PPL
    ppl_enabled: bool = True
    ngram_n: int = 3
    ppl_tokenizer: str = "jieba"
    ppl_threshold: float = 200.0
    min_train_samples: int = 1000
    ppl_output: str = "data/scored.jsonl"
    # 去重
    dedup_enabled: bool = True
    jaccard_threshold: float = 0.80
    num_perm: int = 128
    dedup_ngram_size: int = 3
    lsh_weights: Tuple[float, ...] = (0.5, 0.5)
    dedup_output: str = "data/cleaned.jsonl"
    # 全局
    random_seed: int = 42
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        paths = cfg.get("paths", {})
        rules = cfg.get("rule_filter", {})
        ppl = cfg.get("ngram_ppl", {})
        dedup = cfg.get("dedup", {})
        g = cfg.get("global", {})

        return cls(
            input_path=paths.get("input", "data/raw_seeds.jsonl"),
            output_dir=paths.get("output_dir", "data"),
            log_dir=paths.get("log_dir", "logs"),
            rule_enabled=rules.get("enabled", True),
            min_text_length=rules.get("min_text_length", 10),
            max_text_length=rules.get("max_text_length", 10000),
            max_non_chinese_ratio=rules.get("max_non_chinese_ratio", 0.30),
            max_repeat_char_ratio=rules.get("max_repeat_char_ratio", 0.40),
            max_url_count=rules.get("max_url_count", 3),
            min_effective_ratio=rules.get("min_effective_ratio", 0.50),
            rule_output=rules.get("output", "data/passed_rules.jsonl"),
            ppl_enabled=ppl.get("enabled", True),
            ngram_n=ppl.get("n", 3),
            ppl_tokenizer=ppl.get("tokenizer", "jieba"),
            ppl_threshold=ppl.get("ppl_threshold", 200.0),
            min_train_samples=ppl.get("min_train_samples", 1000),
            ppl_output=ppl.get("output", "data/scored.jsonl"),
            dedup_enabled=dedup.get("enabled", True),
            jaccard_threshold=dedup.get("jaccard_threshold", 0.80),
            num_perm=dedup.get("num_perm", 128),
            dedup_ngram_size=dedup.get("ngram_size", 3),
            lsh_weights=tuple(dedup.get("lsh_weights", [0.6, 0.6, 0.6, 0.6])),
            dedup_output=dedup.get("output", "data/cleaned.jsonl"),
            random_seed=g.get("random_seed", 42),
            log_level=g.get("log_level", "INFO"),
        )


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    """双通道日志：控制台 + 文件"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("AquilaLM")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        # 控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(ch)
        # 文件（每次运行覆盖）
        fh = logging.FileHandler(os.path.join(log_dir, "clean_pipeline.log"), mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)

    return logger


# ============================================================
# 工具函数已提取到 utils/io.py，请使用:
#   from utils.io import load_jsonl, save_jsonl

# ============================================================
# 1. 数据探查层
# ============================================================

def profile_data(data: List[Dict], logger: logging.Logger) -> None:
    """
    清洗前对种子数据做分布探查。
    面试要点：阈值不能拍脑袋，必须先看数据分布再定。
    """
    if not data:
        logger.warning("数据为空，跳过探查")
        return

    logger.info("=" * 50)
    logger.info(f"数据探查：共 {len(data)} 条")
    logger.info("=" * 50)

    # --- 文本字段提取 ---
    texts = []
    for item in data:
        # 兼容多种字段名
        text = item.get("text") or item.get("content") or item.get("instruction") or ""
        if text:
            texts.append(text)

    if not texts:
        logger.warning("未找到文本字段，尝试的字段：text, content, instruction")
        return

    # 1. 长度分布
    lengths = [len(t) for t in texts]
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    logger.info(f"  长度分布 (字符数):")
    logger.info(f"    min={min(lengths)}, p10={lengths_sorted[n//10]}, "
                f"median={lengths_sorted[n//2]}, p90={lengths_sorted[n*9//10]}, max={max(lengths)}")
    logger.info(f"    <10字: {sum(1 for l in lengths if l < 10)} 条, "
                f">10000字: {sum(1 for l in lengths if l > 10000)} 条")

    # 2. 语言比例
    non_cn_ratios = []
    for t in texts:
        cn_chars = sum(1 for c in t if '一' <= c <= '鿿')
        non_cn_ratios.append(1.0 - cn_chars / max(len(t), 1))
    logger.info(f"  非中文比例: mean={sum(non_cn_ratios)/len(non_cn_ratios):.3f}, "
                f"max={max(non_cn_ratios):.3f}")

    # 3. 重复字符
    repeat_ratios = []
    for t in texts:
        if len(t) > 0:
            most_common_count = Counter(t).most_common(1)[0][1]
            repeat_ratios.append(most_common_count / len(t))
    logger.info(f"  重复字符比: mean={sum(repeat_ratios)/len(repeat_ratios):.3f}, "
                f"max={max(repeat_ratios):.3f}")

    # 4. URL 数量
    url_pattern = re.compile(r'https?://\S+')
    url_counts = [len(url_pattern.findall(t)) for t in texts]
    logger.info(f"  含URL的文本: {sum(1 for c in url_counts if c > 0)} 条, "
                f"最多URL数: {max(url_counts)}")

    logger.info("=" * 50)


# ============================================================
# 2. 规则过滤层
# ============================================================

@dataclass
class FilterStats:
    """每条规则的过滤统计"""
    total_input: int = 0
    passed: int = 0
    rejected_by: Dict[str, int] = field(default_factory=dict)

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info(f"  规则过滤统计: 输入 {self.total_input} → 通过 {self.passed} "
                    f"(保留率 {self.passed/max(self.total_input,1)*100:.1f}%)")
        for rule_name, count in sorted(self.rejected_by.items(), key=lambda x: -x[1]):
            logger.info(f"    {rule_name}: 丢弃 {count} 条")


class RuleFilter:
    """
    规则链过滤器。
    每条规则独立可配置，返回 (通过/拒绝) 和拒绝原因。
    """

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        # 编译正则（一次编译，反复使用）
        self.url_re = re.compile(r'https?://\S+')
        self.html_re = re.compile(r'<[^>]+>')

    def _extract_text(self, item: Dict) -> str:
        """从 JSON 对象中提取主文本"""
        return item.get("text") or item.get("content") or item.get("instruction") or ""

    # ---- 各规则 ----

    def _rule_length(self, text: str) -> Tuple[bool, str]:
        """长度过滤"""
        length = len(text)
        if length < self.config.min_text_length:
            return False, f"too_short({length}<{self.config.min_text_length})"
        if length > self.config.max_text_length:
            return False, f"too_long({length}>{self.config.max_text_length})"
        return True, ""

    def _rule_non_chinese(self, text: str) -> Tuple[bool, str]:
        """非中文字符比例"""
        if len(text) == 0:
            return False, "empty_text"
        cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
        non_cn_ratio = 1.0 - cn_chars / len(text)
        if non_cn_ratio > self.config.max_non_chinese_ratio:
            return False, f"non_chinese_ratio({non_cn_ratio:.2f}>{self.config.max_non_chinese_ratio})"
        return True, ""

    def _rule_repeat_char(self, text: str) -> Tuple[bool, str]:
        """重复字符比例（防全是同一字）"""
        if len(text) == 0:
            return False, "empty_text"
        most_common = Counter(text).most_common(1)[0][1]
        ratio = most_common / len(text)
        if ratio > self.config.max_repeat_char_ratio:
            return False, f"repeat_char_ratio({ratio:.2f}>{self.config.max_repeat_char_ratio})"
        return True, ""

    def _rule_html_url(self, text: str) -> Tuple[bool, str]:
        """HTML标签和URL残留"""
        urls = len(self.url_re.findall(text))
        html_tags = len(self.html_re.findall(text))
        if urls > self.config.max_url_count:
            return False, f"too_many_urls({urls}>{self.config.max_url_count})"
        if html_tags > self.config.max_url_count:
            return False, f"too_many_html_tags({html_tags}>{self.config.max_url_count})"
        return True, ""

    def _rule_effective_ratio(self, text: str) -> Tuple[bool, str]:
        """有效字符比：去空格/标点/换行后 占比"""
        if len(text) == 0:
            return False, "empty_text"
        # 有效字符 = 中文 + 英文 + 数字
        effective = sum(1 for c in text if c.isalnum() or '一' <= c <= '鿿')
        ratio = effective / len(text)
        if ratio < self.config.min_effective_ratio:
            return False, f"low_effective_ratio({ratio:.2f}<{self.config.min_effective_ratio})"
        return True, ""

    # ---- 链式执行 ----

    def apply(self, data: List[Dict]) -> Tuple[List[Dict], FilterStats]:
        """依次执行全部规则，返回通过的数据 + 统计"""
        stats = FilterStats(total_input=len(data))
        rules = [
            ("length", self._rule_length),
            ("non_chinese", self._rule_non_chinese),
            ("repeat_char", self._rule_repeat_char),
            ("html_url", self._rule_html_url),
            ("effective_ratio", self._rule_effective_ratio),
        ]

        passed = []
        for item in tqdm(data, desc="规则过滤"):
            text = self._extract_text(item)

            # 保留 text 为空的条目（可能是纯结构化数据）
            if not text.strip():
                passed.append(item)
                continue

            rejected = False
            for rule_name, rule_fn in rules:
                ok, reason = rule_fn(text)
                if not ok:
                    stats.rejected_by[rule_name] = stats.rejected_by.get(rule_name, 0) + 1
                    rejected = True
                    self.logger.debug(f"  丢弃 [{reason}]: {text[:80]}...")
                    break  # 一条规则不通过就丢弃，不继续判断

            if not rejected:
                passed.append(item)

        stats.passed = len(passed)
        stats.log_summary(self.logger)
        return passed, stats


# ============================================================
# 3. N-gram PPL 困惑度
# ============================================================

class NgramModel:
    """
    手写 N-gram 回退语言模型（替代 KenLM）。
    面试考点：回退机制 + Kneser-Ney 思想 + PPL 公式推导。
    """

    def __init__(self, n: int = 3, tokenizer_type: str = "jieba"):
        self.n = n
        self.tokenizer_type = tokenizer_type
        # n-gram 计数: {(w1,w2,...): count}
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        # 各阶总计
        self.total_tokens: int = 0
        self.vocab: Set[str] = set()
        # 平滑参数（简化的 additive smoothing）
        self.smooth_alpha: float = 0.01

    def _tokenize(self, text: str) -> List[str]:
        """分词：jieba 词粒度 或 按字粒度"""
        text = text.strip()
        if not text:
            return []
        if self.tokenizer_type == "jieba":
            return list(jieba.cut(text))
        else:
            # 按字切分（char-level）
            return list(text)

    def train(self, texts: List[str]) -> None:
        """
        从语料统计 n-gram 频率。
        时间复杂度 O(总token数 × n)，n 通常为 2~3，可接受。
        """
        self.logger = logging.getLogger("AquilaLM")
        self.logger.info(f"  训练 N-gram 模型 (n={self.n}, tokenizer={self.tokenizer_type})...")

        for text in tqdm(texts, desc="N-gram 训练"):
            tokens = self._tokenize(text)
            if len(tokens) < self.n:
                continue

            self.total_tokens += len(tokens)
            self.vocab.update(tokens)

            # 统计 1-gram 到 n-gram
            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    ngram = tuple(tokens[i:i + order])
                    self.ngram_counts[ngram] += 1

        self.vocab_size = len(self.vocab)
        self.logger.info(f"  完成: vocab_size={self.vocab_size}, "
                         f"total_tokens={self.total_tokens}, "
                         f"unique_ngrams={len(self.ngram_counts)}")

    def _count(self, ngram: Tuple[str, ...]) -> int:
        """查询 n-gram 计数"""
        return self.ngram_counts.get(ngram, 0)

    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        计算 P(word | context)，带回退机制。
        context 是从近到远排列的 (w_{i-1}, w_{i-2}, ...)

        核心逻辑：
        1. 先在最高阶查 count(context + word) / count(context)
        2. 如果为 0 → 回退到 (n-1)-gram
        3. 回退到底 → 用 unigram 概率（additive smoothing）
        """
        if context:
            full_ngram = context + (word,)
            full_count = self._count(full_ngram)
            context_count = self._count(context)

            if full_count > 0 and context_count > 0:
                # 简化的回退：用折扣系数 0.4 保留概率质量给低阶
                discount = 0.4
                return (1 - discount) * full_count / context_count \
                       + discount * self.probability(word, context[1:])

        # unigram 回退层（additive smoothing）
        word_count = self._count((word,))
        return (word_count + self.smooth_alpha) / (self.total_tokens + self.smooth_alpha * self.vocab_size)

    def perplexity(self, text: str) -> float:
        """
        计算单条文本的 PPL。
        PPL = exp(-1/N × Σ log P(w_i | context_i))
        """
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return float('inf')

        log_prob_sum = 0.0
        valid_tokens = 0

        for i in range(len(tokens)):
            # 构建 context: 从位置 i 往前最多 n-1 个词
            context_start = max(0, i - self.n + 1)
            context = tuple(tokens[context_start:i]) if i > 0 else ()
            prob = self.probability(tokens[i], context)
            if prob > 0:
                log_prob_sum += math.log(prob)
                valid_tokens += 1

        if valid_tokens == 0:
            return float('inf')

        avg_log_prob = log_prob_sum / valid_tokens
        return math.exp(-avg_log_prob)


def filter_by_ppl(data: List[Dict], ngram_model: NgramModel,
                  threshold: float, logger: logging.Logger) -> Tuple[List[Dict], Dict]:
    """批量 PPL 评分 + 按阈值过滤"""
    logger.info(f"  PPL 阈值: {threshold}")

    scored = []
    ppl_values = []
    passed_count = 0
    rejected_count = 0

    for item in tqdm(data, desc="PPL 评分"):
        text = item.get("text") or item.get("content") or item.get("instruction") or ""
        ppl = ngram_model.perplexity(text)
        ppl_values.append(ppl)

        item_copy = dict(item)
        item_copy["ppl"] = round(ppl, 2)
        item_copy["ppl_pass"] = ppl <= threshold
        scored.append(item_copy)

        if ppl <= threshold:
            passed_count += 1
        else:
            rejected_count += 1

    # 分布统计
    finite_ppls = [p for p in ppl_values if p != float('inf')]
    logger.info(f"  PPL 分布: min={min(finite_ppls):.1f}, "
                f"median={sorted(finite_ppls)[len(finite_ppls)//2]:.1f}, "
                f"max={max(finite_ppls):.1f}")
    logger.info(f"  PPL 过滤: 通过 {passed_count}, 拒绝 {rejected_count} "
                f"(保留率 {passed_count/max(len(data),1)*100:.1f}%)")

    stats = {
        "ppl_threshold": threshold,
        "ppl_passed": passed_count,
        "ppl_rejected": rejected_count,
        "ppl_min": min(finite_ppls) if finite_ppls else None,
        "ppl_max": max(finite_ppls) if finite_ppls else None,
    }
    return scored, stats


# ============================================================
# 4. 数据健康度分析
# ============================================================

def _extract_texts(data: List[Dict]) -> List[str]:
    """从 JSON 列表中提取文本字段"""
    texts = []
    for item in data:
        t = item.get("text") or item.get("content") or item.get("instruction") or ""
        if t:
            texts.append(t)
    return texts


def profile_health(data: List[Dict], label: str, logger: logging.Logger) -> Dict:
    """
    计算数据集的健康度指标。
    面试要点：清洗不能只看数量变化，必须用分布指标量化效果。
    """
    texts = _extract_texts(data)
    if not texts:
        logger.warning(f"  [{label}] 无有效文本，跳过")
        return {}

    n = len(texts)

    # 1. 长度分布
    lengths = [len(t) for t in texts]
    len_mean = sum(lengths) / n
    len_std = (sum((l - len_mean) ** 2 for l in lengths) / n) ** 0.5

    # 2. 非中文比例
    non_cn_ratios = []
    for t in texts:
        cn = sum(1 for c in t if '一' <= c <= '鿿')
        non_cn_ratios.append(1.0 - cn / max(len(t), 1))
    cn_mean = sum(non_cn_ratios) / n

    # 3. 重复字符比
    repeat_ratios = []
    for t in texts:
        if len(t) > 0:
            mc = max(Counter(t).values())
            repeat_ratios.append(mc / len(t))
    repeat_mean = sum(repeat_ratios) / n if repeat_ratios else 0

    # 4. PPL 分布（如有）
    ppl_vals = [d["ppl"] for d in data if "ppl" in d and d["ppl"] != float('inf')]
    if ppl_vals:
        ppl_mean = sum(ppl_vals) / len(ppl_vals)
        ppl_sorted = sorted(ppl_vals)
        ppl_median = ppl_sorted[len(ppl_sorted) // 2]
        ppl_std = (sum((p - ppl_mean) ** 2 for p in ppl_vals) / len(ppl_vals)) ** 0.5
        ppl_max = max(ppl_vals)
    else:
        ppl_mean = ppl_median = ppl_std = ppl_max = None

    # 5. n-gram 多样性（基于 bigram 种类数/总 bigram 数）
    all_bigrams = []
    for t in texts[:500]:  # 采样 500 条避免过慢
        chars = list(t)
        all_bigrams.extend("".join(chars[i:i + 2]) for i in range(len(chars) - 1))
    bigram_unique_ratio = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    # 记录
    health = {
        "label": label,
        "count": n,
        "len_mean": len_mean,
        "len_std": len_std,
        "len_cv": len_std / max(len_mean, 1),
        "non_cn_mean": cn_mean,
        "repeat_mean": repeat_mean,
        "bigram_diversity": bigram_unique_ratio,
        "ppl_mean": ppl_mean,
        "ppl_median": ppl_median,
        "ppl_std": ppl_std,
        "ppl_max": ppl_max,
    }
    return health


def log_health_comparison(before: Dict, after: Dict, logger: logging.Logger) -> None:
    """打印清洗前后健康度对比表"""
    logger.info("=" * 70)
    logger.info("数据健康度对比")
    logger.info("=" * 70)

    metrics = [
        ("样本数",           "count",           "d",     "{:.0f}"),
        ("长度均值",          "len_mean",        "d",     "{:.0f}"),
        ("长度标准差",         "len_std",         "d",     "{:.0f}"),
        ("长度变异系数(std/mean)", "len_cv",      ".2f",   "{:.3f}"),
        ("非中文比例均值",     "non_cn_mean",     ".2f",   "{:.3f}"),
        ("重复字符比均值",     "repeat_mean",     ".3f",   "{:.4f}"),
        ("Bigram多样性",       "bigram_diversity", ".3f",  "{:.4f}"),
    ]

    if before.get("ppl_mean") is not None and after.get("ppl_mean") is not None:
        metrics += [
            ("PPL 均值",      "ppl_mean",        ".1f",   "{:.1f}"),
            ("PPL 中位数",    "ppl_median",      ".1f",   "{:.1f}"),
            ("PPL 标准差",    "ppl_std",         ".1f",   "{:.1f}"),
            ("PPL 最大值",    "ppl_max",         ".1f",   "{:.1f}"),
        ]

    header = f"{'指标':<30} {'清洗前':>12} {'清洗后':>12} {'变化':>12}"
    logger.info(header)
    logger.info("-" * 70)

    for name, key, direction, fmt in metrics:
        bv = before.get(key)
        av = after.get(key)
        if bv is None or av is None:
            continue

        if direction == "d":
            # 下降是好事
            delta = av - bv
            pct = delta / max(abs(bv), 1) * 100
            change = f"{delta:+.0f} ({pct:+.0f}%)"
        elif direction == ".2f":
            delta = av - bv
            pct = delta / max(abs(bv), 1e-9) * 100
            change = f"{delta:+.3f} ({pct:+.1f}%)"
        elif direction == ".3f":
            delta = av - bv
            pct = delta / max(abs(bv), 1e-9) * 100
            change = f"{delta:+.4f} ({pct:+.1f}%)"
        else:
            delta = av - bv
            pct = delta / max(abs(bv), 1e-9) * 100
            change = f"{delta:+.1f} ({pct:+.1f}%)"

        logger.info(f"{name:<30} {fmt.format(bv):>12} {fmt.format(av):>12} {change:>12}")

    # 解读
    logger.info("")
    logger.info("解读要点：")
    logger.info("  · 数据量下降但不可过多 → 说明没有过度过滤")
    logger.info("  · PPL 标准差下降 → 尾部垃圾文本被清掉")
    logger.info("  · 长度变异系数下降 → 极端异常长度被治理")
    logger.info("  · Bigram 多样性上升 → 数据多样性提升")
    logger.info("  · 重复字符比均值下降 → 低质量重复文本减少")
    logger.info("=" * 70)


# ============================================================
# 5. MinHash + LSH 去重
# ============================================================


def _text_to_ngrams(text: str, n: int) -> Set[str]:
    """将文本转为 n-gram 集合"""
    if not text:
        return set()
    # 按字切分（对中文更好）
    chars = list(text)
    return {"".join(chars[i:i + n]) for i in range(len(chars) - n + 1)}


def _quality_score(item: Dict) -> float:
    """
    综合质量评分，用于相似组内择优保留。
    优先规则：
      1. PPL 越低越好
      2. 文本越长信息量越大（但不过长）
      3. 有 reply/answer 字段的优先（更完整的对话数据）
    """
    score = 0.0
    text = item.get("text") or item.get("content") or item.get("instruction") or ""

    # PPL 低 → 分数高（PPL=10 → +10, PPL=200 → +0.05）
    ppl = item.get("ppl", 100)
    if ppl and ppl > 0:
        score += 100.0 / max(ppl, 1)

    # 长度适中 → 分数高（最优 200~500 字）
    length = len(text)
    if 100 < length < 2000:
        score += min(length, 500) / 500.0

    # 有有效回复 → +2 分
    if item.get("reply") or item.get("answer") or item.get("output"):
        score += 2.0

    return score


class Deduplicator:
    """
    MinHash + LSH 近似去重器。
    封装 datasketch，面试时需能解释底层 Jaccard/MinHash/LSH 原理。
    """

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.lsh: Optional[MinHashLSH] = None

    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """主入口：建立 LSH 索引 → 查找相似对 → 去重"""
        n = len(data)
        self.logger.info(f"  建立 LSH 索引 (threshold={self.config.jaccard_threshold}, "
                         f"num_perm={self.config.num_perm})...")

        # 初始化 LSH
        self.lsh = MinHashLSH(
            threshold=self.config.jaccard_threshold,
            num_perm=self.config.num_perm,
            weights=self.config.lsh_weights,
        )

        # 为每条数据生成 MinHash 签名
        minhashes = []
        for idx, item in enumerate(tqdm(data, desc="MinHash 签名生成")):
            text = item.get("text") or item.get("content") or item.get("instruction") or ""
            ngrams = _text_to_ngrams(text, self.config.dedup_ngram_size)
            mh = MinHash(num_perm=self.config.num_perm)
            for ng in ngrams:
                mh.update(ng.encode("utf-8"))
            minhashes.append((idx, mh))
            self.lsh.insert(idx, mh)

        # 查找重复组
        self.logger.info("  查询相似对...")
        duplicate_groups = []         # 每组是一个相似文档的集合
        processed = set()

        for idx, mh in tqdm(minhashes, desc="LSH 查询"):
            if idx in processed:
                continue
            # 查询与当前文档相似的候选
            candidates = self.lsh.query(mh)
            if len(candidates) <= 1:
                # 独一份，不需要去重
                continue
            # 收集未被处理的候选
            group = set(candidates) - processed
            if len(group) > 1:
                duplicate_groups.append(group)
                processed.update(group)

        # 去重：每组保留质量最高的那条
        removed_count = 0
        kept_indices = set(range(n))
        for group in duplicate_groups:
            group_list = list(group)
            # 按质量评分排序，保留最佳
            scored_items = [(idx, _quality_score(data[idx])) for idx in group_list]
            scored_items.sort(key=lambda x: -x[1])
            best_idx = scored_items[0][0]
            # 其他全部标记为删除
            for idx, _ in scored_items[1:]:
                kept_indices.discard(idx)
                removed_count += 1

        result = [data[i] for i in sorted(kept_indices)]

        self.logger.info(f"  去重统计: 发现 {len(duplicate_groups)} 个相似组, "
                         f"删除 {removed_count} 条, 保留 {len(result)} 条 "
                         f"(保留率 {len(result)/max(n,1)*100:.1f}%)")

        # 给每条数据加去重标记
        for item in result:
            item["dedup_pass"] = True

        return result


# ============================================================
# 5. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AquilaLM 数据清洗流水线")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "profile", "rules", "ppl", "dedup"],
                        help="执行阶段")
    parser.add_argument("--profile-only", action="store_true", help="仅数据探查")
    args = parser.parse_args()

    # 加载配置
    cfg = PipelineConfig.from_yaml(args.config)
    logger = setup_logging(cfg.log_dir, cfg.log_level)

    logger.info("=" * 60)
    logger.info("AquilaLM 数据清洗流水线 启动")
    logger.info(f"  配置文件: {args.config}")
    logger.info(f"  执行阶段: {args.stage}")
    logger.info(f"  输入数据: {cfg.input_path}")
    logger.info("=" * 60)

    # 加载数据
    data = load_jsonl(cfg.input_path)
    logger.info(f"加载数据: {len(data)} 条")

    # 仅探查模式
    if args.profile_only:
        profile_data(data, logger)
        return

    # 全流程或指定阶段
    current = data
    snapshot_before = None  # 清洗前快照，用于健康度对比

    # --- 阶段 A: 规则过滤 ---
    if args.stage in ("all", "rules", "profile"):
        profile_data(current, logger)
        if args.stage == "profile":
            return

    if args.stage in ("all", "rules") and cfg.rule_enabled:
        # 保存规则过滤前快照
        snapshot_before = [dict(item) for item in current]
        logger.info("\n" + "=" * 50)
        logger.info("阶段 1/3: 规则过滤")
        logger.info("=" * 50)
        rule_filter = RuleFilter(cfg, logger)
        current, rule_stats = rule_filter.apply(current)
        save_jsonl(current, cfg.rule_output)
        logger.info(f"规则过滤输出: {cfg.rule_output} ({len(current)} 条)")

    # --- 阶段 B: N-gram PPL ---
    if args.stage in ("all", "ppl") and cfg.ppl_enabled:
        logger.info("\n" + "=" * 50)
        logger.info("阶段 2/3: N-gram PPL 困惑度")
        logger.info("=" * 50)
        # 提取文本训练 N-gram 模型
        texts = []
        for item in current:
            t = item.get("text") or item.get("content") or item.get("instruction") or ""
            if t:
                texts.append(t)
        logger.info(f"  用于训练N-gram模型的文本数: {len(texts)}")

        ngram_model = NgramModel(n=cfg.ngram_n, tokenizer_type=cfg.ppl_tokenizer)
        ngram_model.logger = logger
        ngram_model.train(texts)

        current, ppl_stats = filter_by_ppl(current, ngram_model, cfg.ppl_threshold, logger)
        # 保存全部评分结果（含未通过的，方便后续分析）
        snapshot_ppl_before = [dict(item) for item in current]
        save_jsonl(current, cfg.ppl_output)
        logger.info(f"PPL 评分输出（全部）: {cfg.ppl_output} ({len(current)} 条)")
        # 实际过滤：只保留 ppl_pass=True 的数据进入下一阶段
        current = [item for item in current if item.get("ppl_pass", True)]
        snapshot_ppl_after = [dict(item) for item in current]
        logger.info(f"PPL 硬过滤后: {len(current)} 条 (阈值 {cfg.ppl_threshold})")
        # PPL 过滤前后健康度对比
        health_ppl_before = profile_health(snapshot_ppl_before, "PPL过滤前", logger)
        health_ppl_after  = profile_health(snapshot_ppl_after,  "PPL过滤后", logger)
        if health_ppl_before and health_ppl_after:
            log_health_comparison(health_ppl_before, health_ppl_after, logger)

    # --- 阶段 C: MinHash+LSH 去重 ---
    if args.stage in ("all", "dedup") and cfg.dedup_enabled:
        logger.info("\n" + "=" * 50)
        logger.info("阶段 3/3: MinHash+LSH 去重")
        logger.info("=" * 50)
        deduplicator = Deduplicator(cfg, logger)
        current = deduplicator.deduplicate(current)
        save_jsonl(current, cfg.dedup_output)
        logger.info(f"去重输出: {cfg.dedup_output} ({len(current)} 条)")

    # --- 最终统计 ---
    logger.info("\n" + "=" * 60)
    logger.info(f"流水线完成！")
    logger.info(f"  输入: {len(data)} 条")
    logger.info(f"  输出: {len(current)} 条")
    logger.info(f"  总保留率: {len(current)/max(len(data),1)*100:.1f}%")
    logger.info(f"  最终产出: {cfg.dedup_output}")
    logger.info("=" * 60)

    # --- 健康度对比 ---
    if snapshot_before and current:
        logger.info("")
        health_before = profile_health(snapshot_before, "清洗前(原始数据)", logger)
        health_after = profile_health(current, "清洗后(最终产出)", logger)
        if health_before and health_after:
            log_health_comparison(health_before, health_after, logger)


if __name__ == "__main__":
    main()
