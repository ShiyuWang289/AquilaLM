"""数据健康度分析 — 清洗前后对比，量化清洗收益"""

import logging
from collections import Counter
from typing import Dict, List

from .io import extract_texts


def profile_health(data: List[Dict], label: str, logger: logging.Logger) -> Dict:
    """
    计算数据集的健康度指标。
    """
    texts = extract_texts(data)
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

        delta = av - bv
        pct = delta / max(abs(bv), 1e-9) * 100
        change = f"{delta:+.1f} ({pct:+.1f}%)" if fmt.endswith(".1f") else \
                 f"{delta:+.3f} ({pct:+.1f}%)" if fmt.endswith(".3f") else \
                 f"{delta:+.0f} ({pct:+.0f}%)" if fmt.endswith(".0f") else \
                 f"{delta:+.4f} ({pct:+.1f}%)"

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
