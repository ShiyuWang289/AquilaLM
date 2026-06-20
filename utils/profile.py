"""数据探查工具 — 清洗前分析数据分布，用于科学设定阈值"""

import logging
from collections import Counter
from typing import Dict, List

from .io import extract_texts


def profile_data(data: List[Dict], logger: logging.Logger) -> None:
    """
    清洗前对种子数据做分布探查。
    """
    if not data:
        logger.warning("数据为空，跳过探查")
        return

    logger.info("=" * 50)
    logger.info(f"数据探查：共 {len(data)} 条")
    logger.info("=" * 50)

    texts = extract_texts(data)
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
    import re
    url_pattern = re.compile(r'https?://\S+')
    url_counts = [len(url_pattern.findall(t)) for t in texts]
    logger.info(f"  含URL的文本: {sum(1 for c in url_counts if c > 0)} 条, "
                f"最多URL数: {max(url_counts)}")

    logger.info("=" * 50)
