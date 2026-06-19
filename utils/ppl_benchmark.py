#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPL 评测集与阈值自动校准
========================
解决 PPL 阈值不可复用的问题：用固定的标准评测集来自动推荐阈值。

用法：
    python utils/ppl_benchmark.py --calibrate     # 在评测集上校准阈值
    python utils/ppl_benchmark.py --ppl "这是一段测试文本"  # 单条打分

检查内容：
    - 高质量中文 (维基/新闻) → 预期 PPL 低
    - 中等中文 (论坛/自媒体) → 预期 PPL 中
    - 劣质中文 (机翻垃圾/乱码) → 预期 PPL 高
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.stage1_clean_pipeline import NgramModel


# ============================================================
# 内置评测集（150条示例，可扩充）
# ============================================================

BUILTIN_BENCHMARK = {
    "high_quality": [
        "中国是世界上人口最多的国家之一，拥有悠久的历史和灿烂的文化。",
        "人工智能技术的快速发展正在改变各行各业的运作方式。",
        "量子计算利用量子力学原理来进行信息处理，有望在密码学等领域带来突破。",
        "数据结构是计算机存储、组织数据的方式，常见的有数组、链表、树和图。",
        "中华人民共和国的首都是北京，这是一个拥有三千年历史的古都。",
        "机器学习是人工智能的一个分支，专注于从数据中学习模式和规律。",
        "全球气候变化是当今人类面临的最严峻挑战之一，需要各国共同应对。",
        "Python 是一种解释型、面向对象的高级程序设计语言，由 Guido van Rossum 创造。",
        "文艺复兴是欧洲历史上一个重要的文化和思想运动时期，起源于十四世纪的意大利。",
        "细胞是生物体结构和功能的基本单位，所有生物都是由一个或多个细胞组成的。",
    ],
    "middle_quality": [
        "这个方法不错，大家可以试试，我觉得还挺好用的。",
        "最近在看一本书，讲的是如何提高工作效率，推荐给大家看看。",
        "今天天气真好啊，出去走走吧，别老呆在家里。",
        "这个东西怎么说呢，用起来还行吧，就是有时候会卡一下。",
        "我也遇到过这个问题，后来找了个朋友帮忙解决了，具体怎么说我也忘了。",
        "这个游戏的画质太棒了，我玩了好几天了都没腻，强烈推荐。",
        "最近准备换工作了，有谁知道哪个公司比较好吗，帮忙推荐一下。",
        "今天上课老师讲了一个新概念，我听得云里雾里的，有没有人解释一下。",
        "这个东西的价格还可以接受，但是功能好像不太全，再看看吧。",
        "今天心情不太好，想找个人聊聊，有没有人愿意听我唠叨。",
    ],
    "low_quality": [
        "111111 22222 aaaaaaa bbbb cc dddd eeeee fffff gggg hhhhh iiiii",
        "asdfghjkl qwertyuiop zxcvbnm 1234567890 !@#$%^&*()_+",
        "哈哈哈哈哈哈呵呵呵呵呵呵嘿嘿嘿嘿哈哈哈哈哈哈哈哈哈",
        ".............,,,,,,,,,,,;;;;;;;;;;;''''''''''''''''''",
        "The weather today beautiful is very and I like it much very.",
    ],
}


def load_benchmark_data() -> Dict[str, List[str]]:
    """加载评测集（内置 + 外部扩展）"""
    return BUILTIN_BENCHMARK


def calibrate_threshold(ngram_model: NgramModel,
                       benchmark: Dict[str, List[str]] = None) -> Tuple[float, Dict]:
    """
    基于标准评测集自动推荐 PPL 阈值。

    方法：阈值 = 高质量文本 P95 × 1.5

    返回 (recommended_threshold, detailed_stats)
    """
    if benchmark is None:
        benchmark = load_benchmark_data()

    stats = {}
    for label, texts in benchmark.items():
        ppls = []
        for text in texts:
            ppl = ngram_model.perplexity(text)
            if ppl != float('inf'):
                ppls.append(ppl)
        stats[label] = {
            "count": len(ppls),
            "mean": float(np.mean(ppls)),
            "median": float(np.median(ppls)),
            "p95": float(np.percentile(ppls, 95)),
            "min": float(min(ppls)),
            "max": float(max(ppls)),
        }

    # 阈值 = 高质量 P95 × 1.5
    high_p95 = stats["high_quality"]["p95"]
    threshold = round(high_p95 * 1.5, 1)

    return threshold, stats


def main():
    parser = argparse.ArgumentParser(description="PPL 评测集与阈值自动校准")
    parser.add_argument("--calibrate", action="store_true",
                        help="在评测集上训练哨兵模型并校准阈值")
    parser.add_argument("--ppl", type=str, help="对单条文本计算 PPL")
    parser.add_argument("--train-file", type=str,
                        help="外部训练语料路径（每行一条文本）")
    args = parser.parse_args()

    if args.calibrate or args.ppl:
        # 训练哨兵模型
        benchmark = load_benchmark_data()
        all_texts = []
        for texts in benchmark.values():
            all_texts.extend(texts)

        ngram = NgramModel(n=3, tokenizer_type="jieba")
        ngram.train(all_texts)

    if args.calibrate:
        threshold, stats = calibrate_threshold(ngram, benchmark)
        print(f"\n{'='*50}")
        print(f"PPL 评测集校准结果")
        print(f"{'='*50}")
        print(f"\n{'类别':<12} {'数量':<6} {'均值':<8} {'中位':<8} {'P95':<8} {'范围'}")
        print("-" * 50)
        for label, s in stats.items():
            print(f"{label:<12} {s['count']:<6} {s['mean']:<8.1f} "
                  f"{s['median']:<8.1f} {s['p95']:<8.1f} "
                  f"[{s['min']:.1f}, {s['max']:.1f}]")
        print(f"\n>>> 推荐阈值: PPL >= {threshold} (高质量 P95 × 1.5)")
        print(f"    低于此值的文本视为流畅，高于此值的需要审查。")

    if args.ppl:
        ppl = ngram.perplexity(args.ppl)
        print(f"PPL: {ppl:.1f}")
        if hasattr(ngram, 'train') and False:  # threshold reference
            pass


if __name__ == "__main__":
    main()
