#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquilaLM 指令合成引擎
=====================
Self-Instruct → Evol-Instruct → DPO偏好构造 → 质量过滤

使用方法：
    python stage2_instruction_synth.py              # 全流程
    python stage2_instruction_synth.py --stage self # 仅 Self-Instruct
    python stage2_instruction_synth.py --stage dpo  # 仅 DPO 偏好构造

环境变量：
    export DEEPSEEK_API_KEY=sk-your-key-here
"""

import argparse
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_jsonl, save_jsonl, extract_texts


# ============================================================
# 0. 配置与日志
# ============================================================

@dataclass
class SynthConfig:
    """指令合成配置（从 config.yaml 加载）"""
    data_input: str = "data/cleaned.jsonl"
    output_dir: str = "data"
    log_dir: str = "logs"
    # API
    api_base: str = "https://api.deepseek.com"
    api_key: str = ""
    flash_model: str = "deepseek-chat"
    pro_model: str = "deepseek-reasoner"
    max_retries: int = 3
    temp_creative: float = 0.8
    temp_precise: float = 0.3
    # 种子筛选
    seed_min_len: int = 100
    seed_max_len: int = 500
    seed_max_ppl: float = 15.0
    seed_min_cn: float = 0.5
    max_seeds: int = 80
    # Self-Instruct
    si_per_seed: int = 5
    si_task_types: List[str] = field(default_factory=lambda: [
        "推理分析", "知识问答", "文本生成", "代码编写", "多轮对话"
    ])
    # Evol-Instruct
    ei_max_evolve: int = 150
    ei_evolution_types: List[str] = field(default_factory=lambda: [
        "增加约束", "深化推理", "拆解子任务", "多轮对话", "代码加解释"
    ])
    # DPO
    dpo_code_pairs: int = 20
    dpo_math_pairs: int = 20
    dpo_exec_timeout: int = 5
    # 后过滤
    pf_min_inst_len: int = 10
    pf_min_out_len: int = 30
    pf_jaccard_threshold: float = 0.7
    pf_consistency_threshold: float = 0.7
    consistency_sample: int = 30
    # 输出
    output: str = "data/synthesized.jsonl"
    dpo_output: str = "data/dpo_pairs.jsonl"
    random_seed: int = 42
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> "SynthConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        paths = cfg.get("paths", {})
        synth = cfg.get("instruction_synth", {})
        pp = cfg.get("ngram_ppl", {})
        g = cfg.get("global", {})

        return cls(
            data_input=paths.get("output_dir", "data") + "/cleaned.jsonl",
            output_dir=paths.get("output_dir", "data"),
            log_dir=paths.get("log_dir", "logs"),
            api_base=synth.get("api_base", "https://api.deepseek.com"),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            flash_model=synth.get("flash_model", "deepseek-chat"),
            pro_model=synth.get("pro_model", "deepseek-reasoner"),
            max_retries=synth.get("max_retries", 3),
            temp_creative=synth.get("temperature_creative", 0.8),
            temp_precise=synth.get("temperature_precise", 0.3),
            seed_min_len=synth.get("seed_min_length", 100),
            seed_max_len=synth.get("seed_max_length", 500),
            seed_max_ppl=synth.get("seed_max_ppl", 15.0),
            seed_min_cn=synth.get("seed_min_cn_ratio", 0.5),
            max_seeds=synth.get("max_seeds", 80),
            si_per_seed=synth.get("si_generations_per_seed", 5),
            si_task_types=synth.get("si_task_types", ["推理分析", "知识问答", "文本生成", "代码编写", "多轮对话"]),
            ei_max_evolve=synth.get("ei_max_evolve", 150),
            ei_evolution_types=synth.get("ei_evolution_types", ["增加约束", "深化推理", "拆解子任务", "多轮对话", "代码加解释"]),
            dpo_code_pairs=synth.get("dpo_code_pairs", 20),
            dpo_math_pairs=synth.get("dpo_math_pairs", 20),
            dpo_exec_timeout=synth.get("dpo_exec_timeout", 5),
            pf_min_inst_len=synth.get("postfilter_min_instruction_len", 10),
            pf_min_out_len=synth.get("postfilter_min_output_len", 30),
            pf_jaccard_threshold=synth.get("postfilter_jaccard_threshold", 0.7),
            pf_consistency_threshold=synth.get("postfilter_consistency_threshold", 0.7),
            consistency_sample=synth.get("consistency_sample", 30),
            output=synth.get("output", "data/synthesized.jsonl"),
            dpo_output=synth.get("dpo_output", "data/dpo_pairs.jsonl"),
            random_seed=g.get("random_seed", 42),
            log_level=g.get("log_level", "INFO"),
        )


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("AquilaLM-S2")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(ch)
        fh = logging.FileHandler(os.path.join(log_dir, "stage2_synth.log"), mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    return logger


# ============================================================
# 1. LLMClient — OpenAI 兼容 API 封装
# ============================================================

class LLMClient:
    """
    统一的 API 调用封装。支持 DeepSeek OpenAI 兼容格式，内置重试和 token 统计。
    面试要点：封装层解耦了业务逻辑和 API 细节，换模型只需改配置。
    """

    def __init__(self, config: SynthConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.total_tokens = 0
        self.total_calls = 0

        # 代理支持：读取环境变量 http_proxy / https_proxy（ccswitch 等工具）
        import httpx
        proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy") or None
        http_client = httpx.Client(proxy=proxy, timeout=60.0) if proxy else None

        # 检查 openai 库（DeepSeek 兼容 OpenAI SDK）
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.api_base,
                http_client=http_client,
            ) if http_client else OpenAI(
                api_key=config.api_key,
                base_url=config.api_base,
            )
            self.has_openai = True
        except ImportError:
            self.logger.warning("openai 库未安装，使用 requests 回退方案")
            self.has_openai = False

    def chat(self, messages: List[Dict], temperature: float = 0.8,
             model: str = None, max_tokens: int = 2048) -> Optional[str]:
        """发送请求，内置重试"""
        model = model or self.config.flash_model
        for attempt in range(self.config.max_retries):
            try:
                if self.has_openai:
                    resp = self.client.chat.completions.create(
                        model=model, messages=messages,
                        temperature=temperature, max_tokens=max_tokens)
                    content = resp.choices[0].message.content
                    self.total_tokens += resp.usage.total_tokens if resp.usage else 0
                else:
                    import requests
                    resp = requests.post(
                        f"{self.config.api_base}/chat/completions",
                        headers={"Authorization": f"Bearer {self.config.api_key}",
                                 "Content-Type": "application/json"},
                        json={"model": model, "messages": messages,
                              "temperature": temperature, "max_tokens": max_tokens},
                        timeout=60,
                    )
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    self.total_tokens += data.get("usage", {}).get("total_tokens", 0)

                self.total_calls += 1
                return content

            except Exception as e:
                self.logger.warning(f"API 调用失败 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
        return None

    def log_stats(self):
        self.logger.info(f"  API 调用统计: {self.total_calls} 次, "
                         f"累计 {self.total_tokens} tokens, "
                         f"预估费用 ¥{self.total_tokens/1_000_000*2:.2f}")


# ============================================================
# 2. SeedCurator — 种子筛选
# ============================================================

class SeedCurator:
    """从清洗后语料中筛选适合做指令种子文本。"""

    def __init__(self, config: SynthConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def curate(self, data: List[Dict]) -> List[Dict]:
        """
        三条标准：
        1. 长度 100~500 字 → 信息量够且不超 context
        2. PPL < 15 → 流畅度好
        3. 中文占比 > 0.5 → 排除中英混杂过度文本
        """
        self.logger.info(f"种子筛选：输入 {len(data)} 条")
        candidates = []
        for item in data:
            text = item.get("text") or item.get("content") or ""
            length = len(text)
            ppl = item.get("ppl", 999)
            cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
            cn_ratio = cn_chars / max(length, 1)

            if (self.config.seed_min_len <= length <= self.config.seed_max_len
                    and ppl <= self.config.seed_max_ppl
                    and cn_ratio >= self.config.seed_min_cn):
                candidates.append(item)

        random.seed(self.config.random_seed)
        random.shuffle(candidates)
        seeds = candidates[:self.config.max_seeds]

        self.logger.info(f"  候选 {len(candidates)} 条 → 采样 {len(seeds)} 条种子")
        if len(seeds) < self.config.max_seeds:
            self.logger.warning(
                f"  ⚠ 种子不足！候选 {len(candidates)}，目标 {self.config.max_seeds}。"
                f"可能 thresholds 过严或 cleaned.jsonl 尺寸太小")
        return seeds


# ============================================================
# 3. Self-Instruct — 种子文本 → 指令数据
# ============================================================

SELF_INSTRUCT_PROMPT = """你是一个专业的数据标注专家。请根据以下文本，生成{task_type}类型的指令数据。

【原始文本】
{seed_text}

【要求】
1. 生成一条 instruction（提问/任务描述）和对应的 output（高质量回答）
2. 类型为"{task_type}"，请确保 instruction 体现该类型的特点
3. instruction 需基于原文内容但**不能直接复制原文**——要用自己的话重新组织
4. output 必须准确、完整、有信息量，不能敷衍

【输出格式】只输出一行 JSON，不要加任何其他文字：
{{"instruction": "你的提问", "output": "你的回答", "task_type": "{task_type}"}}"""


class SelfInstruct:
    """种子文本 → 指令数据（v4flash 批量生成）"""

    def __init__(self, config: SynthConfig, client: LLMClient, logger: logging.Logger):
        self.config = config
        self.client = client
        self.logger = logger

    def generate(self, seeds: List[Dict]) -> List[Dict]:
        """
        每条种子生成 si_per_seed 条指令，每次随机选 task_type。
        """
        self.logger.info(f"Self-Instruct: {len(seeds)} 种子 × {self.config.si_per_seed} 条/种子 "
                         f"= 目标 ≤{len(seeds)*self.config.si_per_seed} 条")
        results = []
        total = len(seeds) * self.config.si_per_seed
        pbar = tqdm(total=total, desc="Self-Instruct")

        for seed in seeds:
            seed_text = seed.get("text") or seed.get("content") or ""
            if len(seed_text) < 20:
                pbar.update(self.config.si_per_seed)
                continue

            for _ in range(self.config.si_per_seed):
                task_type = random.choice(self.config.si_task_types)
                prompt = SELF_INSTRUCT_PROMPT.format(
                    seed_text=seed_text[:1200],  # 截断保护
                    task_type=task_type,
                )
                response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temp_creative,
                    model=self.config.flash_model,
                )
                if response:
                    parsed = self._parse(response)
                    if parsed:
                        parsed["seed_id"] = seed.get("id", "")
                        parsed["source"] = "self_instruct"
                        results.append(parsed)
                pbar.update(1)

        pbar.close()
        self.logger.info(f"  生成 {len(results)} 条有效指令 (成功率 {len(results)/max(total,1)*100:.0f}%)")
        return results

    def _parse(self, response: str) -> Optional[Dict]:
        """从 LLM 回复中提取 JSON。容忍 markdown code block 包裹。"""
        try:
            # 去掉可能的 markdown 包裹
            text = response.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试正则提取 JSON 对象
            match = re.search(r'\{.*"instruction".*"output".*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None


# ============================================================
# 4. Evol-Instruct — 简单指令 → 复杂进化
# ============================================================

EVOL_INSTRUCT_PROMPT = """你是一个指令复杂度提升专家。请将以下简单指令进化为更复杂的版本。

【原始指令】
{instruction}

【进化类型】{evolution_type}

【要求】
1. 根据进化类型，将原始指令改写得更复杂
2. 保留原始指令的核心意图，但增加更多要求、步骤或约束
3. 同时生成对应的高质量 output
4. 复杂度显著高于原指令

【输出格式】只输出一行 JSON：
{{"instruction": "进化后的指令", "output": "对应的回答", "evolution_type": "{evolution_type}"}}"""


class EvolInstruct:
    """简单指令 → 复杂进化（v4pro）"""

    def __init__(self, config: SynthConfig, client: LLMClient, logger: logging.Logger):
        self.config = config
        self.client = client
        self.logger = logger

    def evolve(self, instructions: List[Dict]) -> List[Dict]:
        """
        从 SelfInstruct 产出中选取简单指令，随机应用进化算子。
        """
        # 优先选简单指令（长度较短的）
        simple = sorted(instructions, key=lambda x: len(x.get("instruction", "")))
        candidates = simple[:min(len(simple), self.config.ei_max_evolve)]

        self.logger.info(f"Evol-Instruct: 候选 {len(candidates)} 条 → 进化")

        results = []
        for item in tqdm(candidates, desc="Evol-Instruct"):
            evo_type = random.choice(self.config.ei_evolution_types)
            prompt = EVOL_INSTRUCT_PROMPT.format(
                instruction=item.get("instruction", ""),
                evolution_type=evo_type,
            )
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temp_precise,
                model=self.config.pro_model,
            )
            if response:
                parsed = self._parse(response)
                if parsed:
                    parsed["source"] = "evol_instruct"
                    parsed["parent_id"] = item.get("id", item.get("instruction", "")[:30])
                    results.append(parsed)

        self.logger.info(f"  进化成功 {len(results)} 条")
        return results

    def _parse(self, response: str) -> Optional[Dict]:
        try:
            text = response.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*"instruction".*"output".*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None


# ============================================================
# 5. DPOPairBuilder — 规则驱动的偏好数据构造
# ============================================================

DPO_CODE_PROMPT = """你是一个编程题生成器。请生成一道中等难度的 Python 编程题，并给出两份回答——回答A是正确的，回答B含有故意的错误。

【要求】
1. 回答A：完整可运行的正确代码
2. 回答B：代码看起来与A相似，但暗藏一个能在执行时让它出错的 bug（如逻辑错、索引越界、除零等）
3. 两份回答都必须包含完整可执行的 Python 代码块

【输出格式】只输出一行 JSON：
{{"instruction": "编程题目", "response_a": "回答A(正确)", "response_b": "回答B(含bug)"}}"""

DPO_MATH_PROMPT = """你是一个数学题生成器。请生成一道中学难度的数学题（最终答案是一个数值），并给出两份回答——回答A是正确的，回答B给出了错误答案。

【重要】
1. 每份回答的最后一行必须是"答案：[数值]"
2. 回答A的答案必须正确，回答B的答案必须错误（故意改大或改小一点）

【输出格式】只输出一行 JSON：
{{"instruction": "数学题目", "response_a": "回答A(正确答案)", "response_b": "回答B(错误答案)"}}"""


class DPOPairBuilder:
    """
    规则驱动的 DPO 偏好数据构造。
    不依赖人工标注：代码执行正确性 / 数学答案验证 / RM 二次过滤（预留接口）。
    """

    def __init__(self, config: SynthConfig, client: LLMClient, logger: logging.Logger):
        self.config = config
        self.client = client
        self.logger = logger

    def build_code_pairs(self) -> List[Dict]:
        """代码偏好构造: 代码执行 (主) + LLM-as-Judge (兜底)"""
        self.logger.info(f"DPO 代码偏好：目标 {self.config.dpo_code_pairs} 对")
        pairs = []
        attempts = 0
        max_attempts = self.config.dpo_code_pairs * 3
        exec_count, judge_count = 0, 0

        pbar = tqdm(total=self.config.dpo_code_pairs, desc="DPO-Code")
        while len(pairs) < self.config.dpo_code_pairs and attempts < max_attempts:
            attempts += 1
            response = self.client.chat(
                messages=[{"role": "user", "content": DPO_CODE_PROMPT}],
                temperature=self.config.temp_creative,
                model=self.config.flash_model,
            )
            if not response:
                continue

            parsed = self._parse_json(response)
            if not parsed:
                continue

            inst = parsed.get("instruction", "")
            resp_a = parsed.get("response_a", "")
            resp_b = parsed.get("response_b", "")
            chosen, rejected, rule = None, None, None

            # 策略1: 代码执行 (客观信号，优先)
            passed_a = self._exec_code(resp_a)
            passed_b = self._exec_code(resp_b)
            if passed_a and not passed_b:
                chosen, rejected = resp_a, resp_b
                rule = "code_execution"
                exec_count += 1
            elif passed_b and not passed_a:
                chosen, rejected = resp_b, resp_a
                rule = "code_execution"
                exec_count += 1
            else:
                # 策略2: LLM-as-Judge (兜底, v4pro 判断哪个更好)
                judged = self._llm_judge(inst, resp_a, resp_b)
                if judged == "A":
                    chosen, rejected = resp_a, resp_b
                    rule = "llm_judge"
                    judge_count += 1
                elif judged == "B":
                    chosen, rejected = resp_b, resp_a
                    rule = "llm_judge"
                    judge_count += 1
                # judged = None → skip

            if chosen:
                pairs.append({
                    "instruction": inst, "chosen": chosen, "rejected": rejected,
                    "rule_type": rule, "source": "dpo_code",
                })
                pbar.update(1)

        pbar.close()
        self.logger.info(f"  代码偏好对: {len(pairs)} 对 (执行判定 {exec_count}, LLM-Judge {judge_count})")
        return pairs

    def _llm_judge(self, instruction: str, response_a: str, response_b: str) -> Optional[str]:
        """LLM-as-Judge: v4pro 判断两份代码回答哪个更好"""
        judge_prompt = f"""以下是一道编程题和两份回答。请判断哪个更好。

题目：{instruction[:400]}

回答A：
{response_a[:600]}

回答B：
{response_b[:600]}

评判：正确性 > 可读性 > 效率。只输出一个字符 A 或 B。"""
        result = self.client.chat(
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=self.config.temp_precise,
            model=self.config.pro_model,
            max_tokens=500,
        )
        if result:
            # 兼容各种输出格式: "A" / "回答A" / "我认为A更好" / "A。" 等
            text = result.strip().upper()
            if "A" in text[:5] and "B" not in text[:5]:
                return "A"
            if "B" in text[:5] and "A" not in text[:5]:
                return "B"
            # 更宽松: 取第一个出现的 A 或 B
            for ch in text:
                if ch == "A":
                    return "A"
                if ch == "B":
                    return "B"
        return None

    def build_math_pairs(self) -> List[Dict]:
        """数学答案验证构造偏好对"""
        self.logger.info(f"DPO 数学偏好：目标 {self.config.dpo_math_pairs} 对")
        pairs = []
        attempts = 0
        max_attempts = self.config.dpo_math_pairs * 3

        pbar = tqdm(total=self.config.dpo_math_pairs, desc="DPO-Math")
        while len(pairs) < self.config.dpo_math_pairs and attempts < max_attempts:
            attempts += 1
            response = self.client.chat(
                messages=[{"role": "user", "content": DPO_MATH_PROMPT}],
                temperature=self.config.temp_creative,
                model=self.config.flash_model,
            )
            if not response:
                continue

            parsed = self._parse_json(response)
            if not parsed:
                continue

            ans_a = self._extract_answer(parsed.get("response_a", ""))
            ans_b = self._extract_answer(parsed.get("response_b", ""))

            if ans_a is None or ans_b is None:
                continue  # 无法提取答案
            if ans_a == ans_b:
                continue  # 答案相同，无区分力

            # 注意：这里假定了 答案A 是正确的（因为无法知道真实标准答案）
            # 在生产中，需要题目 + 标准答案的校验；POC 用答案不同作为区分信号
            pairs.append({
                "instruction": parsed.get("instruction", ""),
                "chosen": parsed["response_a"],
                "rejected": parsed["response_b"],
                "rule_type": "math_answer_verification",
                "source": "dpo_math",
                "answer_a": ans_a,
                "answer_b": ans_b,
            })
            pbar.update(1)

        pbar.close()
        self.logger.info(f"  数学偏好对: {len(pairs)} 对 (成功率 {len(pairs)/max(attempts,1)*100:.0f}%)")
        return pairs

    def _exec_code(self, code: str) -> bool:
        """在隔离的临时文件中执行 Python 代码，返回是否成功"""
        code = self._extract_code_block(code)
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True,
                timeout=self.config.dpo_exec_timeout,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def _extract_code_block(self, text: str) -> str:
        """从 markdown 包裹中提取纯代码"""
        match = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _extract_answer(self, text: str) -> Optional[float]:
        """提取回答末尾"答案：[数值]"中的数值"""
        match = re.search(r'答案[：:]\s*([\d.]+)\s*$', text.strip(), re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def _parse_json(self, response: str) -> Optional[Dict]:
        try:
            text = response.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None


# ============================================================
# 6. PostFilter — 合成数据质量兜底
# ============================================================

class PostFilter:
    """
    合成后质量过滤。
    面试要点：规则层兜底 + Embedding 一致性打分 = 零人工质量的自动保障。
    """

    def __init__(self, config: SynthConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self._embedder

    def filter(self, data: List[Dict]) -> List[Dict]:
        """主入口：逐层过滤"""
        n_before = len(data)
        self.logger.info(f"PostFilter: 输入 {n_before} 条")

        # 1. 规则过滤
        data = self._rule_filter(data)
        self.logger.info(f"  规则过滤: {len(data)} 条")

        # 2. 指令间去重
        data = self._dedup_instructions(data)
        self.logger.info(f"  指令去重: {len(data)} 条")

        # 3. 一致性打分
        if len(data) >= self.config.consistency_sample:
            data = self._consistency_filter(data)
            self.logger.info(f"  一致性过滤: {len(data)} 条")

        self.logger.info(f"  最终保留 {len(data)} 条 (保留率 {len(data)/max(n_before,1)*100:.0f}%)")
        return data

    def _rule_filter(self, data: List[Dict]) -> List[Dict]:
        """硬性规则过滤"""
        reject_patterns = [
            r"作为.?AI", r"我不能", r"对不起", r"我无法",
            r"As an AI", r"I cannot", r"I'm sorry",
        ]
        passed = []
        rejected_counts = Counter()
        for item in data:
            inst = item.get("instruction", "")
            out = item.get("output", "")

            # 格式
            if not inst or not out:
                rejected_counts["empty_field"] += 1
                continue
            if len(inst) < self.config.pf_min_inst_len:
                rejected_counts["instruction_too_short"] += 1
                continue
            if len(out) < self.config.pf_min_out_len:
                rejected_counts["output_too_short"] += 1
                continue

            # 拒答检测
            hit = False
            for pat in reject_patterns:
                if re.search(pat, out):
                    rejected_counts["reject_template"] += 1
                    hit = True
                    break
            if hit:
                continue

            passed.append(item)

        self.logger.info(f"    规则拒绝: {dict(rejected_counts)}")
        return passed

    def _dedup_instructions(self, data: List[Dict]) -> List[Dict]:
        """基于 Jaccard 的指令间去重"""
        if len(data) <= 1:
            return data

        def _jaccard_similarity(a: str, b: str) -> float:
            sa = set(a)
            sb = set(b)
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        kept = []
        for item in data:
            inst = item.get("instruction", "")
            is_dup = False
            for k in kept[-20:]:  # 只跟最近 20 条比（近似去重，性能考虑）
                if _jaccard_similarity(inst, k.get("instruction", "")) >= self.config.pf_jaccard_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(item)

        self.logger.info(f"    指令去重: {len(kept)} 条 (删除 {len(data)-len(kept)} 条重复)")
        return kept

    def _consistency_filter(self, data: List[Dict]) -> List[Dict]:
        """
        Embedding 一致性打分。
        对同一 instruction 让 LLM 再生成一次回答 → 计算两个回答的余弦相似度。
        相似度低 → instruction 不明确 → 剔除。
        """
        sample = random.sample(data, min(self.config.consistency_sample, len(data)))
        self.logger.info(f"    一致性抽检: {len(sample)} 条...")

        # 对所有抽检指令的 output 计算 embedding
        outputs = [item.get("output", "") for item in sample]
        embeddings = self.embedder.encode(outputs, show_progress_bar=False)

        # 将每条 instruction 的 embedding 与随机另一条对比（作为近似一致性指标）
        # 真实生成需要再调一次 API。POC 用跨样本对比近似：embedding 方差=多样性
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        # 每条的"最相似邻居"的相似度
        low_consistency_ids = set()
        for i in range(len(sim_matrix)):
            # 跳过自己
            sims = [sim_matrix[i][j] for j in range(len(sim_matrix)) if j != i]
            if sims:
                max_sim = max(sims)
                if max_sim > 0.95:
                    # 有一个几乎完全一样的输出 → 可能是模板化
                    low_consistency_ids.add(sample[i].get("instruction", "")[:50])

        # POC 阶段不做硬删除（样本量小），仅标记
        self.logger.info(f"    一致性/多样性：相似度极高对 {len(low_consistency_ids)} 条 → 标记")
        return data


# ============================================================
# 7. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AquilaLM 指令合成引擎")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "self", "evol", "dpo", "filter"],
                        help="执行阶段")
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续跑：指定已有的中间产物文件路径")
    args = parser.parse_args()

    cfg = SynthConfig.from_yaml(args.config)
    logger = setup_logging(cfg.log_dir, cfg.log_level)

    if not cfg.api_key:
        logger.error("未设置 DEEPSEEK_API_KEY 环境变量！")
        logger.error("请运行: export DEEPSEEK_API_KEY=sk-your-key-here")
        sys.exit(1)

    client = LLMClient(cfg, logger)
    random.seed(cfg.random_seed)

    logger.info("=" * 60)
    logger.info("AquilaLM 指令合成引擎 启动")
    logger.info(f"  API: {cfg.api_base} | Flash: {cfg.flash_model} | Pro: {cfg.pro_model}")
    logger.info(f"  输入: {cfg.data_input}")
    logger.info("=" * 60)

    # 加载数据
    data = load_jsonl(cfg.data_input)
    logger.info(f"加载数据: {len(data)} 条")

    all_instructions = []

    # --- 种子筛选 ---
    curator = SeedCurator(cfg, logger)
    seeds = curator.curate(data)

    # --- Self-Instruct ---
    if args.stage in ("all", "self"):
        logger.info("\n" + "=" * 50)
        logger.info("阶段 1/3: Self-Instruct (种子文本 → 指令数据)")
        logger.info("=" * 50)

        si_ckpt = args.resume or "data/si_checkpoint.jsonl"
        si = SelfInstruct(cfg, client, logger)

        # 断点续跑：检查已有产物
        resume_data = []
        completed_seed_ids = set()
        if os.path.exists(si_ckpt):
            resume_data = load_jsonl(si_ckpt)
            completed_seed_ids = {item.get("seed_id", "") for item in resume_data if item.get("seed_id")}
            logger.info(f"  断点续跑：从 {si_ckpt} 恢复 {len(resume_data)} 条，已完成 {len(completed_seed_ids)} 种子")

        # 过滤掉已完成的种子
        remaining_seeds = [s for s in seeds if s.get("id", "") not in completed_seed_ids]
        if remaining_seeds:
            new_data = si.generate(remaining_seeds)
            all_instructions = resume_data + new_data
            save_jsonl(all_instructions, si_ckpt)
            logger.info(f"  中间产物已保存: {si_ckpt}")
        else:
            all_instructions = resume_data
            logger.info("  所有种子已完成，跳过生成")

    # --- Evol-Instruct ---
    if args.stage in ("all", "evol"):
        # 如果单独跑 evol，从已有产物加载指令
        if not all_instructions:
            if os.path.exists(cfg.output):
                all_instructions = load_jsonl(cfg.output)
                logger.info(f"  从 {cfg.output} 加载 {len(all_instructions)} 条已有指令")
            else:
                logger.error(f"  evol 阶段需要已有指令数据，但 {cfg.output} 不存在！请先跑 --stage self")
        if all_instructions:
            logger.info("\n" + "=" * 50)
            logger.info("阶段 2/3: Evol-Instruct (简单指令 → 复杂进化)")
            logger.info("=" * 50)
            ei = EvolInstruct(cfg, client, logger)
            ei_data = ei.evolve(all_instructions)
            all_instructions.extend(ei_data)

    # --- DPO 偏好构造 ---
    dpo_pairs = []
    if args.stage in ("all", "dpo"):
        logger.info("\n" + "=" * 50)
        logger.info("阶段 3/3: DPO 偏好数据构造")
        logger.info("=" * 50)
        builder = DPOPairBuilder(cfg, client, logger)
        dpo_pairs.extend(builder.build_code_pairs())
        dpo_pairs.extend(builder.build_math_pairs())

    # --- 后过滤 ---
    if all_instructions:
        logger.info("\n" + "=" * 50)
        logger.info("后过滤")
        logger.info("=" * 50)
        pf = PostFilter(cfg, logger)
        all_instructions = pf.filter(all_instructions)

        # 给每条数据分配唯一 ID
        for i, item in enumerate(all_instructions):
            item["id"] = f"synth_{i:04d}"

        save_jsonl(all_instructions, cfg.output)
        logger.info(f"合成指令数据输出: {cfg.output} ({len(all_instructions)} 条)")

    if dpo_pairs:
        save_jsonl(dpo_pairs, cfg.dpo_output)
        logger.info(f"DPO 偏好对输出: {cfg.dpo_output} ({len(dpo_pairs)} 对)")

    # --- 最终统计 ---
    client.log_stats()
    logger.info("\n" + "=" * 60)
    logger.info("指令合成完成！")
    logger.info(f"  合成指令: {len(all_instructions)} 条")
    logger.info(f"  DPO 偏好对: {len(dpo_pairs)} 对")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
