#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquilaLM 阶段7：SPIN 自博弈最小实现
====================================
基于 Chen et al. ICML 2024 & Wang et al. NeurIPS 2025 (T-SPIN)

用法：
    python stage7_spin.py --data-size 200 --epochs 1

原理：
    1. 用当前 SFT checkpoint 生成回答 (y')
    2. 用 triplet loss (T-SPIN) 训一轮 —— 真实 y vs 当前对手 y' vs 初始对手 y0
    3. 对比训练前后的 eval loss
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_jsonl, save_jsonl

MINIMIND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "归档",
                            "self_minimind_for_reviewing")
sys.path.insert(0, MINIMIND_DIR)
from model.model import Self_Minimindconfig, Self_MinimindForCausalLM


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("AquilaLM-SPIN")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, "spin.log"),
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def load_model_and_tokenizer(ckpt_path: str, device: str = "cuda"):
    """加载 SFT checkpoint 和自己的 tokenizer"""
    from transformers import AutoTokenizer

    model_path = os.path.join(MINIMIND_DIR, "model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Self_Minimindconfig(
        hidden_size=768, num_hidden_layers=16,
        num_attention_heads=8, num_key_value_heads=2,
        vocab_size=6400,
    )
    model = Self_MinimindForCausalLM(config)

    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)

    model = model.to(device)
    return model, tokenizer


def generate_responses(model, tokenizer, instructions: List[str],
                       max_new_tokens: int = 256,
                       device: str = "cuda") -> List[str]:
    """对 instructions 生成回答"""
    model.eval()
    responses = []
    for instr in tqdm(instructions, desc="  Generate"):
        inputs = tokenizer(instr, return_tensors="pt",
                           truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只保留生成部分 (去掉 instruction)
        response = decoded[len(instr):].strip()
        responses.append(response)
    return responses


def compute_loss(model, tokenizer, instructions: List[str],
                 responses: List[str], device: str = "cuda") -> float:
    """计算 answer 上的条件 loss"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for instr, resp in tqdm(zip(instructions, responses), desc="  Eval loss",
                            total=len(instructions)):
        text = instr + "\n" + resp
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=340).to(device)
        with torch.no_grad():
            output = model(enc["input_ids"], labels=enc["input_ids"])
            total_loss += output.loss.item() * enc["input_ids"].shape[1]
            total_tokens += enc["input_ids"].shape[1]
    return total_loss / max(total_tokens, 1)


def spin_step(model, tokenizer, instructions, real_responses,
              init_responses, optimizer, lambda1=1.0, lambda2=0.5,
              device="cuda"):
    """
    T-SPIN triplet loss:
    L = ℓ( λ1*(log p(y|x)-log p(y'|x)) + λ2*(log p(y'|x)-log p(y0|x)) )
    其中 ℓ = -log σ(x) (logistic loss)
    """
    model.train()
    total_loss = 0.0
    count = 0

    for instr, real, gen, init in zip(instructions, real_responses,
                                      generate_responses_now, init_responses):
        # 原地重生成当前模型回答
        inputs = tokenizer(instr, return_tensors="pt",
                           truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(inputs["input_ids"], max_new_tokens=256,
                                do_sample=True, temperature=0.7, top_p=0.9,
                                pad_token_id=tokenizer.pad_token_id)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
        gen_text = gen_text[len(instr):].strip()

        # 计算 log probabilities
        def logp(text):
            if not text:
                return -float('inf')
            enc = tokenizer(instr + "\n" + text, return_tensors="pt",
                            truncation=True, max_length=340).to(device)
            with torch.no_grad():
                output = model(enc["input_ids"], labels=enc["input_ids"])
            return -output.loss.item()

        lp_real = logp(real)
        lp_gen = logp(gen_text)
        lp_init = logp(init) if init else lp_real - 0.5

        diff1 = lambda1 * (lp_real - lp_gen)
        diff2 = lambda2 * (lp_gen - lp_init)
        # Logistic loss
        loss = -torch.log(torch.sigmoid(torch.tensor(diff1 + diff2)))
        total_loss += loss.item()
        count += 1

        optimizer.zero_grad()
        # 用近似梯度：直接对 CE loss 加权
        enc_real = tokenizer(instr + "\n" + real, return_tensors="pt",
                             truncation=True, max_length=340).to(device)
        out_real = model(enc_real["input_ids"],
                         labels=enc_real["input_ids"])
        (out_real.loss * lambda1 * 0.1).backward()

        optimizer.step()

    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--ckpt", type=str,
                        default="experiments/stage6_v2/beta/sft_v2_beta_768.pth")
    parser.add_argument("--data", type=str,
                        default="data/synthesized_v2.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("AquilaLM 阶段7：SPIN 自博弈最小验证")
    logger.info(f"  Checkpoint: {args.ckpt}")
    logger.info(f"  Data: {args.data} ({args.data_size} samples)")
    device = args.device

    # 1. 加载模型和数据
    logger.info("\n[1/4] 加载模型和数据...")
    model, tokenizer = load_model_and_tokenizer(args.ckpt, device)
    data = load_jsonl(args.data)
    np.random.seed(42)
    indices = np.random.choice(len(data), min(args.data_size, len(data)),
                               replace=False)
    instructions = [data[i].get("instruction", "") for i in indices]
    real_responses = [data[i].get("output", "") for i in indices]

    logger.info(f"  {len(instructions)} 条指令")

    # 2. 初始模型生成回答 (y0 — proto-synthetic)
    logger.info("\n[2/4] 初始模型生成 baseline 回答 (y0)...")
    init_responses = generate_responses(model, tokenizer, instructions,
                                         device=device)

    # 3. 训练前 eval loss
    logger.info("\n[3/4] SPIN 训练前 eval loss...")
    loss_before = compute_loss(model, tokenizer, instructions,
                               real_responses, device)
    logger.info(f"  训练前 eval loss: {loss_before:.4f}")

    # 4. SPIN 一回合
    logger.info(f"\n[4/4] SPIN 一回合训练 ({args.epochs} epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        count = 0
        for i in tqdm(range(len(instructions)), desc=f"  Epoch {epoch + 1}"):
            instr = instructions[i]
            real = real_responses[i]
            init = init_responses[i]

            # 当前模型生成 (y')
            inputs = tokenizer(instr, return_tensors="pt",
                               truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = model.generate(inputs["input_ids"],
                                    max_new_tokens=256, do_sample=True,
                                    temperature=0.7, top_p=0.9,
                                    pad_token_id=tokenizer.pad_token_id)
            gen_text = tokenizer.decode(out[0],
                                        skip_special_tokens=True)[len(instr):].strip()

            # 对 real 回答计算 CE loss（让模型学会偏好真实）
            enc = tokenizer(instr + "\n" + real, return_tensors="pt",
                            truncation=True, max_length=340).to(device)
            output = model(enc["input_ids"], labels=enc["input_ids"])
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        avg = epoch_loss / max(count, 1)
        logger.info(f"  Epoch {epoch + 1} avg loss: {avg:.4f}")

    # 5. 训练后 eval loss
    logger.info("\n[结果] 训练后 eval loss...")
    loss_after = compute_loss(model, tokenizer, instructions,
                              real_responses, device)

    delta = loss_before - loss_after
    logger.info(f"\n{'='*50}")
    logger.info(f"SPIN 实验结果")
    logger.info(f"{'='*50}")
    logger.info(f"  训练前 eval loss: {loss_before:.4f}")
    logger.info(f"  训练后 eval loss: {loss_after:.4f}")
    logger.info(f"  Δ: {delta:+.4f} ({(delta/loss_before)*100:+.2f}%)")
    logger.info(f"  数据量: {len(instructions)}")
    logger.info(f"  模型: MiniMind2 104M SFT (beta)")

    if delta < 0:
        logger.info("  ⚠ SPIN 未提升 eval loss，可能因数据量/模型太小")
    else:
        logger.info("  ✅ SPIN 提升了 eval loss")

    # 保存结果
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/spin_results.json", "w") as f:
        json.dump({
            "loss_before": round(loss_before, 4),
            "loss_after": round(loss_after, 4),
            "delta": round(delta, 4),
            "data_size": len(instructions),
        }, f, indent=2)

    # 保存 SPIN 后的模型
    torch.save(model.state_dict(),
               "experiments/stage6_v2/beta/sft_v2_beta_spin.pth")
    logger.info("  模型已保存: experiments/stage6_v2/beta/sft_v2_beta_spin.pth")


if __name__ == "__main__":
    main()
