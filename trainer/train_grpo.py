import os
import sys
import re
import gc
import argparse
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import Self_Minimindconfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, SkipBatchSampler, init_model,
)

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════════
# 1. 奖励计算
# ────────────────────────────────────────────────────────────────────────────────
# 与 PPO 版本的区别：每个 prompt 有 num_generations 条 response，
#   奖励计算时需要按 (prompt_idx, gen_idx) 双层遍历
# ════════════════════════════════════════════════════════════════════════════════

def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """计算每条 response 的综合奖励 → [B * num_generations] tensor"""

    def reasoning_format_reward(rewards):
        pat1 = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pat2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        fmt_scores = [0.5 if (re.match(pat1, r, re.S) or re.match(pat2, r, re.S)) else 0.0
                      for r in responses]
        rewards += torch.tensor(fmt_scores, device=args.device)

        tag_scores = [sum(0.25 for tag in ["<think>", "</think>", "<answer>", "</answer>"]
                         if r.count(tag) == 1) for r in responses]
        rewards += torch.tensor(tag_scores, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        rewards = reasoning_format_reward(rewards)

    # ── Reward Model 评分 ─────────────────────────────────────
    # 【GRPO 特有】每个 prompt 对应 num_generations 条 response，需双层遍历
    with torch.no_grad():
        rm_scores = []
        scale = 3.0
        for i, prompt in enumerate(prompts):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            for j in range(args.num_generations):
                response = responses[i * args.num_generations + j]
                chat = messages + [{"role": "assistant", "content": response}]
                score = max(min(reward_model.get_score(reward_tokenizer, chat), scale), -scale)

                if args.reasoning == 1:
                    ans = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                    if ans:
                        ans_chat = messages + [{"role": "assistant", "content": ans.group(1).strip()}]
                        ans_score = max(min(reward_model.get_score(reward_tokenizer, ans_chat), scale), -scale)
                        score = score * 0.4 + ans_score * 0.6
                rm_scores.append(score)

        rewards += torch.tensor(rm_scores, device=args.device)
    return rewards


# ════════════════════════════════════════════════════════════════════════════════
# 2. Per-token log 概率计算
# ────────────────────────────────────────────────────────────────────────────────
# 【语法难点】logits_to_keep 参数：只计算最后 n_keep+1 个位置的 logits，
#   节省显存（completion 部分远短于 prompt+completion 整体）
#
# 【语法难点】is_inference() 检查：某些推理优化模式下 tensor 不可修改，
#   需要 detach().clone() 创建可写副本
# ════════════════════════════════════════════════════════════════════════════════

def get_per_token_logps(mdl, input_ids, n_keep):
    """计算序列最后 n_keep 个 token 的逐 token log 概率 → [B, n_keep]"""
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # logits_to_keep: 只计算最后 n_keep+1 个位置的 logits，节省显存
    logits = mdl(input_ids=input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]

    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        # gather: 从 log_softmax 中取出目标 token 的 log 概率
        token_logps = torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_logps)
    return torch.stack(per_token_logps)


# ════════════════════════════════════════════════════════════════════════════════
# 3. GRPO 单 Epoch 训练
# ────────────────────────────────────────────────────────────────────────────────
# GRPO vs PPO 的核心区别：
#   PPO:  需要 Critic 网络估计 V(s)，advantage = R - V(s)
#   GRPO: 无 Critic，每个 prompt 生成一组回复，组内标准化作为 advantage
#         → 用 "组内相对排名" 代替 Critic 的价值估计
#
# 【理论重点】GRPO 损失函数（逐 token 计算）：
#   per_token_loss = -(ratio · advantage - β · KL)
#   其中：
#     ratio = exp(log π_θ - log π_θ.detach())  ← 梯度只从 log π_θ 流过
#     KL = exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
#        ← 非负 KL 估计器，比直接用 log 差更稳定
# ════════════════════════════════════════════════════════════════════════════════

def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer,
                     start_step=0, wandb=None):

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        # ── Step 1: Tokenize + 截断 prompt ────────────────────
        prompt_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            return_token_type_ids=False, add_special_tokens=False,
        ).to(args.device)

        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ── Step 2: Rollout（每个 prompt 生成 num_generations 条回复）──
        # 【GRPO 特有】num_return_sequences > 1，一次生成一组候选回复
        with torch.no_grad():
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = raw_model.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len,
                do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = prompt_inputs["input_ids"].size(1)
        completion_ids = outputs[:, prompt_len:]

        # ── Step 3: 计算 policy 和 ref 的逐 token log 概率 ────
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))

        # ── Step 4: 奖励 → 组内标准化 → 优势 ──────────────────
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer)

        # 【理论重点】GRPO 核心：组内标准化替代 Critic
        #   将同一 prompt 的 num_generations 条回复视为一组
        #   advantage = (reward - group_mean) / group_std
        #   → 组内最好的回复获正优势，最差的获负优势
        grouped = rewards.view(-1, args.num_generations)
        mean_r = grouped.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped.std(dim=1).repeat_interleave(args.num_generations)
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 全局再标准化

        # ── Step 5: Completion mask（只计算到 EOS 为止）────────
        # 【语法难点】先找每条序列第一个 EOS 位置，再用广播构建 mask
        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device)
                           .expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        # ── Step 6: GRPO 损失 ─────────────────────────────────
        # 【理论重点】非负 KL 估计器：KL = exp(d) - d - 1，其中 d = log(π_ref/π_θ)
        #   相比直接用 d 作为 KL，该估计器保证非负，数值更稳定
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        # 【理论重点】ratio = exp(logp - logp.detach()) 恒等于 1，但保留了梯度
        #   等价于让 advantage 作为 logp 的梯度系数，同时通过 detach 阻止
        #   advantage 的梯度回传 —— 这是 GRPO 简化版的策略梯度实现
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach())
                           * advantages.unsqueeze(1) - args.beta * per_token_kl)

        loss = ((per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1)).mean() / args.accumulation_steps
        loss.backward()

        # ── Step 7: 梯度裁剪 + 参数更新 ──────────────────────
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()

        # ── 日志 ──────────────────────────────────────────────
        if step % args.log_interval == 0 or step == iters:
            log_dict = dict(
                policy_loss=loss.item(), reward=rewards.mean().item(),
                avg_response_len=completion_mask.sum(dim=1).float().mean().item(),
                advantages_mean=advantages.mean().item(),
                learning_rate=optimizer.param_groups[0]["lr"],
            )
            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Loss: {log_dict['policy_loss']:.6f}, Reward: {log_dict['reward']:.6f}, "
                   f"Avg Len: {log_dict['avg_response_len']:.2f}, LR: {log_dict['learning_rate']:.2e}")
            if wandb and is_main_process():
                wandb.log(log_dict)

        # ── 模型保存 ──────────────────────────────────────────
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            src = model.module if isinstance(model, DistributedDataParallel) else model
            torch.save({k: v.half() for k, v in src.state_dict().items()}, ckp)
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                epoch=epoch, step=step, wandb=wandb, save_dir="../checkpoints", scheduler=scheduler,
            )
            model.train()

        # ── 显存清理（GRPO 每步显存开销大：B * num_generations 条序列）──
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped, advantages, completion_mask
        torch.cuda.empty_cache()
        gc.collect()


# ════════════════════════════════════════════════════════════════════════════════
# 4. 主函数入口
# ────────────────────────────────────────────────────────────────────────────────
# GRPO 只需 3 个模型（vs PPO 的 5 个）：
#   Policy(训练) / Reference(冻结) / Reward Model(外部,冻结)
#   无 Critic、无 Old Actor → 更简单，但依赖多次采样估计优势
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AquilaLM GRPO")

    # 基础训练
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="grpo", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=8e-8)

    # 硬件
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)

    # 训练策略
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)

    # 模型架构
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])

    # 生成参数
    parser.add_argument("--max_seq_len", default=66, type=int, help="Prompt 最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536)

    # GRPO 专用参数
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl")
    parser.add_argument("--num_generations", type=int, default=8, help="每个 prompt 的采样数(组大小)")
    parser.add_argument("--beta", type=float, default=0.02, help="KL 惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])

    # 实验跟踪
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AquilaLM-GRPO")

    args = parser.parse_args()

    # ── 1. 环境初始化 ──────────────────────────────────────────
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ── 2. 模型配置 ───────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = Self_Minimindconfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
                if args.from_resume == 1 else None)

    # ── 3. Wandb ──────────────────────────────────────────────
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.wandb_project,
            name=f"GRPO-E{args.epochs}-BS{args.batch_size}-G{args.num_generations}",
            id=wandb_id, resume="must" if wandb_id else None,
        )

    # ── 4. 初始化 3 个模型 ────────────────────────────────────
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    # Policy 模型（训练中）
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reference 模型（冻结，提供 KL 惩罚基准）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model.eval().requires_grad_(False)

    # Reward Model（外部模型，冻结）
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # ── 5. 数据 & 优化器 ─────────────────────────────────────
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_position_embeddings)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    iters = len(DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler))
    total_steps = max(1, (iters // args.accumulation_steps) * args.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)

    # ── 6. Checkpoint 恢复 ───────────────────────────────────
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ── 7. DDP 包装 ──────────────────────────────────────────
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ── 8. 训练循环 ──────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f"Epoch [{epoch+1}/{args.epochs}]: 从 step {start_step+1} 续训")
            grpo_train_epoch(epoch, loader, len(loader) + start_step, ref_model,
                             reward_model, reward_tokenizer, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            grpo_train_epoch(epoch, loader, len(loader), ref_model,
                             reward_model, reward_tokenizer, 0, wandb)