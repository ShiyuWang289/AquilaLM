import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model import Self_Minimindconfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)


# ════════════════════════════════════════════════════════════════════════════════
# 1. 核心计算函数
# ────────────────────────────────────────────────────────────────────────────────
# 【理论重点】DPO 直接偏好优化：
#   不需要训练 Reward Model，直接用偏好数据对(chosen, rejected)优化策略
#   只需 2 个模型：Policy(训练) + Reference(冻结)
#   损失函数：L = -log σ(β · (log π/π_ref(y_w) - log π/π_ref(y_l)))
#   其中 y_w=chosen, y_l=rejected, β 控制优化强度
# ════════════════════════════════════════════════════════════════════════════════

def logits_to_log_probs(logits, labels):
    """词表 logits → 目标 token 的 log 概率 [B, L]
    
    【语法难点】gather(dim=2, index): 沿词表维取出 label 对应的 log 概率
      log_softmax → [B, L, V] 全词表 log 概率
      gather      → [B, L, 1] 只取目标 token
      squeeze(-1) → [B, L]
    """
    log_probs = F.log_softmax(logits, dim=2)                      # [B, L, V]
    return log_probs.gather(2, labels.unsqueeze(2)).squeeze(-1)    # [B, L]


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """DPO 损失计算
    
    【理论重点】DPO 公式推导的核心思想：
      1. 将 RLHF 的奖励函数 r(x,y) 用策略比率 log(π/π_ref) 替代
      2. Bradley-Terry 偏好模型：P(y_w > y_l) = σ(r(y_w) - r(y_l))
      3. 合并得：L = -log σ(β · ((log π_w - log π_l) - (log π_ref_w - log π_ref_l)))
    
    输入数据排列: 前半 batch = chosen, 后半 batch = rejected
    """
    # 按 mask 计算序列平均 log 概率（除以有效 token 数）
    seq_len = mask.sum(dim=1).clamp_min(1e-8)
    ref_avg = (ref_log_probs * mask).sum(dim=1) / seq_len
    policy_avg = (policy_log_probs * mask).sum(dim=1) / seq_len

    # 按 batch 前后半拆分 chosen / rejected
    half = ref_avg.shape[0] // 2
    pi_logratios = policy_avg[:half] - policy_avg[half:]      # log π(y_w) - log π(y_l)
    ref_logratios = ref_avg[:half] - ref_avg[half:]           # log π_ref(y_w) - log π_ref(y_l)

    # 【理论重点】DPO 损失 = -log σ(β · (策略偏好差 - 参考偏好差))
    #   策略偏好差 - 参考偏好差 = 策略相对参考的"额外偏好程度"
    #   β 越大 → 优化越激进，β 越小 → 越保守（常用 0.1~0.5）
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()


# ════════════════════════════════════════════════════════════════════════════════
# 2. 单 Epoch 训练
# ────────────────────────────────────────────────────────────────────────────────
# 每步流程：
#   拼接 chosen+rejected → ref 前向(frozen) → policy 前向 → DPO loss → 更新
# ════════════════════════════════════════════════════════════════════════════════

def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        # ── 数据准备：将 chosen 和 rejected 拼接为一个 batch ───
        # 【理论重点】DPO 需要同时计算 chosen 和 rejected 的概率，
        #   拼接后一次前向即可，前半 = chosen，后半 = rejected
        x = torch.cat([batch["x_chosen"], batch["x_rejected"]], dim=0).to(args.device)
        y = torch.cat([batch["y_chosen"], batch["y_rejected"]], dim=0).to(args.device)
        mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]], dim=0).to(args.device)
        attn_mask = torch.cat([batch["attention_mask_chosen"], batch["attention_mask_rejected"]], dim=0).to(args.device)

        # 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate) 
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with autocast_ctx: # 混合精度上下文
            # Reference 前向（冻结，不计算梯度）
            with torch.no_grad():
                ref_logits = ref_model(x, attention_mask=attn_mask).logits
            ref_lp = logits_to_log_probs(ref_logits, y)

            # Policy 前向（正在训练）
            outputs = model(x, attention_mask=attn_mask)
            policy_lp = logits_to_log_probs(outputs.logits, y)

            # DPO 损失 + MoE 辅助损失
            loss = dpo_loss(ref_lp, policy_lp, mask, beta=beta) + outputs.aux_loss
            loss = loss / args.accumulation_steps

        # ── 反向传播 + 梯度累积更新 ───────────────────────────
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # unscale作用：将梯度从 scaled space 转回正常范围，准备进行梯度裁剪和优化器更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer) # 优化器更新
            scaler.update() 
            optimizer.zero_grad(set_to_none=True)

        # ── 日志 ─────────────────────────────────────────────
        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - start_time
            eta_min = elapsed / (step + 1) * iters / 60 - elapsed / 60
            current_loss = loss.item() * args.accumulation_steps

            Logger(f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) "
                   f"loss:{current_loss:.6f} lr:{lr:.12f} eta:{eta_min:.0f}min")

            if wandb:
                wandb.log({"loss": current_loss, "lr": lr, "epoch_Time": eta_min})

        # ── 模型保存 ─────────────────────────────────────────
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            src = model.module if isinstance(model, DistributedDataParallel) else model
            torch.save({k: v.half() for k, v in src.state_dict().items()}, ckp)

            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch,
                step=step, wandb=wandb, save_dir="../checkpoints",
            )
            model.train()


# ════════════════════════════════════════════════════════════════════════════════
# 3. 主函数入口
# ────────────────────────────────────────────────────────────────────────────────
# DPO vs PPO：
#   DPO: 2个模型(Policy+Reference)，离线偏好数据，无需 Reward Model
#   PPO: 5个模型(Actor+OldActor+Critic+Ref+Reward)，在线 rollout
#   DPO 更简单高效，但 PPO 探索能力更强
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 参数定义 ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="AquilaLM DPO")

    # 基础训练
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="dpo", type=str)
    parser.add_argument("--epochs", type=int, default=1, help="DPO 通常 1-2 轮")
    parser.add_argument("--batch_size", type=int, default=4)
    # DPO 学习率极小(~4e-8)，防止过度优化导致遗忘
    parser.add_argument("--learning_rate", type=float, default=4e-8)

    # 硬件
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)

    # 训练策略
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)

    # 模型架构
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])

    # DPO 专用参数
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")
    parser.add_argument("--from_weight", default="full_sft", type=str, help="基于 SFT 模型训练")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    # β 控制偏好优化强度，常用 0.1~0.5
    parser.add_argument("--beta", default=0.1, type=float, help="DPO β 参数")

    # 实验跟踪
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AquilaLM-DPO")

    args = parser.parse_args()

    # ── 1. 环境初始化 ──────────────────────────────────────────
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ── 2. 模型配置 & Checkpoint 检测 ─────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = Self_Minimindconfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
                if args.from_resume == 1 else None)

    # ── 3. 混合精度 ───────────────────────────────────────────
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ── 4. Wandb ──────────────────────────────────────────────
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.wandb_project,
            name=f"DPO-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}",
            id=wandb_id, resume="must" if wandb_id else None,
        )

    # ── 5. 初始化 2 个模型 ────────────────────────────────────
    # DPO 只需 2 个模型，远比 PPO(5个) 简单：
    #   Policy: 正在训练的策略模型
    #   Reference: 冻结的 SFT 模型，提供 baseline 概率

    # Policy 模型（训练中）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f"Policy 模型参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    # Reference 模型（冻结，与 Policy 初始权重相同）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval().requires_grad_(False)
    Logger(f"Reference 模型参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M")

    # ── 6. 数据 & 优化器 ─────────────────────────────────────
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ── 7. 从 Checkpoint 恢复 ────────────────────────────────
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ── 8. DDP 包装 ──────────────────────────────────────────
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ── 9. 训练循环 ──────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f"Epoch [{epoch+1}/{args.epochs}]: 从 step {start_step+1} 续训")
            train_epoch(epoch, loader, len(loader) + start_step, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)