import os
import sys
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import Self_MinimindConfig 
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode,
    setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings("ignore")  # 保持输出清洁


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """SFT训练核心循环（精简注释版）"""
    start_time = time.time()
    
    for step, (input_ids, labels, attention_mask) in enumerate(loader, start=start_step + 1):
        # 🔥 SFT核心区别1：直接使用labels（监督信号明确）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)  # 必须传入，防止padding干扰
        
        # 🌟 SFT核心区别2：模型直接返回loss（内置loss计算）
        with autocast_ctx:
            outputs = model(
                input_ids, 
                labels=labels,           # 关键：监督信号
                attention_mask=attention_mask  # 关键：屏蔽padding
            )
            # SFT总loss = 任务loss + MoE辅助loss（如有）
            loss = outputs.loss + (outputs.aux_loss if outputs.aux_loss is not None else 0.0)
            loss = loss / args.accumulation_steps  # 梯度累积均摊
        
        # 混合精度反向传播（同Pretrain）
        scaler.scale(loss).backward()
        
        # 梯度累积达到阈值则更新（同Pretrain）
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 日志与保存（同Pretrain，略）
        if step % args.log_interval == 0 or step == iters:
            _log_training_stats(epoch, step, iters, loss, outputs, start_time, wandb)
        
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            _save_checkpoint(epoch, step, wandb)


def _log_training_stats(epoch, step, total_steps, loss, outputs, start_time, wandb):
    """封装日志逻辑，提升主循环可读性"""
    spend_time = time.time() - start_time
    current_loss = loss.item() * args.accumulation_steps
    current_aux_loss = outputs.aux_loss.item() if outputs.aux_loss is not None else 0.0
    current_logits_loss = current_loss - current_aux_loss
    current_lr = optimizer.param_groups[-1]["lr"]
    eta_min = spend_time / (step + 1) * total_steps // 60 - spend_time // 60
    
    log_msg = (
        f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{total_steps}), "
        f"loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, "
        f"aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, "
        f"eta: {eta_min:.1f}min"
    )
    Logger(log_msg)
    
    if wandb:
        wandb.log({
            "loss": current_loss,
            "logits_loss": current_logits_loss,
            "aux_loss": current_aux_loss,
            "lr": current_lr,
            "epoch_time": eta_min,
        })


def _save_checkpoint(epoch, step, wandb=None):
    """封装检查点保存逻辑"""
    model.eval()
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp_path = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    
    # 获取真实模型（处理DDP/.compile包装）
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    
    # 半精度保存（同Pretrain）
    state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
    torch.save(state_dict, ckp_path)
    
    # 保存完整训练状态（优化器/epoch/step等）
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir="../checkpoints",
        scaler=scaler,
    )
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquilaLM Full SFT")
    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="full_sft", type=str)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    
    # ========== 硬件和性能 ==========
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--num_workers", type=int, default=8)
    
    # ========== 训练策略 ==========
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    
    # ========== 模型架构 ==========
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=340, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    
    # ========== 数据和恢复 ==========
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl")
    parser.add_argument("--from_weight", default="pretrain", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    
    # ========== 实验跟踪 ==========
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AquilaLM-Full-SFT")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    
    global args
    args = parser.parse_args()
    
    # ========== 1. 初始化环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置模型与检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = Self_MinimindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    
    # 尝试加载检查点（断点续训）
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") \
        if args.from_resume == 1 else None
    
    # ========== 3. 混合精度设置 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 初始化模型与数据 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 🌟 SFT核心区别3：使用SFTDataset（监督信号明确）
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 同Pretrain（AMP/优化器）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 5. 恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
    
    # ========== 6. DDP封装（注意：跳过RoPE缓存）==========
    if dist.is_initialized():
        # 🌟 为何跳过 freqs_cos/sin？
        # RoPE位置编码缓存是确定性计算结果，各卡相同，同步纯属冗余
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        Logger(f"DDP initialized on rank {local_rank}")
    
    # ========== 7. 启动训练 ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        setup_seed(42 + epoch)  # 每epoch不同随机种子
        indices = torch.randperm(len(train_ds)).tolist()
        
        # 处理断点续训
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        if skip > 0:
            Logger(f"Epoch {epoch+1}: 跳过前{skip}步，从step {skip+1}开始")
        
        train_epoch(epoch, loader, len(loader) + skip, start_step if epoch == start_epoch else 0)
        start_step = 0  # 后续epoch从0开始
    
    # ========== 8. 清理 ==========
    if dist.is_initialized():
        dist.destroy_process_group()