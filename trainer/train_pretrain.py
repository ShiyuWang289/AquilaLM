"""
🎯 学习优先级指南：
🔥 深入掌握：梯度累积 | 混合精度 | 分布式训练 | 断点续训
💡 建议了解：梯度裁剪 | 学习率调度 | 随机种子
🌱 了解即可：参数解析 | 日志打印 | 实验跟踪
"""
import os
import sys
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from datasets import disable_caching  # 避免datasets缓存占用磁盘

# =============== 🌱 了解即可：工具函数导入 ===============
# （掌握作用即可，无需深究实现）
from model.model import Self_MinimindConfig 
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

# 禁用警告和tokenizer多进程（避免DataLoader冲突）
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
disable_caching()  # 流式加载更高效


# =============== 🔥 深入掌握：训练核心循环 ===============
def train_epoch(epoch, loader, total_steps, start_step=0, wandb=None):
    """
    核心训练逻辑
    ✅ 重点掌握：梯度累积 | 混合精度 | 梯度裁剪 | 损失计算
    """
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        # --- 数据转移 ---
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        
        # --- 动态学习率（余弦退火）---
        current_step = epoch * total_steps + step
        lr = get_lr(current_step, args.epochs * total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # --- 混合精度前向 + 损失计算 ---
        with autocast_ctx:
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = (outputs.loss + outputs.aux_loss) / args.accumulation_steps  # 梯度累积缩放
        
        # --- 梯度累积 + 更新 ---
        scaler.scale(loss).backward()
        
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 还原真实梯度（梯度裁剪前必需！）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 防梯度爆炸
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # --- 日志记录（每N步）---
        if step % args.log_interval == 0 or step == total_steps:
            elapsed = time.time() - start_time
            real_loss = loss.item() * args.accumulation_steps
            eta_min = int(elapsed / step * total_steps // 60 - elapsed // 60)
            
            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}]({step}/{total_steps}) "
                f"loss:{real_loss:.6f} lr:{lr:.2e} ETA:{eta_min}min"
            )
            if wandb and is_main_process():
                wandb.log({"loss": real_loss, "lr": lr, "eta_min": eta_min})
        
        # --- 模型保存（主进程）---
        if (step % args.save_interval == 0 or step == total_steps) and is_main_process():
            _save_checkpoint(epoch, step, wandb)


# =============== 🔥 深入掌握：检查点保存逻辑 ===============
def _save_checkpoint(epoch, step, wandb=None):
    """分布式安全保存（高频考点）"""
    model.eval()
    
    # 构建保存路径（MoE标识）
    suffix = "_moe" if getattr(lm_config, "use_moe", False) else ""
    ckp_path = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{suffix}.pth"
    
    # 提取纯净模型状态（DDP需.module）
    state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
    state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度节省空间
    torch.save(state_dict, ckp_path)
    
    # 保存完整训练状态（用于断点续训）
    lm_checkpoint(
        lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
        scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir="../checkpoints"
    )
    model.train()


# =============== 🔥 深入掌握：主训练流程 ===============
if __name__ == "__main__":
    # =============== 🌱 了解即可：参数解析 ===============
    parser = argparse.ArgumentParser(description="AquilaLM Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0,1])
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0,1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AquilaLM-Pretrain")
    args = parser.parse_args()
    
    # =============== 🔥 深入掌握：分布式初始化 ===============
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    
    # 随机种子（不同进程不同种子，保证采样多样性）
    seed = 42 + (dist.get_rank() if dist.is_initialized() else 0)
    setup_seed(seed)
    
    # =============== 💡 建议了解：环境配置 ===============
    os.makedirs(args.save_dir, exist_ok=True)
    
    lm_config = Self_MinimindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    
    # 混合精度上下文（CPU不支持autocast）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # =============== 💡 建议了解：实验跟踪（WandB/SwanLab）==============
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 国产替代
        ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume else None
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(project=args.wandb_project, name=f"AquilaLM-E{args.epochs}-B{args.batch_size}", id=wandb_id, resume="must" if wandb_id else None)
    
    # =============== 🔥 深入掌握：模型/数据/优化器初始化 ===============
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # =============== 🔥 深入掌握：断点续训恢复 ===============
    start_epoch, start_step = 0, 0
    if args.from_resume:
        ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if ckp_data:
            model.load_state_dict(ckp_data["model"])
            optimizer.load_state_dict(ckp_data["optimizer"])
            scaler.load_state_dict(ckp_data["scaler"])
            start_epoch = ckp_data["epoch"]
            start_step = ckp_data.get("step", 0)
            Logger(f".Resume training from epoch {start_epoch}, step {start_step}")
    
    # =============== 🔥 深入掌握：DDP封装（关键！）==============
    if dist.is_initialized():
        # RoPE缓存为确定性计算结果，无需同步（节省带宽）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # =============== 🔥 深入掌握：训练主循环 ===============
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)  # 每epoch重置采样器种子
        
        # 断点续训：跳过已训练step
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=1, pin_memory=True)
            Logger(f"Epoch {epoch+1}: Skipping first {start_step} steps")
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                sampler=train_sampler, num_workers=1, pin_memory=True
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    Logger("✅ Pretraining completed!")