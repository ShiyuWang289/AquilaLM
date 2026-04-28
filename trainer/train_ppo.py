# ==========导入部分==========
import os
import sys

# 📚 Python模块系统
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse  # 命令行参数解析
import re  # 正则表达式，用于奖励计算
import warnings  # 警告控制
import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练支持
import torch.nn.functional as F  # 神经网络函数
from transformers import AutoTokenizer  # HuggingFace分词器
from contextlib import nullcontext  # 上下文管理器
from torch import optim, nn  # 优化器和神经网络
from torch.nn.parallel import DistributedDataParallel  # 分布式并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载
from torch.nn.utils import clip_grad_norm_  # 梯度裁剪
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度
from transformers import AutoModel  # HuggingFace模型加载
from model.model import Self_Minimindconfig, Self_MinimindForCausalLM
from dataset.lm_dataset import RLAIFDataset  # RL数据集
from trainer.trainer_utils import (  # 训练工具函数
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
)
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════════
#                           1. Critic Model — 价值网络
#   Actor  → 输出动作概率分布 (即 LLM 的 next-token logits)
#   Critic → 输出标量状态价值 V(s)，用于计算优势 A = R - V(s)
# 结构：复用 LLM backbone，将最后的 lm_head(H→V词表) 替换为 value_head(H→1)
# ════════════════════════════════════════════════════════════════════════════════
class CriticModel(Self_MinimindForCausalLM):
    def __init__(self,params):
        super().__init__(params)
        self.value_head=nn.Linear(params.hidden_size,1) # 增加一个线性层作为价值头，用于输出状态价值

    def forward(self,input_ids,attention_mask=None,**kwargs):
        outputs=self.model(input_ids,attention_mask=attention_mask,**kwargs) # 获取语言模型的输出
        hidden_states=self.model.norm(outputs[0]) # 对最后一层的隐藏状态进行归一化处理
        values=self.value_head(hidden_states).squeeze(-1) # 通过价值头输出状态价值，并去掉最后一个维度
        return values
# ════════════════════════════════════════════════════════════════════════════════
#                           2. 奖励计算
# 奖励由两部分组成（reasoning 模式下）：
#   ① 格式奖励：正则匹配 <think>...</think><answer>...</answer> 结构
#   ② 内容奖励：Reward Model 打分，clamp 到 [-3, 3] 防止极端值
# ════════════════════════════════════════════════════════════════════════════════
def calculate_rewards(prompts,responses,reward_model,reward_tokenizer):
    """计算奖励函数，结合格式奖励和内容奖励"""
    # ── 格式奖励函数（仅 reasoning 模式）──────────────────────────────
    def resoning_format_reward(rewards):
        # 检查 response 是否符合 <tool_call>...<answer>... 格式，符合则加分
        pat1 = r"^<tool_call>\n.*?\n<tool_call>\n<answer>\n.*?\n</answer>$"
        pat2 = r"^<tool_call>\n.*?\n<tool_call>\n\n<answer>\n.*?\n</answer>$"  # 兼容多一个空行的情况
        format_scores=[
            0.5 if (re.match(pat1,r,re.S) or re.match(pat2,r,re.S)) else 0.0
            for r in responses
        ]       
        rewards+=torch.tensor(format_scores,device=args.device)
        # 标签计数奖励：每个正确出现恰好 1 次的标签 +0.25
        tag_scores = []
        for r in responses:
            s = sum(0.25 for tag in ["<think>", "</think>", "<answer>", "</answer>"]
                    if r.count(tag) == 1)
            tag_scores.append(s)
        rewards += torch.tensor(tag_scores, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1: # reasoning 模式下计算格式奖励
        rewards = resoning_format_reward(rewards)

    # ── Reward Model 评分 ─────────────────────────────────────────
    with torch.no_grad():
        rm_scores=[]
        for prompt,response in zip(prompts,responses):
            # 从 chat-template 格式的 prompt 中解析出 messages 列表
            pattrn = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches= re.findall(pattrn, prompt, re.DOTALL)
            messages= [{"role": role, "content": content.strip()} for role, content in matches]
            chat = messages + [{"role": "assistant", "content": response}]  # 将 response 作为 assistant 的回复添加到消息列表中
            score =  reward_model.get_score(reward_tokenizer, chat)  # 调用 Reward Model 获取评分
            scale = 3.0  # 奖励缩放因子
            score = max(min(score,scale),-scale)  # 将评分 clamp 到 [-3, 3] 范围内

            if args.reasoning == 1:  # reasoning 模式下:单独给 <answer> 内容打分，加权组合
                answer_macth=re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if answer_macth:
                    answer_chat = messages + [
                        {"role": "assistant", "content": answer_macth.group(1).strip()}
                    ]
                    ans_score = reward_model.get_score(reward_tokenizer, answer_chat)
                    ans_score = max(min(ans_score, scale), -scale)
                    score = score * 0.4 + ans_score * 0.6  # 综合评分，内容奖励占比更大
            rm_scores.append(score) 
        rewards += torch.tensor(rm_scores, device=args.device)
    return rewards
# ════════════════════════════════════════════════════════════════════════════════
# 3. 序列 log 概率计算 — 提取为辅助函数
# ────────────────────────────────────────────────────────────────────────────────
# 【语法难点】核心操作链：log_softmax → gather → mask → sum
#
#   logits[:, :-1]  对齐: logits[i] 预测 token[i+1]，即 labels = tokens[:, 1:]
#   .gather(2, idx) 沿词表维取出目标 token 的 log 概率
#   * mask           只保留 response 部分（排除 prompt 和 pad）
#   .sum(dim=1)      汇总为序列级 log π(response | prompt)
# ════════════════════════════════════════════════════════════════════════════════

def compute_seq_logp(model, gen_out, labels, full_mask, final_mask):
    """计算模型对生成序列中 response 部分的总 log 概率 → [B]"""
    logits = model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, L, V]
    # gather: 从 [B, L-1, V] 中按 labels 索引取出每个位置的 log P → [B, L-1]
    logp = (F.log_softmax(logits[:, :-1], dim=-1)
            .gather(2, labels.unsqueeze(-1))
            .squeeze(-1))
    return (logp * final_mask).sum(dim=1)  # [B]


# ════════════════════════════════════════════════════════════════════════════════
# 4. PPO 单 Epoch 训练
# ────────────────────────────────────────────────────────────────────────────────
# 每步流程：
#   Rollout → Reward → Advantage → Policy Loss(clip) → Value Loss → KL → 更新
#
# 【理论重点】PPO-Clip 损失：
#   ratio = π_θ(a|s) / π_θk(a|s)
#   L = -min(ratio·A, clip(ratio, 1±ε)·A)
#   → 限制策略更新幅度在 [1-ε, 1+ε] 的信赖域内
#
# 【理论重点】总损失 = policy_loss + vf_coef·value_loss + kl_coef·kl_ref
#   kl_ref: 与冻结的 ref_model 的 KL 散度，防止策略偏离 SFT 模型过远
# ════════════════════════════════════════════════════════════════════════════════

def ppo_train_epoch(
    epoch, loader, iters, old_actor_model, ref_model,
    actor_scheduler, critic_scheduler, reward_model, reward_tokenizer,
    start_step=0, wandb=None,
):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        # ── Step 1: Tokenize（RL 数据集返回字符串，在线编码）──────
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_seq_len,
        ).to(args.device)
        prompt_lengths = enc.attention_mask.sum(dim=1)  # 每条 prompt 的实际 token 数

        # ── Step 2: Actor Rollout（在线采样生成回复）────────────
        with torch.no_grad():
            # 【语法难点】DDP 包装后需 .module 解包才能调 .generate()
            raw_model = (actor_model.module if isinstance(actor_model, DistributedDataParallel)
                         else actor_model)
            gen_out = raw_model.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            )

        # 解码 response 文本（跳过 prompt 部分）
        responses_text = [
            tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]

        # ── Step 3: 计算奖励 ────────────────────────────────────
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)

        # ── Step 4: Critic 价值估计 → 优势函数 ──────────────────
        full_mask = (gen_out != tokenizer.pad_token_id).long()
        value_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, L]

        # 【语法难点】高级索引：取每条序列最后一个有效 token 的 value
        #   arange([0,1,...,B-1]) 索引行，last_indices 索引列 → [B]
        last_indices = full_mask.sum(dim=1) - 1
        values = value_seq[torch.arange(len(last_indices)), last_indices]

        # 【理论重点】优势 A = R - V(s)，detach 防止梯度流入 Critic
        advantages = rewards - values.detach()

        # ── Step 5: 构建 response mask ──────────────────────────
        labels = gen_out[:, 1:].clone()  # shift-by-1: logits[i] 预测 token[i+1]
        seq_len = gen_out.size(1) - 1

        # 【语法难点】广播构建 mask：[1, L-1] >= [B, 1] → [B, L-1]
        resp_mask = (torch.arange(seq_len, device=gen_out.device).unsqueeze(0)
                     >= prompt_lengths.unsqueeze(1))
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))

        # ── Step 6: 计算三个模型的序列 log 概率 ─────────────────
        actor_logp = compute_seq_logp(actor_model, gen_out, labels, full_mask, final_mask)

        with torch.no_grad():
            old_logp = compute_seq_logp(old_actor_model, gen_out, labels, full_mask, final_mask)
            ref_logp = compute_seq_logp(ref_model, gen_out, labels, full_mask, final_mask)

        # ── Step 7: PPO-Clip 损失 ──────────────────────────────
        # 【理论重点】ratio = exp(log π_θ - log π_θk) 即重要性采样比
        #   用 exp(log差) 代替 概率相除，数值更稳定
        ratio = torch.exp(actor_logp - old_logp)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, rewards)
        kl_ref = (actor_logp - ref_logp).mean()

        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref
        loss.backward()

        # ── Step 8: 梯度裁剪 + 参数更新 ────────────────────────
        if step % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ── 日志 ────────────────────────────────────────────────
        if is_main_process() and (step % args.log_interval == 0 or step == iters):
            kl = (actor_logp - old_logp).mean()
            # 计算平均 response 长度（找第一个 EOS 位置）
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = response_ids == tokenizer.eos_token_id
            lengths = torch.where(
                is_eos.any(dim=1),
                torch.argmax(is_eos.int(), dim=1) + 1,
                torch.tensor(response_ids.shape[1], device=is_eos.device),
            )
            avg_len = lengths.float().mean().item()

            log_dict = dict(
                actor_loss=policy_loss.item(), critic_loss=value_loss.item(),
                reward=rewards.mean().item(), kl=kl.item(), kl_ref=kl_ref.item(),
                avg_response_len=avg_len, actor_lr=actor_optimizer.param_groups[0]["lr"],
            )
            if wandb is not None:
                wandb.log(log_dict)

            Logger(
                f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                f"Actor Loss: {log_dict['actor_loss']:.6f}, Critic Loss: {log_dict['critic_loss']:.6f}, "
                f"Reward: {log_dict['reward']:.6f}, KL: {log_dict['kl']:.6f}, "
                f"KL_ref: {log_dict['kl_ref']:.6f}, Avg Len: {avg_len:.2f}, "
                f"LR: {log_dict['actor_lr']:.2e}"
            )

        # ── 定期同步 Old Actor ──────────────────────────────────
        # 【理论重点】Old Actor 是 PPO 重要性采样的基准策略 π_θk
        #   定期从 Actor 复制参数，过于频繁≈on-policy（低效），过于稀疏→ratio偏离1
        if step % args.update_old_actor_freq == 0:
            src = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            # detach+cpu 避免显存翻倍，再搬回 GPU
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in src.state_dict().items()})
            old_actor_model.to(args.device)

        # ── 模型保存 ────────────────────────────────────────────
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp_path = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            src = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            torch.save({k: v.half() for k, v in src.state_dict().items()}, ckp_path)

            lm_checkpoint(
                lm_config, weight=args.save_weight, model=actor_model,
                optimizer=actor_optimizer, epoch=epoch, step=step, wandb=wandb,
                save_dir="../checkpoints", scheduler=actor_scheduler,
                critic_model=critic_model, critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
            )
            actor_model.train()


# ════════════════════════════════════════════════════════════════════════════════
# 5. 主函数入口
# ────────────────────────────────────────────────────────────────────────────────
# PPO 训练需要 5 个模型：
#   Actor(训练) / Old Actor(冻结,定期同步) / Critic(训练)
#   Reference(冻结,始终不变) / Reward Model(外部,冻结)
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 参数定义 ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="AquilaLM PPO")

    # 基础训练
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="ppo_actor", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    # PPO 学习率极小(~8e-8)，远低于 SFT(~1e-5)，防止策略剧变
    parser.add_argument("--learning_rate", type=float, default=8e-8)
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8)

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
    parser.add_argument("--max_gen_len", type=int, default=1536, help="Response 最大长度")

    # 数据
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl")

    # PPO 超参数
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="裁剪范围 [1-ε, 1+ε]")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value loss 权重")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL 惩罚权重")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1])
    parser.add_argument("--update_old_actor_freq", type=int, default=4)
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])

    # 实验跟踪
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AquilaLM-PPO")

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
        import swanlab as wandb # type: ignore
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.wandb_project,
            name=f"PPO-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}",
            id=wandb_id, resume="must" if wandb_id else None,
        )

    # ── 5. 初始化 5 个模型 ────────────────────────────────────
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    # Actor（策略模型，正在训练）
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # 【语法难点】PPO 需要左侧 padding：generate() 从最右侧续写，
    #   左 padding 保证有效 token 紧靠右侧，续写位置正确
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Old Actor（冻结，定期从 Actor 同步）
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)

    # Reference（冻结，始终不变，用于 KL 惩罚基准）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # Critic（价值网络，正在训练，从 SFT 权重初始化 backbone）
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(torch.load(ckp, map_location=args.device), strict=False)
    critic_model = critic_model.to(args.device)

    # Reward Model（外部模型，冻结）
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # ── 6. 数据 & 优化器 & 调度器 ────────────────────────────
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)

    iters = len(DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler))
    total_steps = max(1, (iters // args.accumulation_steps) * args.epochs)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_steps, eta_min=args.critic_learning_rate / 10)

    # ── 7. 从 Checkpoint 恢复 ────────────────────────────────
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data["model"])
        critic_model.load_state_dict(ckp_data["critic_model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        critic_optimizer.load_state_dict(ckp_data["critic_optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        critic_scheduler.load_state_dict(ckp_data["critic_scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ── 8. DDP 包装 ──────────────────────────────────────────
    if dist.is_initialized():
        for m in (actor_model, critic_model):
            m._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)

    # ── 9. 训练循环 ──────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f"Epoch [{epoch+1}/{args.epochs}]: 从 step {start_step+1} 续训")
            ppo_train_epoch(epoch, loader, len(loader) + start_step, old_actor_model, ref_model,
                            actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model,
                            actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)