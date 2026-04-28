from transformers import PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple , List , Union
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PreTrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


# huggingface的类，提供模型初始参数
class Self_Minimindconfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim)) #可学习参数γ，初始化为全1
        # 核心公式：y = x * (γ / sqrt(mean(x^2) + eps))
        def _norm(self,x):
            return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)  #mean只对最后一个维度求平均
        def forward(self,x):
            return self.weight * self._norm(x.float()).type_as(x)  #输入x的形状为[batch_size, seq_len, dim]，输出与输入形状相同
        
def precompute_freqs(
        dim:int,
        end:int = 32 *1024,
        rope_base:float = 1e6,
        rope_scaling: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算RoPE频率, 理论核心：θ_i = base^(-2i/d) → 旋转角 = position × θ_i"""
    # ===1. 标准RoPE频率θ_i===
    indices = torch.arange(0, dim, 2, dtype=torch.float32)[: dim//2] # indices = [0, 2, 4, ..., dim-2] → 每2维一组（共 dim//2 组）
    freqs = rope_base ** (-indices / dim)  # 公式：θ_i = base^(-2i/d)  （i 为组索引）
    attn_factor = 1.0  # 注意力温度补偿系数（YaRN 中 >1）
    # ===2. YaRN RoPE频率缩放===
    if rope_scaling and (end>rope_scaling.get("original_max_position_embeddings", 2048)):
        orig_max = rope_scaling["original_max_position_embeddings"]  # L_train
        factor = rope_scaling["factor"]        # s = L_target / L_train
        beta_fast = rope_scaling["beta_fast"]  # 高频边界（b > β_fast → 不缩放）
        beta_slow = rope_scaling["beta_slow"]  # 低频边界（b < β_slow → 全缩放 1/s）
        attn_factor = rope_scaling.get("attention_factor", 1.0)  # 温度补偿 γ_attn

        def wavelength_to_idx(b:float)->float:
            """根据位置b计算对应的频率索引i（非整数），
            b = L_train / λ  （归一化波长倒数，b 越大 → 频率越高） 
            λ = 2π · base^(2i/d) → i = (d/2) · log_base(L_train / (2πb))
            核心公式：i = - (d / 2) * log_base(θ_i) = (d / 2) * log_base(orig_max / b)"""
            return(dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        
        # 计算高频区终点（low）与低频区起点（high）
        low = max(math.floor(wavelength_to_idx(beta_fast)), 0)   # b > β_fast → 高频 → γ=0
        high = min(math.ceil(wavelength_to_idx(beta_slow)), dim // 2 - 1)  # b < β_slow → 低频 → γ=1
        # 计算ramp：线性过渡系数 γ 
        pos = torch.arange(dim // 2, dtype=torch.float32)
        ramp = torch.clamp((pos - low) / max(high - low, 1e-5), 0, 1)
        # freqs缩放公式：θ_i' = θ_i · [(1-γ) + γ/s]  
        freqs = freqs * (1 - ramp + ramp / factor)
    
    # ===3.生成旋转总角度 position × freqs ===
    positions = torch.arange(end, dtype=torch.float32)  # [0, 1, ..., end-1]
    angles = torch.outer(positions, freqs)  # [end, dim//2]：每个位置×每组频率

    # === 4. 扩展至完整维度（每组频率重复2次，匹配Q/K的[x0,x1]结构）===
    # 注意：乘 attn_factor 补偿缩放导致的注意力发散
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attn_factor
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attn_factor
    return cos, sin

def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    🌀 应用旋转：Q_rot = Q·cos - Q_half·sin （复数乘法等效）
    理论：将 (x0, x1) 旋转 θ → (x0·cosθ - x1·sinθ, x0·sinθ + x1·cosθ)
    """
    # 旋转后半部分：[-x1, x0] → 实现 (x0, x1) → (-x1, x0)（复数乘 i）
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # 对齐维度（如 [bs, seq, head, dim] → cos/sin 在 seq 维匹配）
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    
    # 核心旋转：Q_rot = Q * cos + rotate_half(Q) * sin
    # 数学等价：[x0, x1] ⊗ [cosθ, sinθ] = [x0·cosθ - x1·sinθ, x0·sinθ + x1·cosθ]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    """将KV头沿头维度复制n_rep次，使KV头数 = Q头数
     例: [B, L, 8, D] + n_rep=4 → [B, L, 32, D]（每组KV被4个Q头共享）"""
    if n_rep == 1:
        return x
    return x.unsqueeze(2).expand(-1,-1,n_rep,-1,-1).reshape(x.shape[0],x.shape[1],-1,x.shape[3]) #x.shape[3]指head_dim，因为x的形状是[B, L, 8, D]，此时unsqueeze和expand还没有应用

class Attention(nn.Module):
    """"🌈 GQA (Grouped-Query Attention) 模块核心思想：多个Query头共享同一组Key/Value头"""
    def __init__(self,args):
        super().__init__()

        # === GQA核心配置 ===
        # 若未指定num_key_value_heads → 退化为MHA（KV头=Q头）
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        # 安全校验：Q头数必须能被KV头数整除（确保分组均匀）
        assert args.num_attention_heads % self.num_key_value_heads == 0, \
            "❌ num_attention_heads 必须能被 num_key_value_heads 整除，以确保每组KV头对应整数个Q头。"
        # === 关键维度参数 ===
        self.n_local_heads=args.num_attention_heads     # Q头数（总头数）
        self.n_local_kv_heads=self.num_key_value_heads     # KV头数
        self.n_rep=self.n_local_heads//self.n_local_kv_heads     # 每组共享的Q头数
        self.head_dim=args.hidden_size//args.num_attention_heads     # 每头维度=总维度/头数
        # === 投影层（GQA关键：KV投影维度更小！）===
        # Q投影：输出 = Q头数 × 每头维度（例：32×128=4096）
        self.q_proj=nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=False
        )
        # K/V投影：输出 = KV头数 × 每头维度（例：8×128=1024）
        self.k_proj=nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.v_proj=nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        # 输出投影：将多头输出拼接后映射回隐藏维度
        self.o_proj=nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False
        )
        # === 正则化与优化 ===
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout
        # 检查是否启用Flash Attention（PyTorch 2.0+）
        self.flash=(
            hasattr(F, "scaled_dot_product_attention") 
            and args.flash_attention
        )
        if not self.flash and args.flash_attention:
            print("⚠️ Flash Attention 不可用，已自动降级为普通注意力计算。请确保使用 PyTorch 2.0+ 并启用相关配置。")

    def forward(
            self,
            x:torch.Tensor, # 输入张量 [batch_size, seq_len, hidden_size]
            position_embeddings:Tuple[torch.Tensor,torch.Tensor], # 预计算的RoPE频率 (cos, sin)
            past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None, # 过去的KV缓存（推理时使用）
            use_cache:bool=False, # 是否返回新的KV缓存（推理时使用）
            attention_mask:Optional[torch.Tensor]=None,
    )->Tuple[torch.Tensor,Optional[Tuple[torch.Tensor,torch.Tensor]]]:
        """前向流程：1. 投影 → 2. 分头 → 3. 位置编码 → 4. KV缓存拼接 → 5. GQA扩展KV → 6. Attention计算 → 7. 合并输出"""
        batch_size,seq_len=x.shape[0],x.shape[1]
        # === 1. 投影 ===
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        # === 2. 分头 ===
        xq=xq.view(batch_size,seq_len,self.n_local_heads,self.head_dim)
        xk=xk.view(batch_size,seq_len,self.n_local_kv_heads,self.head_dim)
        xv=xv.view(batch_size,seq_len,self.n_local_kv_heads,self.head_dim)
        # === 3. 位置编码（RoPE）===
        cos,sin=position_embeddings
        xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin)
        # === 4. KV缓存拼接（仅推理时）===
        if past_key_value is not None:
            # 拼接历史kv与当前kv（维度：seq_len增加），避免重复计算历史部分
            # past_key_value = (历史K, 历史V) [batch_size, seq_len_past, kv_heads, head_dim]
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        past_key_value=(xk,xv) if use_cache else None
        # === 5.GQA核心 - 扩展KV头（使KV头数=Q头数）===
        xq=xq.transpose(1,2)  # [batch_size, Q_heads, seq_len, head_dim]
        xk=repeat_kv(xk,self.n_rep).transpose(1,2)
        xv=repeat_kv(xv,self.n_rep).transpose(1,2)
        # === 6.Attention计算（双路径）===
        if(
            self.flash
            and seq_len>1
            and past_key_value is None # 无KV缓存（训练场景）
            and (attention_mask is None or torch.all(attention_mask==1)) # 无注意力掩码或全1掩码（非填充场景）  
        ):
            # 使用Flash Attention（仅训练且无掩码时）
            output=F.scaled_dot_product_attention(
                xq,xk,xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True # 启用因果掩码（禁止看未来）
            )
            output = output.transpose(1,2).reshape(batch_size,seq_len,-1) # [batch_size, seq_len, Q_heads*head_dim]
            output = self.o_proj(output) # [batch_size, seq_len, hidden_size]
            output = self.resid_dropout(output)
            return output, past_key_value
        else:
            # 标准Attention实现（兼容所有场景）
            # （1）计算注意力分数：scores =  QK^T / √d
            scores=torch.matmul(xq,xk.transpose(-2,-1)) / math.sqrt(self.head_dim) #scores形状：[batch_size, Q_heads, seq_len, seq_len_kv]
            # （2）应用注意力掩码（如有）
            if seq_len>1:
                causal_mask=torch.triu(
                    torch.full((seq_len,seq_len),float("-inf"),device=scores.device),
                    diagonal=1
                )
                # 修复：因果掩码只应作用于当前键（最后seq_len个位置），避免掩码历史键
                scores[:,:,-seq_len:,-seq_len:]+=causal_mask.unsqueeze(0).unsqueeze(0) # 仅对当前查询和当前键的交互应用因果掩码
            # (3) 应用额外的注意力掩码（如填充掩码）
            if attention_mask is not None:
                extend_mask=attention_mask.unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, seq_len_kv]
                scores=scores+(1.0-extend_mask)*-1e9 # 将填充位置的分数设置为极小值，确保softmax后权重接近0
            #(4) softmax归一化+Dropout
            scores=F.softmax(scores.float(),dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            #(5) 加权聚合V
            output = torch.matmul(scores,xv) # 输出形状：[batch_size, Q_heads, seq_len, head_dim]
            # === 步骤7: 合并多头输出 ===
            output=output.transpose(1,2).reshape(batch_size,seq_len,-1) # [batch_size, seq_len, Q_heads*head_dim]
            # === 步骤8: 输出投影 + 残差Dropout ===
            output=self.o_proj(output) # [batch_size, seq_len, hidden_size]
            output=self.resid_dropout(output)

            return output,past_key_value

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        # === 中间层维度计算（硬件友好设计）===
        if config.intermediate_size is None:
            intermediate_size=int(config.hidden_size *8/3)  #经验公式：*8/3
            config.intermediate_size=64 * ((intermediate_size+64-1) //64)  #调整为64倍数（GPU内存对齐优化，减少碎片）
        # === 三投影层（门控FFN核心）===
        self.gate_proj=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)  #门控路径
        self.up_proj=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)    #特征路径
        self.down_proj=nn.Linear(config.intermediate_size,config.hidden_size,bias=False)  #降维输出
        self.dropout=nn.Dropout(config.dropout)
        self.act_fn=ACT2FN[config.hidden_act] #通常为SiLU（SwiGLU核心）
    def forward(self,x):
        # 核心公式：output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
        gated=self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))

class MoEGate(nn.Module):
    """
    ✨ MoE门控核心：智能分诊 + 负载均衡
    功能：为每个token选择top-k专家 + 计算负载均衡损失
    """
    def __init__(self,config:Self_Minimindconfig):
        super().__init__()
        self.topk=config.num_experts_per_tok
        self.n_experts=config.n_routed_experts
        self.alpha=config.aux_loss_alpha  #aux_loss 权重
        self.norm_topk=config.norm_topk_prob  #是否对top-k概率归一化
        self.seq_aux=config.seq_aux  #负载均衡计算方式
        # 门控权重：[专家数, 隐藏维度] → 将token映射到专家得分
        self.weght=nn.Parameter(torch.empty(self.n_experts,config.hidden_size))
        nn.init.xavier_uniform_(self.weght,a=math.sqrt(5))  #权重初始化

    def forward(self,x:torch.Tensor):
        """输入x形状：[B, L, D] → 输出：top-k专家索引、对应权重、辅助损失"""
        bsz,seq_len,_=x.shape
        x_flat=x.view(-1,x.shape[-1]) # [B*L, D]：将批次和序列维度合并，方便计算每个token的专家得分
        # === 1. 计算专家得分 ===
        logits=F.linear(x_flat,self.weight) #[N,n_experts]
        scores=logits.softmax(dim=-1)  #softmax归一化为概率分布
        # === 2. 选择top-k专家 ===
        topk_weight,topk_idx=torch.topk(scores,self.top_k,dim=-1) #[N,K]
        # === 3. 权重归一化（仅topk内归一）===
        if self.top_k>1 and self.norm_topk:
            topk_weight=topk_weight/topk_weight.sum(dim=-1,keepdim=True)  #仅在top-k专家内归一化，保持概率分布特性
        # === 4. 计算负载均衡辅助损失 ===
        if self.training and self.alpha>0:
            if self.seq_aux:
                # 序列级均衡：每序列内专家使用频率均衡
                ce = torch.zeros(bsz, self.n_experts, device=x.device)
                # 统计每个序列内每个专家被选中的次数：把topk_idx展平后，使用scatter_add_在对应专家索引位置累加1，得到每个序列内每个专家的使用频率
                ce.scatter_add_(1, topk_idx.view(bsz, -1), 
                               torch.ones(bsz, topk_idx.numel()//bsz, device=x.device))
                ce = ce / (seq_len * self.top_k / self.n_experts)  # 归一化
                aux_loss = (ce * scores.view(bsz, seq_len, -1).mean(1)).sum(1).mean() * self.alpha
            else:
                # 全局均衡（更常用）：整个batch专家使用频率均衡
                mask = F.one_hot(topk_idx.view(-1), self.n_experts).float()  # [N*k, E]
                usage = mask.mean(0)          # 每个专家被选中的频率 [E]
                importance = scores.mean(0)   # 每个专家的平均重要性 [E]
                aux_loss = (importance * usage * self.n_experts).sum() * self.alpha
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
        
        return topk_idx, topk_weight, aux_loss




class MoEFeedForward(nn.Module):
    """  架构组成：
    ├─ 路由专家池：n_routed_experts 个专业化FFN（例：动物/家具/天气专家）
    ├─ 门控路由层：MoEGate（智能分诊秘书）
    └─ 共享专家池：n_shared_experts 个基础语义专家（保底机制）
    """
    def __init__(self,config:Self_Minimindconfig):
        super().__init__()
        self.config=config
        self.num_experts=config.n_routed_experts
        self.topk=config.num_experts_per_tok # 每token激活专家数（通常=2）
        # 定义专家池（每个专家都是一个独立的FeedForward网络）
        self.experts=nn.ModuleList([FeedForward(config) for _ in range(self.num_experts)])
        # 定义门控路由层（输入：token特征，输出：专家选择概率）
        self.gate = MoEGate(config)
        # 定义共享专家池（所有token共享，提供保底能力）
        self.shared_experts=nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """路由 → 专家处理 → 加权融合 → 共享增强
            输入: [B, L, D] | 输出: [B, L, D]"""
        batch_size,seq_len,hidden_dim=x.shape
        total_tokens=batch_size*seq_len
        # === 1. 门控路由：计算每个token的专家选择概率 ===
        topk_idx,topk_weight,aux_loss=self.gate(x) #gate函数作用：输入x，输出每个token的得分最高的top-k专家索引、对应得分权重和辅助损失
        self.aux_loss=aux_loss  # 存储辅助损失，训练时优化路由质量
        # === 2. 展平处理：统一token维度 ===
        x_flat=x.view(total_tokens,hidden_dim)
        # === 3. 初始化输出（全零累加容器）===
        y_flat=torch.zeros_like(x_flat)
        # === 4. 核心：遍历每个专家，处理分配给它的token ===
        for expert_idx in range(self.num_experts):
            expert_mask=(topk_idx==expert_idx)
            if not expert_mask.any():
                continue  # 跳过未分配到任何token的专家
            token_positions=torch.nonzero(expert_mask,as_tuple=True)[0] # 找到分配给当前专家的token位置索引
            weights=topk_weight[expert_mask]
            expert_input=x_flat[token_positions]
            expert_output=self.experts[expert_idx](expert_input)
            y_flat.index_add_(0,token_positions,expert_output*weights.unsqueeze(1))  # 加权累加专家输出
        # === 5. 加入共享专家输出（所有token都经过共享专家处理）===
        if self.shared_expers is not None:
            for expert in self.shared_experts:
                y_flat+=expert(x).view(total_tokens,hidden_dim)
        # === 6. 恢复原始形状 ===
        return y_flat.view(batch_size,seq_len,hidden_dim)



class Self_MinimindBlock(nn.Module):
    """Transformer块：包含GQA注意力和门控FFN，前者负责信息交互，后者负责非线性变换和特征重组"""
    def __init__(self,layer_id:int,config:Self_Minimindconfig):
        super().__init__()
        # === 基础维度配置（供调试/日志使用）===
        self.num_attention_heads=config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.head_dim=self.hidden_size//self.num_attention_heads
        # === 模块定义 ===
        self.self_attention=Attention(config)
        self.layer_id=layer_id
        self.before_attention_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.before_FFN_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp=(
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)
        )

    def forward(
            self,
            hidden_states, # [batch_size, seq_len, hidden_size]：上一层输出/词嵌入
            position_embeddings:Tuple[torch.Tensor,torch.Tensor], # 预计算的RoPE频率 (cos, sin)
            past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None, # KV缓存（推理时使用）
            use_cache=False, # 是否返回新的KV缓存（推理时使用）
            attention_mask:Optional[torch.Tensor]=None, # 注意力掩码（如填充掩码）
    ):
        """输入 → [RMSNorm → Attention → +残差] → [RMSNorm → FFN/MoE → +残差] → 输出"""
        # === 步骤1：保存原始输入（用于残差连接）===
        res=hidden_states
        # === 步骤2：前置RMSNorm + Attention ===
        hidden_states,present_key_value=self.self_attention(
            self.before_attention_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        # === 步骤3：第一次残差连接（Attention路径）===
        hidden_states=res+hidden_states
        # === 步骤4：前置RMSNorm + FFN/MoE + 第二次残差连接 ===
        res=hidden_states
        hidden_states=hidden_states+self.mlp(
            self.before_FFN_layernorm(hidden_states)
        )
        # === 步骤5：返回输出 + KV缓存 ===
        return hidden_states,present_key_value  # 推理时传递给下一层/下一步，避免重复计算历史KV

class Self_MinimindModel(nn.Module):
    """    Token IDs → Embedding → [Block₀ → Block₁ → ... → Blockₙ] → RMSNorm → Logits
                                      ↑          ↑               ↑
                                 (可选KV缓存) (MoE辅助损失收集) (位置编码)
    """   
    def __init__(self,config:Self_Minimindconfig):
        super().__init__()
        self.config=config
        # === 1. 词嵌入层：离散ID → 连续向量空间 ===
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)
        # === 2. Transformer块(Block)堆叠 ===
        # 每层 = Attention + FFN/MoE + 双RMSNorm + 双残差连接
        self.layers=nn.ModuleList([
            self.MinimindBlock(layer_id=i,config=config)
            for i in range(config.num_hidden_layers)
        ])
        # === 3. 输出层前的RMSNorm ===
        self.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        # === 4. RoPE 位置编码（预计算 + 缓存）===
        freqs_cos,freqs_sin=precompute_freqs(
            dim=config.hidden_size//config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,  # 缩放策略
        )
        # 注册为buffer，随模型保存/加载，但不参与梯度更新
        # rope的计算结果是固定的cos表和sin表，训练时不更新，保存模型时直接扔掉（能秒重算），推理时现场生成
        self.register_buffer("freqs_cos",freqs_cos,persistent=False)
        self.register_buffer("freqs_sin",freqs_sin,persistent=False)

    def forward(
            self,
            input_ids:torch.Tensor, # [batch_size, seq_len]：输入token ID序列
            attention_mask:Optional[torch.Tensor]=None,
            past_key_values:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,# 每层的KV缓存列表（推理时使用）
            use_cache:bool=False,
            **kwargs, # 其他参数（如labels，训练时使用）
    )->Tuple[torch.Tensor,List,torch.Tensor]:
        """前向流程：1. 输入嵌入 → 2. Transformer块堆叠 → 3. RMSNorm → 4. 输出Logits
            输入：token IDs + 可选注意力掩码 + 可选KV缓存
            输出：hidden_states + 每层KV缓存 + MoE辅助损失（如适用）"""
        batch_size,seq_length=input_ids.shape
        # === 步骤1: 兼容性处理：===
        if hasattr(past_key_values, "layers"): # 针对“不标准”的输入结构（含有layers属性），格式归零
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers) #对归零过的past_key_values进行格式化处理
        # === 步骤2: 计算位置偏移（推理关键！）===
        # 此时 past_key_values[0] = (key_tensor, value_tensor)
        # key_tensor.shape = [batch=1, seq_len=1, heads=32, head_dim=128]
        #                                  ↑ shape[1]维度 = 已缓存的token数量！
        start_pos=past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        #  === 步骤3: 词嵌入 + Dropout ===
        hidden_states=self.dropout(self.embed_tokens(input_ids))
        # === 步骤4: 动态切片位置编码（精准对齐当前序列）===
        position_embeddings=(
            self.freqs_cos[start_pos:start_pos+seq_length],
            self.freqs_sin[start_pos:start_pos+seq_length],
        )
        # === 步骤5: Transformer块堆叠（逐层处理）===
        presents=[] # 用于收集每层的KV缓存，供推理时下一步使用
        for layer_idx,(layer,past_kv) in enumerate(zip(self.layers,past_key_values)):
            #zip函数作用：将层列表和KV缓存列表打包成一个迭代器，每次迭代返回一个层和对应的KV缓存，确保一一对应
            hidden_states,present=layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present) # 收集每层的KV缓存
        # === 步骤6: 输出RMSNorm ===
        hidden_states=self.norm(hidden_states)
        # === 步骤7: 聚合MoE辅助损失（仅训练时有效）===
        aux_loss=sum(
            [
                layer.mlp.aux_loss #从每层MoE块收集aux_loss
                for layer in self.layers
                if isinstance(layer.mlp,MoEFeedForward) #仅统计使用MoE的层
            ],
            start=hidden_states.new_zeros(1).squeeze() # 初始化为0，确保类型和设备一致
        )
        return hidden_states,presents,aux_loss
    
class Self_MinimindForCausalLM(PreTrainedModel,GenerationMixin):
    """
    🧠 因果语言模型（Causal LM）头部封装
    📦 数据流：
    Token IDs → MokioMindModel → Hidden States → LM Head → Logits
                                      ↑               ↑
                                (KV缓存/presents)  (权重绑定)
    """
    config_class=Self_Minimindconfig
    def __init__(self,config:Self_Minimindconfig):
        super().__init__(config)
        # === 1. 主干模型：Transformer编码器（含MoE/KV缓存）===
        self.model=Self_MinimindModel(config)
        # === 2. 语言模型头：隐藏状态 → 词表概率 ===
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        # === 3. 权重绑定（关键优化！）===
        # 输入嵌入与输出投影共享同一权重矩阵：参数量减少、梯度对齐更稳定、符合"输入-输出对称性"直觉
        self.model.embed_tokens.weight=self.lm_head.weight

    def forward(
            self,
            input_ids:Optional[torch.Tensor]=None,
            attention_mask:Optional[torch.Tensor]=None,
            labels:Optional[torch.Tensor]=None, #用于训练的标签（目标token IDs）
            past_key_values:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
            use_cache:bool=False,
            logits_to_keep:Union[int,torch.Tensor]=0, # =0表示计算全部logits，>0表示仅计算最后k个位置的logits（推理优化）
            **kwargs,
    )->CausalLMOutputWithPast:
        """
        🔄 前向全流程（训练/推理统一接口）
        
        📌 关键设计：
        • logits_to_keep：动态控制logits计算范围
          - 训练时=0 → 计算全序列logits（用于损失）
          - 推理时=1 → 仅计算最后token logits（提速+省显存）
        • 损失对齐：shift操作实现"用t时刻隐藏状态预测t+1时刻token"
        • aux_loss透传：MoE负载均衡损失自动聚合
        
        返回: CausalLMOutputWithPast (含loss/logits/past_key_values/aux_loss)
        """
        # === 1. 主干模型前向传播（获取隐藏状态 + KV缓存 + aux_loss）===
        hidden_states,past_key_values,aux_loss=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        ) #hidden_states形状：[batch_size, seq_len, hidden_size]
        # === 2. 动态切片隐藏状态（仅保留需要计算logits的位置）===
        #   • 训练: logits_to_keep=0 → slice(None) → 全序列
        #   • 推理: logits_to_keep=1 → slice(-1, None) → 仅最后token
        slice_indices=(
            slice(-logits_to_keep,None)
            if isinstance(logits_to_keep,int) and logits_to_keep>0
            else slice(None)
        )
        logits=self.lm_head(hidden_states[:,slice_indices,:]) # 精准裁剪序列维度，保留batch和特征维度[batch_size, slice_len, vocab_size]
        # === 3. 计算损失（仅训练时）===
        loss=None
        if labels is not None:
            #原始序列:  [I]  [love]  [AI]  [[EOS]]  [?]
            #logits:    h0    h1     h2     h3     h4   → 取[:-1] → [h0, h1, h2, h3]
            #labels:    10    20     30     40     50   → 取[1:]  → [20, 30, 40, 50]
            #预测关系:  h0→20  h1→30  h2→40  h3→50  ✅
            shift_logits=logits[...,:-1,:].contiguous()  #batch维度保持，seq_len维度取取前n-1个，vocab维度保持，contiguous保证内存连续性（有利于后续view操作）
            shift_labels=labels[...,1:].contiguous()
            #   计算交叉熵损失，ignore_index=-100表示忽略标签为-100的位置（如填充位置）
            loss=F.cross_entropy(
                shift_logits.view(-1,shift_logits.size(-1)), # 将batch和seq_len合并为一个维度，vocab维度保持不变
                shift_labels.view(-1),  # -1表示压缩维度，将batch和seq_len合并为一个维度：总token数
                ignore_index=-100,
            )
        # === 4. 返回结果（含损失、logits、KV缓存、aux_loss）===
        output=CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss=aux_loss

        return output



        





            


    



