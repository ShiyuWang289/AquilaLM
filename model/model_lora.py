import torch
from torch import nn


class LoRA(nn.Module):
    """
    低秩适配器 ΔW = B @ A
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)     # A,使x降维: x(k) → 低秩空间(r)
        self.B = nn.Linear(rank, out_features, bias=False)    # B,使结果升维: 低秩空间(r) → 输出(d)

        nn.init.normal_(self.A.weight, std=0.02)  # A 高斯初始化：提供初始梯度多样性
        nn.init.zeros_(self.B.weight)             # B 全零初始化：防止随机增量破坏预训练知识

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：ΔWx = B(A(x))
        """
        return self.B(self.A(x))


# ══════════════════ 工具函数 ══════════════════

def apply_lora(model: nn.Module, rank: int = 8) -> None:
    """
    对模型中所有方阵线性层注入 LoRA（无侵入式修改）：h = W₀x + BAx

    语法要点：
    - named_modules(): 递归遍历所有子模块，返回 (名称, 模块) 元组
    - isinstance(): 类型检查，判断 module 是否为 nn.Linear
    - setattr 的等价写法：module.lora = lora，动态给对象添加属性
    """

    def _make_forward(orig_fn, lora_fn):
        """
        工厂函数：为每个线性层创建独立的闭包，
        每次调用 _make_forward() 创建新作用域，orig_fn / lora_fn 被"冻结"在各自的闭包中 
        """
        def fwd(x):
            return orig_fn(x) + lora_fn(x)   # W₀x + BAx
        return fwd

    for _, module in model.named_modules(): # 返回 (名称, 模块) 元组
        # 仅对方阵、线性层应用（如注意力层的 Q/K/V/O 投影）
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank)
            module.lora = lora                  # 动态挂载，后续 save/load 通过 hasattr 检测
            module.forward = _make_forward(
                module.forward,                 # orig_fn: 冻结的 W₀x
                lora                            # lora_fn: 可训练的 BAx
            )

def save_lora_weights(model: nn.Module, path: str) -> None:
    """
    仅保存 LoRA 权重（体积极小，通常 < 原模型 1%）
    
    语法要点：
    - hasattr(obj, "attr"): 检查对象是否具有某属性，返回 bool
    - state_dict(): PyTorch 标准接口，返回 {参数名: 张量} 字典
    - f-string 拼接键名：确保加载时能定位到正确的模块
    """
    state = {}
    for name, module in model.named_modules(): 
        if hasattr(module, "lora"):  # 检查对象是否具有lora属性
            for k, v in module.lora.state_dict().items():
                state[f"{name}.lora.{k}"] = v.clone()  
    torch.save(state, path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    """
    从文件加载 LoRA 权重（支持多任务 < 100ms 热切换）

    语法要点：
    - k[len(prefix):]: 字符串切片去前缀，比 replace 安全
      例: "layer.0.lora.A.weight"[len("layer.0.lora."):] → "A.weight"
    - 字典推导式 {k: v for k, v in ... if cond}: 过滤出属于当前模块的参数
    """
    device = next(model.parameters()).device            # 取第一个参数的设备，确保加载到同一设备
    state = torch.load(path, map_location=device)       # map_location: 自动映射到目标设备

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora."                     # 例: "encoder.layer.0.attn.q_proj.lora."
            sub_state = {
                k[len(prefix):]: v                       # 去前缀 → state_dict 期望的键名
                for k, v in state.items()
                if k.startswith(prefix)                  # 筛选属于本模块的参数
            }
            if sub_state:
                module.lora.load_state_dict(sub_state)   # 严格匹配加载