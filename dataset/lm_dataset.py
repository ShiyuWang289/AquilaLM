from numpy import add
from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# 0.  全局预处理 / 后处理工具函数
# ──────────────────────────────────────────────────────────────────────────────
# ==================== 常量定义（提升可维护性） ====================
SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are a reliable AI assistant. Provide accurate and useful answers.",
]
EMPTY_THINK_PATTERN = "<think>\n\n</think>\n\n"

def pre_processing_chat(conversations:list,add_system_ratio:float=0.2,system_prompts:list=None)->list:   #概率控制：add_system_ratio=0.2 → 20%样本插入
    """对话前处理：智能插入system prompt（提升泛化能力）"""
    prompts=system_prompts if system_prompts is not None else SYSTEM_PROMPTS  #`system_prompts`：支持外部传入定制池（提升灵活性）
    # 仅当首条非system且满足概率时插入:
    if (conversations and conversations[0].get("role")!="system" and random.random()<add_system_ratio): 
        # 随机选择prompt + 拼接原始对话（保持对话结构）
        return [{"role":"system","content":random.choice(prompts)}] + conversations
    return conversations

def post_processing_chat(prompt_content:str,empty_think_ratio:float=0.05)->str:
    """对话后处理：智能清理空思考块（CoT训练关键）
        （<think>\n\n</think>）无信息量  → 95%概率删除，避免模型学习"无意义思考"
        5%概率保留空块 → 让模型学会处理"思考内容为空"的边界情况
    """
    # 仅当存在空思考块且满足删除概率时执行清理
    if (EMPTY_THINK_PATTERN in prompt_content and random.random() > empty_think_ratio ):
        return prompt_content.replace(EMPTY_THINK_PATTERN, "")
    return prompt_content


# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    """ 训练目标：Next-Token Prediction（下一个 token 预测）
         数据格式：{"text": "一段原始文本"}
         训练特点：
           - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
           - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
           - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
           - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。"""
    def __init__(self,data_path,tokenizer,max_length=512):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples=load_dataset("json",data_files=data_path,split="train")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        sample=self.samples[index]

        # === 1. 文本编码（严格控制长度，预留BOS/EOS位置）===
        tokens=self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False, # 先不添加特殊标记，后面手动添加BOS/EOS
            max_length=self.max_length-2, # 留出位置给BOS和EOS
            truncation=True,  
        ).input_ids
        # === 2. 拼接特殊token（BOS、EOS）===
        tokens=( [self.tokenizer.bos_token_id] + tokens +[self.tokenizer.eos_token_id] )
        # === 3. Padding到固定长度（右侧填充）===
        padded_tokens=tokens+[self.tokenizer.pad_token_id]*(self.max_length - len(tokens))
        input_ids=torch.tensor(padded_tokens,dtype=torch.long)
        # === 4. 构建labels（防止pad参与loss计算）===
        labels=input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id]=-100
        # === 5. 构建attention_mask（告诉模型哪些位置有效，哪些是pad） ===
        attention_mask=(input_ids != self.tokenizer.pad_token_id).long()  #确保注意力层忽略填充部分

        return{
            "input_ids":input_ids,
            "lables":labels,
            "attention_mask":attention_mask
        }
# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset —— 有监督微调（Supervised Fine-Tuning）数据集
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    """
    # 训练目标：让模型学会"只预测 assistant 回复"，忽略 user/system 输入
    # 数据格式：{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}
    # 训练特点：
    #   - 通过 generate_labels 扫描 bos_id（assistant 回复起始标记）定位每段回复，
    #     仅将 assistant 回复的 token 位置设为有效 label，其余全部为 -100。
    #   - 支持 function calling：若 system 消息携带 "functions" 字段，
    #     会透传给 apply_chat_template，生成带工具描述的提示词。
    #   - 与 PretrainDataset 的关键区别：标签是"稀疏"的，只有 assistant 部分非 -100。
    """
    def __init__(self,jsonl_path:str,tokenizer,max_length:int=1024):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.samples=load_dataset("json",data_file=jsonl_path,split="train") #load_dataset流式加载,split="train" 将单文件视为训练集；
        self.bos_id=tokenizer(f"{tokenizer.bos_token}assistant\n",add_special_tokens=False).input_ids
        self.eos_id=tokenizer(f"{tokenizer.eos_token}\n",add_special_tokens=False).input_ids
    def __len__(self)->int:
        return len(self.samples)
    def creat_chat_prompt(self,conversations:list)->str:
        """将对话历史转换为模型输入字符串"""
        messages=conversations.copy() # 防止污染原始数据
        # 检测Function Calling场景：system消息含functions字段
        tools=(
            conversations[0]["functions"]
            if (conversations and conversations[0]["role"]=="system" and conversations[0].get("functions"))
            else None
        )
        # 生成标准化对话模板（含role标记、special tokens等）
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
    def generate_labels(self,input_ids:list)->list:
        """生成稀疏标签序列：仅assistant回复部分参与loss计算"""
        labels=[-100]*len(input_ids) # 初始化全忽略
        i=0
        while i<len(input_ids):
            # 检测是否匹配assistant回复起始标记
            if input_ids[i:i+len(self.bos_id)]==self.bos_id:
                start=i+len(self.bos_id)
                end=start
                # 向后扫描寻找结束标记
                while end<len(input_ids):
                    if input_ids[end:end+len(self.eos_id)]==self.eos_id:
                        break
                    end+=1
                 # 关键：将[回复内容 + EOS]区间设为有效label（避免无限生成）
                end_pos=min(end+len(self.eos_id),len(input_ids))
                for j in range(start,end_pos):
                    labels[j]=input_ids[j]  # 仅此处参与loss计算
                i=end_pos  # 更新i，跳过已处理区间（支持多轮对话）
            else:
                i+=1
            return labels
    def __getitem__(self,index:int):
        """单样本处理全流程"""
        sample=self.samples[index]
        # === Step 1: 数据增强（pre_processing实现:随机插入system prompt）===
        conversations = pre_processing_chat(sample["conversations"])
        # === Step 2: 生成标准化对话字符串 ===
        prompt = self.create_chat_prompt(conversations)
        # === Step 3: 安全清洗（post_processing实现:清理空<think>块）===
        prompt = post_processing_chat(prompt)
        # === Step 4: Tokenize + 截断/填充 ===
        # [:max_length] 截断；+ [pad]*n 右侧填充（因果LM要求）
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # === Step 5: 生成稀疏标签（核心！）===
        labels = self.generate_labels(input_ids)
        # === Step 6: 生成attention_mask（关键！）===
        attention_mask = (
            torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),      # 稀疏监督核心
            "attention_mask": attention_mask                        # 防padding干扰
        }

# ──────────────────────────────────────────────────────────────────────────────
# 3. DPODataset —— 直接偏好优化（Direct Preference Optimization）数据集
# ──────────────────────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    """
    # 【理论重点】DPO 数据格式与训练目标：
    #   数据：{"chosen": [对话列表], "rejected": [对话列表]}
    #   目标：最大化 chosen 的对数似然，最小化 rejected 的，使输出更符合人类偏好
    # 与 SFT 数据集的核心区别：
    #   SFT: 每条样本返回 1 份序列 (input_ids, labels)
    #   DPO: 每条样本返回 2 份序列 (chosen + rejected)，训练时做对比
    #        loss_mask 仅标记 assistant 回复部分，保证对比信号来自模型实际输出
    #
    # 自回归错位：x = tokens[:-1], y = tokens[1:], mask = mask[1:]
    #   x[t] 预测 y[t] = tokens[t+1]，标准 next-token 格式"""
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # assistant 回复的起止标记，用于 generate_loss_mask 中定位回复区间
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def _encode_conversation(self, messages):
        """将对话列表渲染为字符串 → tokenize → 生成 loss_mask → 构造自回归训练对
        
        返回: (x, y, mask, attention_mask)，均为 [max_length-1] 的 tensor
        """
        # 渲染 + tokenize
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = post_processing_chat(prompt)
        encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")

        ids = encoding["input_ids"]
        loss_mask = self.generate_loss_mask(ids)

        # 【语法难点】自回归错位：x = ids[:-1] 预测 y = ids[1:]
        #   mask 取 [1:] 与 y 对齐，决定哪些位置参与 loss 计算
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        attn_mask = (x != self.padding).long()  # padding 位置为 0，有效位置为 1

        return x, y, mask, attn_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        x_c, y_c, mask_c, attn_c = self._encode_conversation(sample["chosen"])
        x_r, y_r, mask_r, attn_r = self._encode_conversation(sample["rejected"])

        return {
            "x_chosen": x_c, "y_chosen": y_c,
            "mask_chosen": mask_c, "attention_mask_chosen": attn_c,
            "x_rejected": x_r, "y_rejected": y_r,
            "mask_rejected": mask_r, "attention_mask_rejected": attn_r,
        }

    def generate_loss_mask(self, input_ids):
        """生成 0/1 掩码：assistant 回复区间为 1，其余为 0
        
        算法：扫描 bos_id 标记 → 找到对应 eos_id → 区间内置 1
        与 SFTDataset.generate_labels 逻辑相同，只是返回 mask 而非 token id
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


# ──────────────────────────────────────────────────────────────────────────────
# 4. RLAIFDataset —— 基于 AI 反馈的强化学习数据集（用于 PPO / GRPO）
# ──────────────────────────────────────────────────────────────────────────────
class RLAIFDataset(Dataset):
    """训练目标：为 RL 训练提供"问题-参考答案"对，由 actor 在线采样生成回复，再由 reward model 或规则函数打分优化
       数据格式：{"conversations": [{"content": "..."}, {"content": "..."}]}
       奇数索引 (0,2,4...) 为 user 发言， 偶数索引 (1,3,5...) 为 assistant 发言（最后一条为参考答案）
    """
    def __init__(self,jsonl_path:str,tokenizer,max_length:int=1024):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.samples=load_dataset("json",data_file=jsonl_path,split="train") #load_dataset流式加载,split="train" 将单文件视为训练集；
        self.bos_id=tokenizer(f"{tokenizer.bos_token}assistant",add_special_tokens=False).input_ids
        self.eos_id=tokenizer(f"{tokenizer.eos_token}",add_special_tokens=False).input_ids
    def __len__(self)->int:
        return len(self.samples)
    def create_chat_prompt(self,conversations):
        """从对话列表中分离问题和参考答案
           处理流程：
           1. 按奇偶索引为每条消息分配 user/assistant 角色。
           2. 记录最后一条消息内容为 answer（即本轮期望的参考回答）。
           3. 用除最后一条之外的消息渲染 prompt，并开启 add_generation_prompt=True，使模板在末尾自动追加"assistant 开始回复"的引导标记。
           4. RL actor 收到 prompt 后进行 rollout，生成的回复与 answer 对比打分。
        """
        messages=[]
        answer=""
        for i,turn in enumerate(conversations):
            role="user" if i%2==0 else "assistant"
            messages.append({"role":role,"content":turn["content"]}) # 为content分配角色
            answer=turn["content"]  # 最后一条消息即参考答案
            prompt=self.tokenizer.apply_chat_template(
                messages[:-1], # prompt中不包含最后一条消息（即参考答案）
                tokenize=False,
                add_generation_prompt=True  # 在末尾追加续写引导 token，告诉模型"现在开始生成"
            )
            prompt=post_processing_chat(prompt) # 清理空思考块（同SFTDataset）
            return prompt,answer
        def __getitem__(self,index):
            sample=self.samples[index]
            prompt,answer=self.create_chat_prompt(sample["conversations"])
            return {"prompt":prompt,"answer":answer}
if __name__=="__main__":
    pass 