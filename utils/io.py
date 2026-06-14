"""JSONL 读写工具"""

import json
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件，每行一条 JSON"""
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def save_jsonl(data: List[Dict], path: str) -> None:
    """保存为 JSONL，ensure_ascii=False 保留中文"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_texts(data: List[Dict]) -> List[str]:
    """从 JSON 列表中提取文本字段（兼容多种命名）"""
    texts = []
    for item in data:
        t = item.get("text") or item.get("content") or item.get("instruction") or ""
        if t:
            texts.append(t)
    return texts
