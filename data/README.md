# 数据目录

完整数据集较大（>100MB），不纳入 git 版本管理。

## 数据来源

- **C4 中文子集** (allenai/c4, zh)：Google C4 多语言语料的中文部分
  ```python
  from datasets import load_dataset
  ds = load_dataset('allenai/c4', 'zh', split='train', streaming=True)
  ```

- **下载命令**（需要 VPN / HF 镜像）：
  ```bash
  HF_ENDPOINT=https://hf-mirror.com python -c "
  from datasets import load_dataset
  import json
  ds = load_dataset('allenai/c4', 'zh', split='train', streaming=True)
  with open('data/c4_zh_5k.jsonl', 'w') as f:
      for i, item in enumerate(ds):
          if i >= 5000: break
          f.write(json.dumps({'id': f'c4_{i}', 'text': item['text'], 'source': 'c4_zh'}, ensure_ascii=False) + '\\n')
  "
  ```

## 小样本演示

`data/samples/c4_zh_sample_100.jsonl` 包含 100 条清洗后的干净数据，可用于：
- CI 自动化验证
- 快速演示管线功能
- 作为指令合成的种子数据
