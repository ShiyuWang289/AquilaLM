# 入门指南

## 依赖
- Python 3.10+
- PyTorch 2.x
- transformers, datasets

## 配置
1. 创建并激活虚拟环境。
2. 安装 pyproject.toml 中的依赖。

## 运行
```bash
# 预训练
python trainer/train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl

# SFT
python trainer/train_full_sft.py --data_path ../dataset/sft_mini_512.jsonl
```

## 说明
- 单卡运行建议使用较小 batch size。
- 现代 GPU 建议开启 AMP 以提升吞吐。
