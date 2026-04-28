# Getting Started

## Requirements
- Python 3.10+
- PyTorch 2.x
- transformers, datasets

## Setup
1. Create and activate a virtual environment.
2. Install dependencies listed in pyproject.toml.

## Run
```bash
# Pretrain
python trainer/train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl

# SFT
python trainer/train_full_sft.py --data_path ../dataset/sft_mini_512.jsonl
```

## Notes
- Use smaller batch sizes for single-GPU runs.
- Enable AMP for better throughput on modern GPUs.
