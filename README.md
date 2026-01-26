# SFT-KD — Fine-tuning and Knowledge Distillation Environment

This repository contains a minimal scaffold to prepare an environment for fine-tuning and knowledge distillation using Hugging Face Transformers.

Quick setup

1. Create a Python virtual environment and activate it.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install requirements

```bash
pip install -r requirements.txt
```

3. (Optional) Configure `accelerate` for distributed training

```bash
accelerate config
```

Run examples

- Fine-tune (Trainer + optional PEFT/LoRA):

```bash
accelerate launch scripts/train_finetune.py --model_name_or_path gpt2 --dataset_name glue --task_name mrpc
```

- Distillation (simple logits-matching example):

```bash
accelerate launch scripts/distill.py --teacher_name_or_path bert-base-uncased --student_name_or_path distilbert-base-uncased
```

Notes
- For large models you may need CUDA-enabled `bitsandbytes` and appropriate GPU drivers.
- `peft`/LoRA usage is optional and is annotated in the example script.
