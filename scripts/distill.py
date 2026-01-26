"""Simple knowledge distillation example: match teacher logits.

This is a minimal training loop showing how to compute KL loss between
teacher and student logits. It is intended as a starting point and not
production-optimized.
"""
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def soft_cross_entropy(p_logits, q_logits, temperature=2.0):
    p = torch.nn.functional.log_softmax(p_logits / temperature, dim=-1)
    q = torch.nn.functional.softmax(q_logits / temperature, dim=-1)
    return torch.nn.functional.kl_div(p, q, reduction="batchmean") * (temperature ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_name_or_path", required=True)
    parser.add_argument("--student_name_or_path", required=True)
    parser.add_argument("--dataset_name", default="glue")
    parser.add_argument("--task_name", default="mrpc")
    args = parser.parse_args()

    ds = load_dataset(args.dataset_name, args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_name_or_path)

    def prep(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(prep, batched=True)

    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_name_or_path)
    student = AutoModelForSequenceClassification.from_pretrained(args.student_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device).eval()
    student.to(device).train()

    optim = torch.optim.AdamW(student.parameters(), lr=5e-5)

    loader = torch.utils.data.DataLoader(ds["train"], batch_size=8)

    for epoch in range(1):
        for batch in loader:
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            loss_kd = soft_cross_entropy(teacher_logits, student_logits)

            optim.zero_grad()
            loss_kd.backward()
            optim.step()

    print("Distillation run finished")


if __name__ == "__main__":
    main()
