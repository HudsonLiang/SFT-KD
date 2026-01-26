"""Minimal fine-tuning example using Hugging Face Trainer.

Supports optional PEFT/LoRA integration when `peft` is installed.
"""
import argparse
from datasets import load_dataset
from datasets import DownloadMode
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_name", default="glue")
    parser.add_argument("--task_name", default="mrpc")
    parser.add_argument("--output_dir", default="./outputs/finetune")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset from cache (if available)...")
    raw = load_dataset(args.dataset_name, args.task_name, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

    # If QA mode requested, use seq2seq fine-tuning (question/context -> answer).
    if args.qa:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

        def preprocess(batch):
            inputs = []
            targets = []
            for q_idx in range(len(batch.get("question", [])) if "question" in batch else range(len(batch[next(iter(batch))]))):
                # Prefer 'question' field, otherwise try common alternatives
                question = batch.get("question", [None] * len(batch))[q_idx] if "question" in batch else None
                context = None
                if "context" in batch:
                    context = batch["context"][q_idx]
                # answers may be a list or a single field
                if "answers" in batch:
                    ans = batch["answers"][q_idx]
                    # HF QA 'answers' is often a dict with 'text' list
                    if isinstance(ans, dict) and "text" in ans:
                        answer = ans["text"][0] if len(ans["text"]) > 0 else ""
                    elif isinstance(ans, list):
                        answer = ans[0] if len(ans) > 0 else ""
                    else:
                        answer = str(ans)
                else:
                    # fallback to simple 'answer' field
                    answer = batch.get("answer", [""] * len(batch))[q_idx] if "answer" in batch else ""

                if question is None:
                    # fallback: join two sentence-like columns if present
                    s1 = batch.get("sentence1", [None] * len(batch))[q_idx]
                    s2 = batch.get("sentence2", [None] * len(batch))[q_idx]
                    question = ": ".join([str(x) for x in (s1, s2) if x is not None])

                if context:
                    inp = f"question: {question}  context: {context}"
                else:
                    inp = f"question: {question}"

                inputs.append(inp)
                targets.append(answer)

            model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

            # replace pad token id's in labels by -100 to ignore in loss
            label_ids = labels["input_ids"]
            label_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in lab] for lab in label_ids]
            model_inputs["labels"] = label_ids
            return model_inputs

        dataset = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None) or dataset.get("validation_matched", None),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        return

    # Legacy classification/pair-mode has been removed. This script
    # now focuses on QA-style seq2seq fine-tuning only (use `--qa`).


if __name__ == "__main__":
    main()
