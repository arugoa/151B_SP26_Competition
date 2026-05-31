#!/usr/bin/env python3
"""
Train a competition-specific LoRA adapter on rejection-sampling SFT data.

Input JSONL from generate_rs_sft_data.py with fields:
  prompt, completion, source, is_mcq, row_idx, ...

This script uses completion-only loss by masking the user prompt tokens with -100.
Run from the CSE151B repo/workspace.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_INIT_ADAPTER = "./qwen3-4b-thinking-numinamath-lora-1080ti"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class CompletionOnlyChatDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = row["prompt"]
        completion = row["completion"].strip()

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = list(input_ids)

        # Mask user prompt / generation prefix. If the prompt was truncated, mask as much as still exists.
        mask_len = min(len(prompt_ids), len(labels))
        labels[:mask_len] = [-100] * mask_len

        # If truncation removed all assistant tokens, keep at least the last few tokens trainable.
        if all(x == -100 for x in labels):
            keep = min(64, len(labels))
            labels[-keep:] = input_ids[-keep:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RS-SFT LoRA adapter.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--train-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-adapter-dir", default=DEFAULT_INIT_ADAPTER, help="Continue from existing SFT adapter. Use 'none' to train a fresh LoRA.")
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=151)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading. Not recommended on 24GB GPUs.")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(args.train_file)
    random.shuffle(rows)

    print("Run config:")
    print(json.dumps({**vars(args), "num_train_rows": len(rows)}, indent=2))
    if not rows:
        raise ValueError("No rows loaded from train file")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    if args.init_adapter_dir.lower() != "none":
        if not Path(args.init_adapter_dir).exists():
            raise FileNotFoundError(args.init_adapter_dir)
        print(f"Continuing from adapter: {args.init_adapter_dir}")
        model = PeftModel.from_pretrained(model, args.init_adapter_dir, is_trainable=True)
    else:
        print("Training fresh LoRA adapter")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    ds = CompletionOnlyChatDataset(rows, tokenizer, max_length=args.max_length)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved RS-SFT adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
