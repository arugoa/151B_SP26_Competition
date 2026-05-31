#!/usr/bin/env python3
"""
Evaluate an adapter with a cheap second-pass rescue for no-box outputs.

First pass:
  normal strict-final prompt, greedy generation.

If the output does not contain \boxed{...}:
  run a short second pass that shows the original problem and the model's attempted
  solution, then asks the model to output exactly one final boxed answer.

This targets the "no boxed answer" failure mode without doing expensive
self-consistency over every example.
"""

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_SFT_ADAPTER = "./qwen3-4b-thinking-numinamath-lora-1080ti"
DEFAULT_GRPO_ADAPTER = "./qwen3-4b-thinking-numinamath-grpo-boxed-train1026-eval100-strat-lora-1080ti"
DEFAULT_DATA_PATH = "data/public.jsonl"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_strict_prompt(item: Dict[str, Any]) -> str:
    question = item["question"]

    if item.get("options"):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_lines = [f"{letters[i]}. {option}" for i, option in enumerate(item["options"])]
        return (
            "You are an expert mathematician. "
            "Solve the multiple-choice problem step by step, but do not over-explain. "
            "At the end, you must put only the final answer choice letter inside \\boxed{} and stop immediately after the boxed answer. "
            "For example, write \\boxed{A}, not the full answer text. "
            "Your final line must be exactly of the form \\boxed{A}, where A is one of the provided option letters.\n\n"
            f"Problem:\n{question}\n\n"
            "Options:\n" + "\n".join(option_lines)
        )

    return (
        "You are an expert mathematician. "
        "Solve the following math problem step by step, but do not over-explain. "
        "At the end, you must put the final answer inside \\boxed{} and stop immediately after the boxed answer. "
        "Use exact forms such as fractions, radicals, powers, pi, inverse trig functions, logs, or symbolic expressions whenever possible. "
        "Do not replace exact expressions with decimal approximations unless the problem explicitly asks for a decimal. "
        "If an exact symbolic form is available, put that exact form in \\boxed{}. "
        "Do not round decimals unless the problem explicitly asks you to round. "
        "If a decimal answer is necessary, preserve at least 1e-8 precision. "
        "If there are multiple [ANS] blanks, put the answers in order separated by commas inside one \\boxed{}. "
        "If the problem asks for one expression, combine all terms into one expression using + and - signs, not commas. "
        "Your final line must be exactly of the form \\boxed{...}.\n\n"
        f"Problem:\n{question}"
    )


def build_rescue_prompt(item: Dict[str, Any], first_response: str) -> str:
    question = item["question"]

    if item.get("options"):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_lines = [f"{letters[i]}. {option}" for i, option in enumerate(item["options"])]
        return (
            "The previous solution did not end in the required boxed-answer format.\n"
            "Do not solve the problem again from scratch. Use the attempted solution to infer the final answer.\n"
            "Output exactly one line and nothing else: \\boxed{A}, where A is one of the provided option letters.\n\n"
            f"Problem:\n{question}\n\n"
            "Options:\n" + "\n".join(option_lines) + "\n\n"
            "Attempted solution:\n"
            f"{first_response[-5000:]}\n\n"
            "Final boxed answer only:"
        )

    return (
        "The previous solution did not end in the required boxed-answer format.\n"
        "Do not solve the problem again from scratch. Use the attempted solution to infer the final answer.\n"
        "Output exactly one line and nothing else: \\boxed{...}.\n"
        "Prefer exact symbolic forms over decimals unless a decimal is required.\n\n"
        f"Problem:\n{question}\n\n"
        "Attempted solution:\n"
        f"{first_response[-5000:]}\n\n"
        "Final boxed answer only:"
    )


def extract_boxed_content_debug(text: str) -> str:
    text = str(text)
    boxes = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx == -1:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            boxes.append(text[brace_start:i - 1].strip())
        start = i
    return boxes[-1] if boxes else ""


def has_box(text: str) -> bool:
    return bool(extract_boxed_content_debug(text))


def clean_response(text: str) -> str:
    text = str(text).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()
    text = re.sub(r"^\s*(final answer|answer)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def starts_reasoning(text: str) -> bool:
    text = str(text).strip().lower()
    return any(text.startswith(s) for s in [
        "okay", "we need", "let", "first", "to solve", "the problem",
        "i ", "we ", "since", "because",
    ])


def select_stratified_eval(data: List[Dict[str, Any]], eval_size: int, split_seed: int) -> List[Dict[str, Any]]:
    copied = []
    for i, item in enumerate(data):
        item2 = dict(item)
        item2["_row_idx"] = i
        copied.append(item2)

    mcq_items = [x for x in copied if x.get("options")]
    free_items = [x for x in copied if not x.get("options")]

    rng = random.Random(split_seed)
    rng.shuffle(mcq_items)
    rng.shuffle(free_items)

    num_eval_mcq = round(eval_size * len(mcq_items) / len(copied))
    num_eval_free = eval_size - num_eval_mcq

    eval_data = mcq_items[:num_eval_mcq] + free_items[:num_eval_free]
    return sorted(eval_data, key=lambda x: x["_row_idx"])


def load_done_rows(output_path: Path):
    results = []
    done = set()
    if not output_path.exists():
        return results, done
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    results.append(r)
                    done.add(int(r["row_idx"]))
                except Exception:
                    pass
    return results, done


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, input_max_length: int) -> str:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=input_max_length,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    del inputs, output, generated_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


def print_summary(results: List[Dict[str, Any]], label: str) -> None:
    mcq_res = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def ncorrect(xs):
        return sum(bool(r.get("judger_correct", r.get("correct"))) for r in xs)

    def acc(xs):
        return ncorrect(xs) / len(xs) * 100 if xs else 0.0

    print("=" * 70)
    print(f"{label} EVALUATION")
    print("=" * 70)
    print(f"  MCQ        : {ncorrect(mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {ncorrect(free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {ncorrect(results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 70)

    if results:
        no_box_first = [r for r in results if not r.get("first_has_box")]
        rescued = [r for r in results if r.get("used_rescue")]
        rescue_correct = [r for r in rescued if r.get("judger_correct", r.get("correct"))]
        final_no_box = [r for r in results if not r.get("has_box")]
        boxed = [r for r in results if r.get("has_box")]
        correct_boxed = [r for r in boxed if r.get("judger_correct", r.get("correct"))]

        print("\nRescue diagnostics:")
        print(f"  First-pass no-box outputs        : {len(no_box_first)} / {len(results)}")
        print(f"  Rescue pass used                 : {len(rescued)} / {len(results)}")
        print(f"  Rescue correct                   : {len(rescue_correct)} / {len(rescued)}")
        print(f"  Final no-box outputs             : {len(final_no_box)} / {len(results)}")
        print(f"  Correct among final boxed outputs: {len(correct_boxed)} / {len(boxed)}")


def parse_args():
    p = argparse.ArgumentParser(description="Eval with no-box rescue pass.")
    p.add_argument("--adapter", choices=["sft", "grpo", "custom"], required=True)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--split-seed", type=int, default=151)
    p.add_argument("--max-new-tokens", type=int, default=5000)
    p.add_argument("--rescue-max-new-tokens", type=int, default=128)
    p.add_argument("--input-max-length", type=int, default=1024)
    p.add_argument("--rescue-input-max-length", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16"])
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.adapter_dir:
        adapter_dir = args.adapter_dir
    elif args.adapter == "sft":
        adapter_dir = DEFAULT_SFT_ADAPTER
    elif args.adapter == "grpo":
        adapter_dir = DEFAULT_GRPO_ADAPTER
    else:
        raise ValueError("--adapter-dir required for custom adapter")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Run config:")
    print(json.dumps({**vars(args), "resolved_adapter_dir": adapter_dir}, indent=2))

    if not Path(args.data_path).exists():
        raise FileNotFoundError(args.data_path)
    if not Path(adapter_dir).exists():
        raise FileNotFoundError(adapter_dir)
    if not Path("judger.py").exists():
        raise FileNotFoundError("judger.py missing. Run from repo root.")

    sys.path.insert(0, ".")
    from judger import Judger  # type: ignore
    judger = Judger(strict_extract=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    print(f"Loaded adapter: {adapter_dir}")

    data = load_jsonl(args.data_path)
    eval_data = select_stratified_eval(data, args.eval_size, args.split_seed)
    if args.limit is not None:
        eval_data = eval_data[:args.limit]

    print(f"Evaluating {len(eval_data)} examples")
    print("Eval row indices:", [x["_row_idx"] for x in eval_data])

    if not args.resume and output_path.exists():
        output_path.unlink()

    results, done = load_done_rows(output_path) if args.resume else ([], set())
    if done:
        print(f"Resuming from {len(done)} completed rows")

    with open(output_path, "a" if args.resume else "w") as f:
        for item in tqdm(eval_data, desc="Eval + rescue"):
            row_idx = int(item["_row_idx"])
            if row_idx in done:
                continue

            prompt = build_strict_prompt(item)
            first_response = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                input_max_length=args.input_max_length,
            )
            first_has_box = has_box(first_response)

            used_rescue = False
            rescue_response = ""
            raw_response = first_response

            if not first_has_box:
                used_rescue = True
                rescue_prompt = build_rescue_prompt(item, first_response)
                rescue_response = generate_text(
                    model, tokenizer, rescue_prompt,
                    max_new_tokens=args.rescue_max_new_tokens,
                    input_max_length=args.rescue_input_max_length,
                )
                if has_box(rescue_response):
                    raw_response = rescue_response
                else:
                    # If rescue still fails, judge the original response.
                    raw_response = first_response

            is_mcq = bool(item.get("options"))
            gold = item["answer"]
            gold_list = gold if isinstance(gold, list) else [gold]
            gold_list = [str(x) for x in gold_list]

            try:
                correct = bool(judger.auto_judge(
                    pred=raw_response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                ))
            except Exception as e:
                correct = False
                judge_error = repr(e)
            else:
                judge_error = None

            boxed_debug = extract_boxed_content_debug(raw_response)
            record = {
                "id": item.get("id"),
                "row_idx": row_idx,
                "is_mcq": is_mcq,
                "gold": gold,
                "first_response": first_response,
                "first_has_box": first_has_box,
                "used_rescue": used_rescue,
                "rescue_response": rescue_response,
                "raw_response": raw_response,
                "response": clean_response(raw_response),
                "boxed_debug": boxed_debug,
                "judger_correct": correct,
                "correct": correct,
                "judge_error": judge_error,
                "raw_len_chars": len(raw_response),
                "first_raw_len_chars": len(first_response),
                "has_box": has_box(raw_response),
                "starts_reasoning": starts_reasoning(raw_response),
                "adapter": args.adapter,
                "adapter_dir": adapter_dir,
                "max_new_tokens": args.max_new_tokens,
                "rescue_max_new_tokens": args.rescue_max_new_tokens,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

            results.append(record)
            done.add(row_idx)

    print(f"Saved {len(results)} records to {output_path}")
    print_summary(results, label=f"{args.adapter.upper()} + RESCUE")


if __name__ == "__main__":
    main()
