#!/usr/bin/env python3
"""
Generate competition-specific rejection-sampling SFT data.

For each public training example outside the held-out eval split:
  1. Sample K full strict-final completions from the current best adapter.
  2. Judge each sample with judger.py.
  3. Save all judged-correct completions as SFT examples.
  4. If no sample is correct, optionally save a short gold-answer fallback target.

Output JSONL schema:
  {
    "row_idx": int,
    "id": ..., "is_mcq": bool,
    "prompt": str,
    "completion": str,
    "source": "sample_correct" | "gold_fallback",
    "sample_idx": int | null,
    "gold": ...,
    "boxed_debug": str,
    "raw_len_chars": int,
    "num_correct_samples": int,
    "num_boxed_samples": int
  }

Run from the CSE151B repo/workspace where data/public.jsonl and judger.py exist.
"""

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_SFT_ADAPTER = "./qwen3-4b-thinking-numinamath-lora-1080ti"
DEFAULT_DATA_PATH = "data/public.jsonl"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    eval_data = sorted(eval_data, key=lambda x: x["_row_idx"])
    return eval_data


def add_row_idx(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, item in enumerate(data):
        x = dict(item)
        x["_row_idx"] = i
        out.append(x)
    return out


def build_strictfinal_prompt(item: Dict[str, Any]) -> str:
    """The strict-final prompt that gave your current best 52% eval."""
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


def extract_boxed_content_debug(text: str) -> str:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    depth = 1
    chars: List[str] = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
    return ""  # incomplete box


def canonical_gold_answer(item: Dict[str, Any]) -> str:
    ans = item["answer"]
    if isinstance(ans, list):
        return ", ".join(str(x) for x in ans)
    return str(ans)


def build_gold_fallback_completion(item: Dict[str, Any]) -> str:
    """Short fallback target for examples no sampled solution solved."""
    if item.get("options"):
        # MCQ gold may be either "A" or ["A"] depending on data.
        ans = item["answer"]
        if isinstance(ans, list):
            ans_str = str(ans[0])
        else:
            ans_str = str(ans)
        ans_str = ans_str.strip().upper()
        return f"The final answer is \\boxed{{{ans_str}}}."

    return f"The final answer is \\boxed{{{canonical_gold_answer(item)}}}."


def judge_response(judger: Any, response: str, item: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    gold = item["answer"]
    gold_list = gold if isinstance(gold, list) else [gold]
    gold_list = [str(x) for x in gold_list]
    try:
        correct = bool(judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list)))
        return correct, None
    except Exception as e:
        return False, repr(e)


def load_done_rows(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.exists():
        return done
    with open(output_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                # A row is considered done if any record for it exists. This is okay because this script
                # writes all saved records for a row contiguously before moving to the next row.
                done.add(int(obj["row_idx"]))
            except Exception:
                pass
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate rejection-sampling SFT data from the public train split.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--adapter-dir", default=DEFAULT_SFT_ADAPTER)
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-size", type=int, default=100, help="Held-out validation split size to exclude from generation.")
    p.add_argument("--split-seed", type=int, default=151)
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of train examples for smoke/pilot runs.")
    p.add_argument("--k", type=int, default=4, help="Samples per training problem.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=3000)
    p.add_argument("--input-max-length", type=int, default=1024)
    p.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--save-all-correct", action="store_true", help="Save all correct samples instead of only the shortest correct sample.")
    p.add_argument("--no-gold-fallback", action="store_true", help="Do not save fallback examples for unsolved rows.")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not Path(args.data_path).exists():
        raise FileNotFoundError(args.data_path)
    if not Path(args.adapter_dir).exists():
        raise FileNotFoundError(args.adapter_dir)
    if not Path("judger.py").exists():
        raise FileNotFoundError("Run from repo root containing judger.py")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.resume:
        print(f"Overwriting {out_path}")
        out_path.unlink()
    done_rows = load_done_rows(out_path) if args.resume else set()

    print("Run config:")
    print(json.dumps(vars(args), indent=2))
    if done_rows:
        print(f"Resuming: {len(done_rows)} row_idx already present in {out_path}")

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
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    print(f"Loaded adapter: {args.adapter_dir}")

    data = add_row_idx(load_jsonl(args.data_path))
    eval_rows = {int(x["_row_idx"]) for x in select_stratified_eval(data, args.eval_size, args.split_seed)}
    train_data = [x for x in data if int(x["_row_idx"]) not in eval_rows]
    if args.limit is not None:
        train_data = train_data[: args.limit]

    print(f"Loaded {len(data)} total examples")
    print(f"Excluded {len(eval_rows)} eval rows")
    print(f"Generating RS data for {len(train_data)} train examples, K={args.k}")

    n_rows = 0
    n_rows_solved = 0
    n_correct_saved = 0
    n_fallback_saved = 0

    mode = "a" if args.resume else "w"
    with open(out_path, mode) as f:
        for item in tqdm(train_data, desc="RS generate"):
            row_idx = int(item["_row_idx"])
            if row_idx in done_rows:
                continue
            n_rows += 1
            prompt = build_strictfinal_prompt(item)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.input_max_length,
            ).to(model.device)

            candidates: List[Dict[str, Any]] = []
            num_boxed = 0
            for sample_idx in range(args.k):
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                gen_tokens = output[0][inputs["input_ids"].shape[-1]:]
                response = tokenizer.decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                has_box = "\\boxed{" in response and bool(extract_boxed_content_debug(response))
                if has_box:
                    num_boxed += 1
                correct, judge_error = judge_response(judger, response, item)
                if correct:
                    candidates.append({
                        "sample_idx": sample_idx,
                        "completion": response,
                        "boxed_debug": extract_boxed_content_debug(response),
                        "raw_len_chars": len(response),
                        "judge_error": judge_error,
                    })
                del output, gen_tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Prefer all correct if requested; otherwise save the shortest correct sample to bias concise solutions.
            saved_this_row = []
            if candidates:
                n_rows_solved += 1
                candidates_sorted = sorted(candidates, key=lambda c: c["raw_len_chars"])
                saved_this_row = candidates_sorted if args.save_all_correct else candidates_sorted[:1]
                for c in saved_this_row:
                    rec = {
                        "row_idx": row_idx,
                        "id": item.get("id"),
                        "is_mcq": bool(item.get("options")),
                        "prompt": prompt,
                        "completion": c["completion"],
                        "source": "sample_correct",
                        "sample_idx": c["sample_idx"],
                        "gold": item.get("answer"),
                        "boxed_debug": c["boxed_debug"],
                        "raw_len_chars": c["raw_len_chars"],
                        "num_correct_samples": len(candidates),
                        "num_boxed_samples": num_boxed,
                    }
                    f.write(json.dumps(rec) + "\n")
                    n_correct_saved += 1
            elif not args.no_gold_fallback:
                rec = {
                    "row_idx": row_idx,
                    "id": item.get("id"),
                    "is_mcq": bool(item.get("options")),
                    "prompt": prompt,
                    "completion": build_gold_fallback_completion(item),
                    "source": "gold_fallback",
                    "sample_idx": None,
                    "gold": item.get("answer"),
                    "boxed_debug": canonical_gold_answer(item),
                    "raw_len_chars": None,
                    "num_correct_samples": 0,
                    "num_boxed_samples": num_boxed,
                }
                f.write(json.dumps(rec) + "\n")
                n_fallback_saved += 1

            f.flush()
            os.fsync(f.fileno())
            done_rows.add(row_idx)
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=" * 70)
    print("RS DATA GENERATION SUMMARY")
    print("=" * 70)
    print(f"Rows processed this run      : {n_rows}")
    print(f"Rows solved by sampling      : {n_rows_solved} / {n_rows} ({(100*n_rows_solved/n_rows) if n_rows else 0:.2f}%)")
    print(f"Correct completions saved    : {n_correct_saved}")
    print(f"Gold fallback examples saved : {n_fallback_saved}")
    print(f"Output                       : {out_path}")


if __name__ == "__main__":
    main()
