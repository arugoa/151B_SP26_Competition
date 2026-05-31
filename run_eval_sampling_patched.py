#!/usr/bin/env python3
"""
Run a stratified held-out evaluation for Qwen3-4B adapters with optional sampling.

Adds:
  --do-sample
  --temperature
  --top-p
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


def build_finetuned_prompt(item: Dict[str, Any]) -> str:
    question = item["question"]

    if item.get("options"):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_lines = [f"{letters[i]}. {option}" for i, option in enumerate(item["options"])]
        return (
            "You are an expert mathematician. "
            "Solve the multiple-choice problem step by step, but do not over-explain. "
            "You MUST always end your response with \\boxed{} containing only the single letter of the correct option. "
            "No matter how long or complex the reasoning, your very last output MUST be \\boxed{X} where X is one of the provided letters. "
            "Even if you are unsure, commit to the best letter. Never leave the boxed answer blank or omit it. "
            "For example, write \\boxed{A}, not the full answer text.\n\n"
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
        "If a decimal answer is necessary, preserve at least 15 significant digits. "
        "For example: write 19.7745967126229 not 19.78, write 0.564642473395035 not 0.56464247. "
        "If there are multiple [ANS] blanks, put ALL answers in order separated by commas inside ONE \\boxed{}. "
        "If the answer is a tuple or ordered set of values (e.g. the problem asks for a list or sequence), "
        "wrap ALL values together in parentheses inside the box: \\boxed{(v1, v2, v3, ...)}. "
        "If the problem asks for one expression, combine all terms into one expression using + and - signs, not commas. "
        "For 'select all that apply' questions, concatenate the letters without spaces or commas: e.g. \\boxed{ACD} not \\boxed{A,C,D}. "
        "Your final line must be exactly of the form \\boxed{...}.\n\n"
        "Examples of correct precision:\n"
        "  Q: Find x to high precision. A: \\boxed{19.7745967126229}\n"
        "  Q: Compute the three values. A: \\boxed{0.564642473395035, 0.825335614909678, 0.684136808341692}\n"
        "  Q: Find the interval. A: \\boxed{(0.0144256271318742, 0.255574372868126)}\n\n"
        f"Problem:\n{question}"
    )


def normalize_select_all_response(response: str, gold_list: list) -> str:
    """
    If gold is a single concatenated-letter answer like 'ACD', and the model
    output comma-separated letters like A,C,D or multiple \\boxed{A},\\boxed{C},\\boxed{D},
    collapse them so the judger's len check passes.
    """
    import re
    # Check if gold looks like a concatenated multi-letter answer (e.g. 'ACD', 'CF', 'BE')
    if len(gold_list) == 1 and re.match(r'^[A-Z]{2,}$', gold_list[0]):
        n = len(gold_list[0])
        # Find all boxed contents
        boxes = re.findall(r'\\boxed\{([^}]*)\}', response)
        if not boxes:
            return response
        last_box = boxes[-1].strip()
        # If last box has comma-separated single letters matching length, collapse them
        parts = [p.strip() for p in last_box.split(',')]
        if len(parts) == n and all(len(p) == 1 and p.isupper() for p in parts):
            collapsed = ''.join(parts)
            # Replace the last boxed answer with the collapsed version
            response = re.sub(r'\\boxed\{[^}]*\}\s*$', f'\\\\boxed{{{collapsed}}}', response.rstrip())
        # Also handle multiple separate \\boxed{X} at the end
        elif len(boxes) == n and all(len(b.strip()) == 1 and b.strip().isupper() for b in boxes):
            collapsed = ''.join(b.strip() for b in boxes)
            # Strip trailing multiple boxes, replace with one
            response = re.sub(r'(\\boxed\{[A-Z]\}\s*[,\s]*){' + str(n) + r'}$',
                              f'\\\\boxed{{{collapsed}}}', response.rstrip())
    return response


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_response(text: str) -> str:
    text = str(text).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()
    text = re.sub(r"^\s*(final answer|answer)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    return text


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


def starts_reasoning(text: str) -> bool:
    text = str(text).strip().lower()
    bad_starts = ["okay", "we need", "let", "first", "to solve", "the problem", "i ", "we ", "since", "because"]
    return any(text.startswith(s) for s in bad_starts)


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


def load_done_rows(output_path: Path) -> tuple[List[Dict[str, Any]], set[int]]:
    results = []
    done = set()
    if not output_path.exists():
        return results, done
    with open(output_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                results.append(r)
                if r.get("row_idx") is not None:
                    done.add(int(r["row_idx"]))
            except Exception:
                continue
    return results, done


def print_summary(results: List[Dict[str, Any]], label: str) -> None:
    mcq_res = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def n_correct(subset):
        return sum(bool(r.get("judger_correct", r.get("correct"))) for r in subset)

    def acc(subset):
        return n_correct(subset) / len(subset) * 100 if subset else 0.0

    print("=" * 70)
    print(f"{label} EVALUATION")
    print("Scored with judger.py")
    print("=" * 70)
    print(f"  MCQ        : {n_correct(mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {n_correct(free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {n_correct(results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 70)

    if results:
        no_box = [r for r in results if not r.get("has_box")]
        wrong = [r for r in results if not r.get("judger_correct", r.get("correct"))]
        reasoning_start = [r for r in results if r.get("starts_reasoning")]
        avg_len = sum(int(r.get("raw_len_chars", 0)) for r in results) / len(results)
        boxed = [r for r in results if r.get("has_box")]
        correct_boxed = [r for r in boxed if r.get("judger_correct", r.get("correct"))]
        print("\nFailure diagnostics:")
        print(f"  Avg raw response length chars      : {avg_len:.1f}")
        print(f"  Raw outputs with no \\boxed{{}}     : {len(no_box)} / {len(results)}")
        print(f"  Outputs starting with reasoning    : {len(reasoning_start)} / {len(results)}")
        print(f"  Wrong outputs                      : {len(wrong)} / {len(results)}")
        print(f"  Correct among boxed outputs        : {len(correct_boxed)} / {len(boxed)}  ({(len(correct_boxed) / len(boxed) * 100) if boxed else 0:.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adapters on held-out public split with optional sampling.")
    parser.add_argument("--adapter", choices=["sft", "grpo", "custom"], required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=151)
    parser.add_argument("--max-new-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--input-max-length", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--print-failures", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.adapter_dir:
        adapter_dir = args.adapter_dir
    elif args.adapter == "sft":
        adapter_dir = DEFAULT_SFT_ADAPTER
    elif args.adapter == "grpo":
        adapter_dir = DEFAULT_GRPO_ADAPTER
    else:
        raise ValueError("--adapter-dir is required when --adapter custom")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Run config:")
    print(json.dumps({
        "adapter": args.adapter,
        "adapter_dir": adapter_dir,
        "model_id": args.model_id,
        "data_path": args.data_path,
        "output": str(output_path),
        "eval_size": args.eval_size,
        "limit": args.limit,
        "split_seed": args.split_seed,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "resume": args.resume,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
    }, indent=2))

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Could not find data file: {args.data_path}")
    if not Path(adapter_dir).exists():
        raise FileNotFoundError(f"Could not find adapter dir: {adapter_dir}")
    if not Path("judger.py").exists():
        raise FileNotFoundError("Could not find judger.py. Run this from the competition repo/workspace.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(0))
        print("Allocated before load:", round(torch.cuda.memory_allocated() / 1024**3, 3), "GB")
        print("Reserved before load:", round(torch.cuda.memory_reserved() / 1024**3, 3), "GB")

    sys.path.insert(0, ".")
    from judger import Judger  # type: ignore
    judger = Judger(strict_extract=False)

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

    llm = PeftModel.from_pretrained(base_model, adapter_dir)
    llm.eval()

    if torch.cuda.is_available():
        print("Allocated after load:", round(torch.cuda.memory_allocated() / 1024**3, 3), "GB")
        print("Reserved after load:", round(torch.cuda.memory_reserved() / 1024**3, 3), "GB")
    print(f"Loaded adapter: {adapter_dir}")

    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} competition questions")
    eval_data = select_stratified_eval(data, args.eval_size, args.split_seed)
    if args.limit is not None:
        eval_data = eval_data[:args.limit]

    print(f"Evaluating {len(eval_data)} stratified holdout examples")
    print(f"Eval split in this run: {sum(bool(x.get('options')) for x in eval_data)} MCQ, {sum(not x.get('options') for x in eval_data)} free-form")
    print("Eval row indices:", [x["_row_idx"] for x in eval_data])

    if not args.resume and output_path.exists():
        output_path.unlink()

    results = []
    already_done_rows = set()
    if args.resume:
        results, already_done_rows = load_done_rows(output_path)
        if already_done_rows:
            print(f"Resuming: found {len(already_done_rows)} completed row_idx values in {output_path}")

    with open(output_path, "a" if args.resume else "w") as f:
        for item in tqdm(eval_data, desc=f"Generating + scoring [{args.adapter}]"):
            row_idx = int(item["_row_idx"])
            if args.resume and row_idx in already_done_rows:
                continue

            prompt = build_finetuned_prompt(item)
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
            ).to(llm.device)

            generation_kwargs = {
                **inputs,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "use_cache": True,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                output = llm.generate(**generation_kwargs)

            generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
            raw_response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            is_mcq = bool(item.get("options"))
            gold = item["answer"]
            gold_list = gold if isinstance(gold, list) else [gold]
            gold_list = [str(x) for x in gold_list]

            # Normalize select-all responses (e.g. \boxed{A,C,D} -> \boxed{ACD})
            # so the judger's length check doesn't fail on concatenated-letter gold answers
            judged_response = normalize_select_all_response(raw_response, gold_list)

            try:
                correct = bool(judger.auto_judge(pred=judged_response, gold=gold_list, options=[[]] * len(gold_list)))
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
                "raw_response": raw_response,
                "response": clean_response(raw_response),
                "boxed_debug": boxed_debug,
                "judger_correct": correct,
                "correct": correct,
                "judge_error": judge_error,
                "raw_len_chars": len(raw_response),
                "has_box": "\\boxed{" in raw_response,
                "starts_reasoning": starts_reasoning(raw_response),
                "adapter": args.adapter,
                "adapter_dir": adapter_dir,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

            results.append(record)
            already_done_rows.add(row_idx)

            del inputs, output, generated_tokens, generation_kwargs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Saved/resumed {len(results)} records to {output_path}")
    print_summary(results, label=args.adapter.upper())

    wrong = [r for r in results if not r.get("judger_correct", r.get("correct"))]
    if args.print_failures > 0 and wrong:
        print(f"\nFirst {min(args.print_failures, len(wrong))} failures:")
        for r in wrong[: args.print_failures]:
            print("=" * 100)
            print("ID:", r["id"])
            print("ROW IDX:", r["row_idx"])
            print("MCQ:", r["is_mcq"])
            print("GOLD:", r["gold"])
            print("JUDGER CORRECT:", r["judger_correct"])
            print("HAS BOX:", r["has_box"])
            print("BOXED DEBUG:", repr(r["boxed_debug"]))
            print("RAW RESPONSE:")
            print(str(r["raw_response"])[:2500])


if __name__ == "__main__":
    main()
