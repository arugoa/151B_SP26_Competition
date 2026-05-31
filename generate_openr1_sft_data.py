#!/usr/bin/env python3
"""
Generate SFT data from OpenR1-Math-style Hugging Face datasets.

Output format is compatible with train_rs_sft.py:
  {"prompt": ..., "completion": ..., "source": ..., ...}

This script is intentionally schema-robust because OpenR1-style datasets may expose
different columns such as:
  problem/question/prompt
  solution/generation/generations/messages/answer
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


def get_text(x: Any) -> str:
    """Best-effort conversion of nested HF dataset fields to text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        # Common chat message / generation dict keys
        for k in ["content", "text", "solution", "generation", "answer", "output", "response"]:
            if k in x and x[k] is not None:
                t = get_text(x[k])
                if t:
                    return t
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, list):
        parts = [get_text(v) for v in x]
        parts = [p for p in parts if p]
        return "\n".join(parts)
    return str(x).strip()


def extract_from_messages(messages: Any) -> tuple[str, str]:
    """Extract user prompt and assistant completion from chat messages if present."""
    if not isinstance(messages, list):
        return "", ""

    user_parts = []
    assistant_parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).lower()
        content = get_text(m.get("content", ""))
        if not content:
            continue
        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    return "\n\n".join(user_parts).strip(), "\n\n".join(assistant_parts).strip()


def first_nonempty(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in row:
            t = get_text(row[k])
            if t:
                return t
    return ""


def extract_problem(row: Dict[str, Any]) -> str:
    # Prefer explicit problem/question fields.
    problem = first_nonempty(row, ["problem", "question", "prompt", "input", "query"])
    if problem:
        return problem

    # Fall back to chat messages.
    msg_prompt, _ = extract_from_messages(row.get("messages"))
    if msg_prompt:
        return msg_prompt

    # Some datasets may use conversations.
    msg_prompt, _ = extract_from_messages(row.get("conversations"))
    if msg_prompt:
        return msg_prompt

    return ""


def choose_generation_from_list(values: Any) -> str:
    """Pick the first nonempty generation from nested/list fields."""
    if values is None:
        return ""

    if isinstance(values, list):
        for v in values:
            t = get_text(v)
            if t:
                return t
        return ""

    return get_text(values)


def extract_completion(row: Dict[str, Any]) -> str:
    # Explicit worked solution fields.
    for k in [
        "solution", "solutions", "generation", "generations",
        "correct_generation", "correct_generations",
        "response", "output", "completion",
    ]:
        if k in row:
            if isinstance(row[k], list):
                t = choose_generation_from_list(row[k])
            else:
                t = get_text(row[k])
            if t:
                return t

    # Chat messages fallback.
    _, msg_completion = extract_from_messages(row.get("messages"))
    if msg_completion:
        return msg_completion

    _, msg_completion = extract_from_messages(row.get("conversations"))
    if msg_completion:
        return msg_completion

    return ""


def extract_answer(row: Dict[str, Any]) -> str:
    return first_nonempty(row, ["answer", "final_answer", "ground_truth", "gt", "label"])


def strip_noise(text: str) -> str:
    text = str(text).strip()
    # Remove common /no_think system marker if it leaks into data.
    text = text.replace("/no_think", "").strip()
    return text


def has_boxed(text: str) -> bool:
    return "\\boxed{" in text or "\\boxed " in text


def append_box_if_needed(completion: str, answer: str) -> str:
    completion = strip_noise(completion)
    answer = strip_noise(answer)
    if has_boxed(completion):
        return completion
    if answer:
        return completion.rstrip() + f"\n\nTherefore, the final answer is \\boxed{{{answer}}}"
    return completion


def strict_prompt(problem: str) -> str:
    return (
        "You are an expert mathematician. "
        "Solve the following math problem step by step, but do not over-explain. "
        "At the end, you must put the final answer inside \\boxed{} and stop immediately after the boxed answer. "
        "Use exact forms such as fractions, radicals, powers, pi, inverse trig functions, logs, or symbolic expressions whenever possible. "
        "Do not replace exact expressions with decimal approximations unless the problem explicitly asks for a decimal. "
        "If an exact symbolic form is available, put that exact form in \\boxed{}. "
        "Your final line must be exactly of the form \\boxed{...}.\n\n"
        f"Problem:\n{problem}"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate train_rs_sft.py-compatible data from OpenR1-Math.")
    p.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    p.add_argument("--config", default=None, help="Optional HF dataset config/subset, e.g. default.")
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=151)
    p.add_argument("--max-problem-chars", type=int, default=6000)
    p.add_argument("--max-completion-chars", type=int, default=12000)
    p.add_argument("--min-completion-chars", type=int, default=100)
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    p.add_argument("--streaming", action="store_true", help="Use streaming load_dataset.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print("Loading dataset...")
    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    # Print column names when available.
    try:
        print("Columns:", list(ds.column_names))
    except Exception:
        print("Columns unavailable in streaming mode.")

    if args.streaming:
        ds_iter = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    else:
        ds = ds.shuffle(seed=args.seed)
        ds_iter = ds

    out = []
    seen = 0
    skipped = 0

    for row in ds_iter:
        row = dict(row)
        seen += 1

        problem = strip_noise(extract_problem(row))
        completion = strip_noise(extract_completion(row))
        answer = strip_noise(extract_answer(row))

        if not problem or not completion:
            skipped += 1
            continue

        if len(problem) > args.max_problem_chars:
            skipped += 1
            continue

        completion = append_box_if_needed(completion, answer)

        if len(completion) < args.min_completion_chars:
            skipped += 1
            continue
        if len(completion) > args.max_completion_chars:
            # Avoid overly long R1 traces that will be truncated heavily during SFT.
            skipped += 1
            continue

        out.append({
            "id": len(out),
            "source": args.dataset,
            "prompt": strict_prompt(problem),
            "completion": completion,
            "problem": problem,
            "answer": answer,
            "raw_keys": list(row.keys()),
        })

        if len(out) >= args.max_rows:
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "output": str(output_path),
        "seen_rows": seen,
        "written_rows": len(out),
        "skipped_rows": skipped,
        "max_rows": args.max_rows,
    }, indent=2))

    if out:
        print("Example:")
        print(json.dumps(out[0], ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
