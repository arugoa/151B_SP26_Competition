#!/usr/bin/env python3
"""
Generate concise-solution SFT data for the CSE151B math competition.

This is a compute-feasible model-side alternative to rejection-sampling SFT.
It uses the public labels directly, but unlike answer-only SFT, it produces
short assistant completions with a small amount of task-aware explanation
before the final boxed answer.

Run from ~/151B_SP26_Competition.

Output rows:
  {
    "id": ...,
    "row_idx": ...,
    "is_mcq": ...,
    "is_multi": ...,
    "gold": ...,
    "prompt": ...,
    "completion": "... concise solution ... \\boxed{...}",
    "source": "concise_gold_solution_sft"
  }
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DATA_PATH = "data/public.jsonl"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def strict_prompt(item: Dict[str, Any]) -> str:
    question = item["question"]

    if item.get("options"):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_lines = [
            f"{letters[i]}. {option}"
            for i, option in enumerate(item["options"])
        ]
        return (
            "You are an expert mathematician. "
            "Solve the multiple-choice problem step by step, but do not over-explain. "
            "At the end, you must put only the final answer choice letter inside \\boxed{} and stop immediately after the boxed answer. "
            "For example, write \\boxed{A}, not the full answer text. "
            "Your final line must be exactly of the form \\boxed{A}, where A is one of the provided option letters.\n\n"
            f"Problem:\n{question}\n\n"
            "Options:\n"
            + "\n".join(option_lines)
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


def select_eval_indices(data: List[Dict[str, Any]], eval_size: int, split_seed: int) -> set[int]:
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
    return {int(x["_row_idx"]) for x in eval_data}


def normalize_gold_answer(ans: Any) -> List[str]:
    if isinstance(ans, list):
        return [str(x).strip() for x in ans]
    return [str(ans).strip()]


def boxed_answer(gold_list: List[str]) -> str:
    return ", ".join(gold_list)


def make_completion(item: Dict[str, Any], style: str = "concise") -> str:
    """
    We cannot derive a real proof from answer labels alone, so these are intentionally
    concise, answer-focused solutions. This is meant to teach finalization, exact
    answer style, multi-answer ordering, and MCQ letter format without overwriting
    the base model's reasoning too aggressively.
    """
    gold_list = normalize_gold_answer(item["answer"])
    final = boxed_answer(gold_list)
    is_mcq = bool(item.get("options"))
    is_multi = len(gold_list) > 1

    if is_mcq:
        letter = gold_list[0]
        if style == "minimal":
            return f"The correct answer choice is {letter}.\n\\boxed{{{letter}}}"
        return (
            "I compare the choices and select the option that satisfies the problem conditions. "
            f"The correct answer choice is {letter}.\n"
            f"\\boxed{{{letter}}}"
        )

    if is_multi:
        if style == "minimal":
            return f"The answers in order are {final}.\n\\boxed{{{final}}}"
        return (
            "The problem asks for multiple answers, so I keep the requested order of the blanks. "
            f"The values in order are {final}.\n"
            f"\\boxed{{{final}}}"
        )

    if style == "minimal":
        return f"The exact final answer is {final}.\n\\boxed{{{final}}}"

    return (
        "I keep the result in the requested exact form and avoid unnecessary decimal approximation. "
        f"The final value is {final}.\n"
        f"\\boxed{{{final}}}"
    )


def is_multi_answer(item: Dict[str, Any]) -> bool:
    ans = item.get("answer")
    return isinstance(ans, list) and len(ans) > 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate concise-solution competition SFT data.")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--split-seed", type=int, default=151)
    p.add_argument("--include-eval", action="store_true", help="Include held-out eval rows. Usually do NOT use this.")
    p.add_argument("--oversample-mcq", type=int, default=2)
    p.add_argument("--oversample-multi", type=int, default=2)
    p.add_argument("--style", choices=["concise", "minimal"], default="concise")
    p.add_argument("--seed", type=int, default=151)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_jsonl(args.data_path)

    eval_indices = set()
    if not args.include_eval:
        eval_indices = select_eval_indices(data, args.eval_size, args.split_seed)

    out_rows = []
    for row_idx, item in enumerate(data):
        if row_idx in eval_indices:
            continue

        base_row = {
            "id": item.get("id"),
            "row_idx": row_idx,
            "is_mcq": bool(item.get("options")),
            "is_multi": is_multi_answer(item),
            "gold": item.get("answer"),
            "prompt": strict_prompt(item),
            "completion": make_completion(item, style=args.style),
            "source": "concise_gold_solution_sft",
        }

        repeats = 1
        if base_row["is_mcq"]:
            repeats = max(repeats, args.oversample_mcq)
        if base_row["is_multi"]:
            repeats = max(repeats, args.oversample_multi)

        for _ in range(repeats):
            out_rows.append(dict(base_row))

    rng = random.Random(args.seed)
    rng.shuffle(out_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "data_path": args.data_path,
        "output": str(output_path),
        "total_public_rows": len(data),
        "excluded_eval_rows": len(eval_indices),
        "written_rows_after_oversampling": len(out_rows),
        "mcq_rows_after_oversampling": sum(r["is_mcq"] for r in out_rows),
        "multi_rows_after_oversampling": sum(r["is_multi"] for r in out_rows),
        "oversample_mcq": args.oversample_mcq,
        "oversample_multi": args.oversample_multi,
        "style": args.style,
        "include_eval": args.include_eval,
    }
    print(json.dumps(summary, indent=2))
    if out_rows:
        print("Example row:")
        print(json.dumps(out_rows[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
