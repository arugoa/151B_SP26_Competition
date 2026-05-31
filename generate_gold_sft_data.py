#!/usr/bin/env python3
"""
Generate competition-format gold-answer SFT data from data/public.jsonl.

This is the compute-cheap alternative to rejection-sampling SFT:
  prompt     = same strict-final competition prompt used at eval
  completion = short boxed gold answer, e.g. \boxed{A} or \boxed{ans1, ans2}

By default, it excludes the same 100-example stratified eval split used in
run_eval_ablation.py, so there is no train/eval leakage.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_DATA_PATH = "data/public.jsonl"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_stratified_eval_indices(data: List[Dict[str, Any]], eval_size: int, split_seed: int) -> set[int]:
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


def gold_to_boxed_completion(item: Dict[str, Any]) -> str:
    gold = item["answer"]
    if isinstance(gold, list):
        parts = [str(x).strip() for x in gold]
        ans = ", ".join(parts)
    else:
        ans = str(gold).strip()

    # If MCQ answer somehow comes as list, use same join, but most are single letters.
    return f"\\boxed{{{ans}}}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate gold-answer SFT JSONL for competition-format training.")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--split-seed", type=int, default=151)
    p.add_argument("--include-eval", action="store_true", help="Do NOT use this for honest validation; includes all public rows.")
    p.add_argument("--limit", type=int, default=None, help="Optional cap for quick smoke tests.")
    p.add_argument("--oversample-mcq", type=int, default=1, help="Repeat MCQ rows this many times.")
    p.add_argument("--oversample-multi", type=int, default=1, help="Repeat multi-answer free-form rows this many times.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_jsonl(args.data_path)
    eval_idxs = set() if args.include_eval else select_stratified_eval_indices(data, args.eval_size, args.split_seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for row_idx, item in enumerate(data):
        if row_idx in eval_idxs:
            continue
        is_mcq = bool(item.get("options"))
        gold = item["answer"]
        is_multi = isinstance(gold, list) and len(gold) > 1

        repeat = 1
        if is_mcq:
            repeat = max(repeat, args.oversample_mcq)
        if is_multi:
            repeat = max(repeat, args.oversample_multi)

        record = {
            "id": item.get("id"),
            "row_idx": row_idx,
            "is_mcq": is_mcq,
            "is_multi": is_multi,
            "gold": gold,
            "prompt": build_strict_prompt(item),
            "completion": gold_to_boxed_completion(item),
            "source": "gold_answer_sft",
        }
        for _ in range(repeat):
            rows.append(dict(record))

    if args.limit is not None:
        rows = rows[: args.limit]

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_mcq = sum(1 for r in rows if r["is_mcq"])
    n_multi = sum(1 for r in rows if r["is_multi"])
    print(json.dumps({
        "data_path": args.data_path,
        "output": str(out_path),
        "total_public_rows": len(data),
        "excluded_eval_rows": len(eval_idxs),
        "written_rows_after_oversampling": len(rows),
        "mcq_rows_after_oversampling": n_mcq,
        "multi_rows_after_oversampling": n_multi,
        "oversample_mcq": args.oversample_mcq,
        "oversample_multi": args.oversample_multi,
        "include_eval": args.include_eval,
    }, indent=2))
    print("Example row:")
    print(json.dumps(rows[0], ensure_ascii=False)[:1500] if rows else "<none>")


if __name__ == "__main__":
    main()
