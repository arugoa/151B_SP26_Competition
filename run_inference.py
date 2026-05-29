"""
run_inference.py
================
CSE 151B SP26 Math Reasoning Competition — single entry-point inference script.

Usage (Python):
    from run_inference import run_inference
    run_inference(data_path="data/private.jsonl", output_path="results/submission.csv")

Usage (CLI):
    python run_inference.py --data_path data/private.jsonl --output_path results/submission.csv
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID     = "0"                  # CUDA_VISIBLE_DEVICES
MAX_TOKENS = 4096

# LoRA adapter paths (relative to repo root)
LORA_GRPO_PATH    = "./lora_adapters/lora_grpo/lora_grpo"
LORA_SFT_PATH     = "./lora_adapters/lora_adapter_openr1_generate/lora_adapter_openr1_generate"

# System prompts
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Before you start to calculate, write down your reasoning and the steps you will take to solve the problem. "
    "Do not change your reasoning after you start calculating, unless there is a serious error. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_letter(text: str) -> str:
    """Extract the predicted answer letter from a \\boxed{} expression."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def extract_boxed_answer(text: str) -> str:
    """Extract the final answer from a \\boxed{} expression (free-form)."""
    matches = []
    for m in re.finditer(r"\\boxed\{", text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            matches.append(text[start : i - 1].strip())
    return matches[-1] if matches else text.strip()


def normalize_answer(response: str, is_mcq: bool) -> str:
    """Post-process a raw model response into a clean answer string."""
    if is_mcq:
        return extract_letter(response)
    return extract_boxed_answer(response)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_inference(
    data_path: str = "data/private.jsonl",
    output_path: str = "results/submission.csv",
) -> None:
    """
    Full end-to-end inference pipeline.

    1. Loads the INT8-quantized Qwen3-4B-Thinking base model via vLLM with LoRA enabled.
    2. First pass  — GRPO adapter generates initial reasoning + answer.
    3. Second pass — SFT adapter refines using the first-pass output as additional context.
    4. Extracts final answers and writes submission CSV.

    Parameters
    ----------
    data_path   : Path to the private test JSONL (no ground-truth answers).
    output_path : Path to write the submission CSV (columns: id, answer).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"

    # ── Lazy imports (heavy; only load when actually running) ──────────────────
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from tqdm import tqdm

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path} ...")
    data = [json.loads(line) for line in open(data_path)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"  {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"Loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model with vLLM (INT8, LoRA enabled) ...")
    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        max_model_len=6240,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
        enable_lora=True,
        max_lora_rank=64,
    )
    print("Model loaded.")

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    # ── Build base prompts ────────────────────────────────────────────────────
    base_prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        base_prompts.append(prompt_text)

    # ── Pass 1: GRPO adapter ──────────────────────────────────────────────────
    print(f"Pass 1 — GRPO adapter ({len(base_prompts)} prompts) ...")
    grpo_outputs = llm.generate(
        base_prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("grpo_adapter", 1, LORA_GRPO_PATH),
    )
    pass1_responses = [out.outputs[0].text.strip() for out in grpo_outputs]

    # ── Build second-pass prompts (base prompt + pass-1 response as context) ──
    pass2_prompts = []
    for base_prompt, pass1_resp in zip(base_prompts, pass1_responses):
        # Append the first-pass answer as an "assistant" turn so the model can
        # reason over it and potentially refine its answer.
        pass2_prompt = tokenizer.apply_chat_template(
            [
                {"role": "assistant", "content": pass1_resp},
                {
                    "role": "user",
                    "content": (
                        "Review your reasoning above. "
                        "If correct, confirm the answer. "
                        "If there is an error, correct it. "
                        "Put the final answer inside \\boxed{}."
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Prepend the original system+user prompt for full context
        pass2_prompts.append(base_prompt + pass2_prompt)

    # ── Pass 2: SFT adapter ───────────────────────────────────────────────────
    print(f"Pass 2 — SFT adapter ({len(pass2_prompts)} prompts) ...")
    sft_outputs = llm.generate(
        pass2_prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("sft_adapter", 2, LORA_SFT_PATH),
    )
    pass2_responses = [out.outputs[0].text.strip() for out in sft_outputs]

    # ── Post-process & extract answers ───────────────────────────────────────
    print("Post-processing answers ...")
    records = []
    for item, response in tqdm(zip(data, pass2_responses), total=len(data)):
        is_mcq = bool(item.get("options"))
        answer = normalize_answer(response, is_mcq)
        records.append({"id": item["id"], "answer": answer})

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("id,answer\n")
        for r in records:
            # Escape commas inside answers (e.g. "3, 7")
            answer_escaped = f'"{r["answer"]}"' if "," in str(r["answer"]) else r["answer"]
            f.write(f"{r['id']},{answer_escaped}\n")

    print(f"\nDone. {len(records)} answers written to {out_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSE 151B Competition — run_inference")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/private.jsonl",
        help="Path to the private test JSONL file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/submission.csv",
        help="Path to write the output submission CSV",
    )
    args = parser.parse_args()
    run_inference(data_path=args.data_path, output_path=args.output_path)