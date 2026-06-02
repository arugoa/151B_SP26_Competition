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
MAX_TOKENS = 32768

# LoRA adapter paths (relative to repo root)
LORA_GRPO_PATH    = "./lora_adapters/lora_grpo/lora_grpo/lora_grpo_v2"
LORA_SFT_PATH     = "./lora_adapters/lora_adapter_openr1_s1k/lora_adapter_openr1_s1k/lora_adapter_openr1_s1k_unquantized_2k"

# System prompts
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "Think freely and explore multiple approaches before committing to one. "
    "If you realize you made an error, correct it immediately and try a different approach. "
    "Always use Chinese for your reasoning and thought process, but your final answer must be in English. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Please always use Chinese for your reasoning and thought process, but your final answer has to be in English, and it needs to be clear and accurate. "
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
    output_path: str = "results/submission.jsonl",
) -> None:
    """
    Full end-to-end inference pipeline.

    1. Loads the BF-16 unquantized Qwen3-4B-Thinking base model via vLLM with LoRA enabled.
    2. Selects appropriate adapter for each question: SFT for MCQ and GRPO for FRQ on a single forward pass.
    3. Extracts final answers and writes submission CSV.

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
    import pandas as pd

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
    print("Loading model with vLLM (BF-16, LoRA enabled) ...")
    llm = LLM(
        model=MODEL_ID,
        #quantization="bitsandbytes",
        #load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.90,
        max_model_len=32768,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
        enable_lora=True,     # <-------------------- Enable/Disable LoRA support
        max_lora_rank=64,
    )
    print("Model loaded.")

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    # ── Build base prompts ────────────────────────────────────────────────────
    QLORA_ADAPTER = LoRARequest("qlora_sft", 1, LORA_SFT_PATH)
    GRPO_ADAPTER = LoRARequest( "grpo_rl", 2, LORA_GRPO_PATH)
    
    prompts = []
    requests = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
        if bool(item.get("options")):
            requests.append(QLORA_ADAPTER)
        else:
            requests.append(GRPO_ADAPTER)

    # ── Forward Pass ─────────────────────────────────────────────────────────
    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(
        prompts, 
        sampling_params=sampling_params,
        lora_request=requests,
    )
    responses = [out.outputs[0].text.strip() for out in outputs]

    # ── Post-process & extract answers ───────────────────────────────────────
    print("Post-processing answers ...")
    results = []
    for item, response in tqdm(zip(data, responses), total=len(data)):
        is_mcq = bool(item.get("options"))

        results.append({
            "id":       item.get("id"),
            "is_mcq":   is_mcq,
            "response": response,
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")

    df = pd.read_json(out_path, lines=True)
    df.to_csv('submission.csv', index=False)

    print(f"Saved {len(results)} records to {out_path}")


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
