# CSE 151B SP26 — Math Reasoning Competition

## Overview

This repository contains our solution for the CSE 151B Spring 2026 Math Reasoning Competition. We fine-tuned **Qwen3-4B-Thinking** (INT8 quantized via vLLM) using a two-stage training pipeline:

1. **SFT (Supervised Fine-Tuning)** on the `open-r1/OpenR1-Math-220k` dataset (10k samples — 3k MCQ + 7k free-form)
2. **GRPO (Group Relative Policy Optimization)** on the competition public dataset using a math accuracy reward

At inference time, we use a two-pass generation strategy: a first-pass adapter generates initial responses, and a second-pass adapter refines them.

---

## Hardware & Timing

| Item | Details |
|---|---|
| **GPU** | NVIDIA A100 (40 GB) |
| **Approximate inference time** | ~30–45 minutes on the full private set |
| **Training time (SFT)** | ~4 hours |
| **Training time (GRPO)** | ~6 hours |

---

## Model Weights

Our fine-tuned LoRA adapters are hosted on HuggingFace Hub. The base model (`Qwen/Qwen3-4B-Thinking-2507`) is loaded directly from HuggingFace at runtime — no manual download required.

**LoRA adapters** (download and place as shown below):

```bash
# From the repo root, create the expected directory structure:
mkdir -p lora_adapters/lora_grpo
mkdir -p lora_adapters/lora_adapter_openr1_s1k

# Download via huggingface-cli:
hf download arihant06/lora_grpo \
    --local-dir ./lora_adapters/lora_grpo/

hf download arihant06/lora_adapter_openr1_s1k \
    --local-dir ./lora_adapters/lora_adapter_openr1_s1k/

hf download arihant06/lora_grpo_combined \
    --local-dir ./lora_adapters/
```

### Expected directory structure

```
151B_SP26_Competition/
├── data/
│   └── private.jsonl          # Private test set (provided by course staff)
├── lora_adapters/
│   ├── lora_grpo/
│   │   └── lora_grpo/         # GRPO adapter weights
│   └── lora_adapter_openr1_s1k/
│       └── lora_adapter_openr1_s1k/   # SFT adapter weights
├── judger.py
├── utils.py
├── run_inference.py
└── starter_code_cse151b_comp.ipynb
```

---

## Environment Setup

The environment requires cuda 13.0 to run, since this was trained using runpods. Please use the requirements.txt file for the environment, or the secondary command if the requirements file does not work.

Another common issue is different systems requiring python3.13 vs 3.11. This is due to various architectural design differences which we could not resolve.

```bash
# Install uv (fast package manager)
wget -qO- https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv --python python3.11 --seed --clear

# Install dependencies
.venv/bin/python -m pip install -r requirements.txt

# If the above command doesnt work, use the below one:
.venv/bin/python -m pip install \
    sympy numpy transformers vllm tqdm bitsandbytes \
    "antlr4-python3-runtime==4.11.1" peft trl pandas datasets "accelerate>=0.26.0"

# Activate
source .venv/bin/activate
```

---

## Running Inference

Call `run_inference()` from `run_inference.py` to reproduce our submission CSV end-to-end:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",   # path to the private test set
    output_path="results/submission.csv"
)
```

Or run directly from the command line:

```bash
python run_inference.py \
    --data_path data/private.jsonl \
    --output_path results/submission.csv
```

The function will:
1. Load the INT8-quantized base model via vLLM with LoRA support enabled
2. Load the QLoRA adapter for any MCQ and GRPO adapter for any FRQ
3. Run the forward pass for each question
4. Write `results/submission.csv` with columns `id` and `response`

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Thinking-2507` |
| Quantization | INT8 (BitsAndBytes via vLLM) |
| `max_tokens` | 32768 |
| `temperature` | 0.8 |
| `top_p` | 0.95 |
| `top_k` | 20 |
| `gpu_memory_utilization` | 0.90 |
| `max_model_len` | 32768 |
| SFT LoRA rank (`r`) | 32 |
| SFT `lora_alpha` | 64 |
| SFT training data | `open-r1/OpenR1-Math-220k` (10k samples) |
| GRPO `num_generations` | 2 |
| GRPO `max_completion_length` | 128 |
| GRPO `learning_rate` | 5e-6 |

---

## Reproducibility Note

Due to sampling stochasticity (`temperature=0.8`), outputs will not be string-identical across runs. Overall accuracy should remain consistent (within a few percentage points) with our leaderboard submission.
