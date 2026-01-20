"""
run_generate.py - Generate text using vLLM with MoE expert logging support

Usage:
  python run_generate.py                          # Without logging
  VLLM_LOG_MOE=moe_routes.jsonl python run_generate.py  # With logging
"""

import os
import json
import time
import random
import gc
import torch

# Clear any existing GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Apply MoE logging patch BEFORE importing vLLM
if os.environ.get('VLLM_LOG_MOE'):
    import vllm_moe_patch

from vllm import LLM, SamplingParams

# OLMoE-1B-7B: A tiny MoE model that fits in T4!
# - Only 7B total parameters (1B active per token)
# - 64 experts, top_k=8
# - ~14GB in float16, fits in T4's 15GB
MODEL_ID = "allenai/OLMoE-1B-7B-0924"

# Set model ID for logging
os.environ['VLLM_MODEL_ID'] = MODEL_ID

# Set seed for reproducibility
random.seed(1234)

# Load prompts
print("Loading prompts...")
prompts = open("prompts.txt", encoding="utf-8").read().split("\n\n---\n\n")
print(f"Loaded {len(prompts)} prompts")

# Create sampling parameters
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=1234)

# Initialize LLM
print(f"Initializing vLLM with {MODEL_ID}...")
llm = LLM(
    model=MODEL_ID,
    max_model_len=512,  # Small context to save memory
    trust_remote_code=True,
    gpu_memory_utilization=0.98, # Increased to 0.98
    enforce_eager=True,  # Disable CUDA graphs
    dtype="half", # Changed to half for T4 compatibility
)

# Generate
print("Generating...")
t0 = time.time()
outs = llm.generate(prompts, sp)
t1 = time.time()

elapsed = t1 - t0
total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

print(f"\nGeneration complete!")
print(f"Time: {elapsed:.2f}s")
print(f"Tokens: {total_tokens}")
print(f"Tokens/sec: {total_tokens/elapsed:.2f}")

# Save timing results
timing_file = "timing.json"
if os.path.exists(timing_file):
    with open(timing_file, 'r') as f:
        timing_data = json.load(f)
else:
    timing_data = {}

# Determine if logging was enabled
log_key = "log" if os.environ.get('VLLM_LOG_MOE') else "no_log"
timing_data[log_key] = {
    "wall_time_sec": elapsed,
    "tokens_generated": total_tokens,
    "tokens_per_sec": total_tokens / elapsed
}

with open(timing_file, 'w') as f:
    json.dump(timing_data, f, indent=2)

print(f"Timing data saved to {timing_file}")

# Close logger if enabled
if os.environ.get('VLLM_LOG_MOE'):
    from moe_logger import get_moe_logger
    get_moe_logger().close()
    print(f"MoE routing log saved to {os.environ.get('VLLM_LOG_MOE')}")

# Print sample outputs
print("\n=== Sample Outputs ===")
for i, out in enumerate(outs[:3]):
    print(f"\nPrompt {i+1}: {prompts[i][:80]}...")
    print(f"Output: {out.outputs[0].text[:150]}...")
