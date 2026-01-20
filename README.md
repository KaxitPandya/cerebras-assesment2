NOTE: Used (allenai/OLMoE-1B-7B-0924), instead of (Qwen/Qwen1.5-MoE-A2.7B-Chat) due to resource constraint
Direct access of Code with logs on Google Colab (No Permission Required): https://colab.research.google.com/drive/1ba0p9iRQEmtomJ2gquANJoBORvqtktgI?usp=sharing

<img width="521" height="440" alt="image" src="https://github.com/user-attachments/assets/bfb50d1f-96c0-4042-896f-f81e39881ab9" />

# vLLM MoE Expert Logging

## Hook Location

**File**: `vllm_moe_patch.py`  
**Target**: `vllm.model_executor.layers.fused_moe.FusedMoE.forward()`  
**Hook Point**: Intercepts `router_logits`, computes `topk_ids`/`topk_weights` via softmax + topk, logs to JSONL, then calls original forward.

## Commands to Run

```bash
# 1. Generate prompts from GSM8K (25 questions)
python make_prompts.py

# 2. Run WITHOUT logging (baseline timing)
python run_generate.py

# 3. Run WITH logging enabled
VLLM_LOG_MOE=moe_routes.jsonl python run_generate.py

# 4. Generate histogram
python plot_expert_histogram.py moe_routes.jsonl expert_hist.png
```

## Analysis Results

**(a) Top-3 Most Used Experts (Layer 0):**
1. Expert #57: ~21% of selections
2. Expert #58: ~8% of selections  
3. Expert #8: ~7% of selections

**(b) Normalized Distribution:**  
With 64 experts, uniform = 1.56% each. Observed range: 0.5%-21%. Most experts receive 1-3%, with Expert #57 as a clear outlier.

**(c) Entropy: 5.1 bits** (max 6.0 bits for 64 experts), normalized: 85%.  
**Interpretation**: High entropy indicates good load balancing—the router distributes tokens broadly with mild specialization, avoiding over-reliance on few experts.

## Deliverables

| File | Description |
|------|-------------|
| `vllm_moe_patch.py` | FusedMoE monkey patch |
| `moe_logger.py` | Singleton logger module |
| `moe_routes.jsonl` | Expert routing log |
| `expert_hist.png` | Usage histogram |
| `timing.json` | No-log vs log timing |
| `plot_expert_histogram.py` | Plotting script |
