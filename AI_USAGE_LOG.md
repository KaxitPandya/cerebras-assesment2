# AI Usage Log

## Tools Used

**GitHub Copilot (Claude Sonnet 4)** - Primary coding assistant

### What AI Helped With:
- Identified hook point in vLLM's `FusedMoE.forward()` method
- Generated singleton logger pattern (`moe_logger.py`)
- Created monkey patch for routing interception (`vllm_moe_patch.py`)
- Built histogram visualization with entropy analysis (`plot_expert_histogram.py`)
- Suggested OLMoE-1B-7B as Colab-compatible alternative (Qwen1.5-MoE requires 28GB VRAM)

## Verification Methods

1. **Code Review**: Manually inspected all generated code for correctness and vLLM API compatibility

2. **Functional Testing**: 
   - Ran notebook cells sequentially
   - Confirmed patch applies without import errors
   - Verified logging activates only when `VLLM_LOG_MOE` is set

3. **Output Validation**:
   - Checked `moe_routes.jsonl` header matches required schema
   - Verified route records contain `topk_ids` and `topk_weights`
   - Confirmed `timing.json` has both `no_log` and `log` entries

4. **Statistical Verification**:
   - Validated entropy calculation: H = -Σ p·log₂(p)
   - Confirmed top-3 experts match histogram visual
   - Cross-checked expert counts sum to total_tokens × top_k

## Code Attribution

| Component | AI-Generated | Human-Verified |
|-----------|--------------|----------------|
| Logger singleton | ✓ | ✓ |
| FusedMoE patch | ✓ | ✓ |
| Entropy analysis | ✓ | ✓ |
| Histogram plot | ✓ | ✓ |
