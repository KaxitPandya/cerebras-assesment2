"""
vLLM MoE Logging Patch for OLMoE

This patches the OLMoE model's SparseMoeBlock to log expert routing.
Import this BEFORE creating the LLM instance.
"""

import os
import torch
from moe_logger import get_moe_logger

if os.environ.get('VLLM_LOG_MOE'):
    print(f"[PATCH] MoE logging enabled, output: {os.environ.get('VLLM_LOG_MOE')}")
    
    # Patch FusedMoE.forward - this receives router_logits as parameter
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoE
        
        _original_forward = FusedMoE.forward
        _call_count = [0]
        _logged_count = [0]
        
        def patched_forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
            """Patched FusedMoE.forward that logs router_logits."""
            _call_count[0] += 1
            
            # Log every 100 calls to avoid spam
            if _call_count[0] <= 3:
                print(f"[PATCH] FusedMoE.forward called #{_call_count[0]}, hidden_states: {hidden_states.shape}, router_logits: {router_logits.shape}")
            
            # OLMoE has 16 layers, each call is one layer
            layer_idx = (_call_count[0] - 1) % 16
            
            logger = get_moe_logger()
            
            if logger.enabled and layer_idx in logger.layers_to_log:
                try:
                    # router_logits shape: [num_tokens, num_experts]
                    routing_weights = torch.softmax(router_logits.float(), dim=-1)
                    topk_weights, topk_ids = torch.topk(routing_weights, k=self.top_k, dim=-1)
                    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                    
                    logger.log_routing(layer_idx, topk_ids, topk_weights)
                    _logged_count[0] += 1
                    
                    if _logged_count[0] <= 3:
                        print(f"[PATCH] Logged {topk_ids.shape[0]} tokens at layer {layer_idx}")
                except Exception as e:
                    print(f"[PATCH] Error logging: {e}")
            
            return _original_forward(self, hidden_states, router_logits)
        
        FusedMoE.forward = patched_forward
        print(f"[PATCH] Applied to FusedMoE.forward, original: {_original_forward}")
        
    except Exception as e:
        print(f"[PATCH] Failed to patch FusedMoE: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[PATCH] MoE logging disabled (VLLM_LOG_MOE not set)")
