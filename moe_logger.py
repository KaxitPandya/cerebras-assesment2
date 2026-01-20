"""
MoE Expert Routing Logger for vLLM

This module provides a singleton logger that records MoE expert routing decisions
to a JSONL file when enabled via the VLLM_LOG_MOE environment variable.
"""

import os
import json
import torch
import vllm
from typing import Optional, List
from threading import Lock


class MoELogger:
    """Singleton logger for MoE expert routing."""
    
    _instance: Optional['MoELogger'] = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.log_path = os.environ.get('VLLM_LOG_MOE', None)
        self.enabled = self.log_path is not None
        self.file_handle = None
        self.header_written = False
        self.token_counter = 0
        self.request_counter = 0
        self.current_request_id = "r0"
        
        # Configuration - OLMoE-1B-7B has 64 experts, top_k=8
        self.layers_to_log = [0]  # Log only layer 0 by default
        self.top_k = 8  # OLMoE uses top_k=8
        self.num_experts = 64  # OLMoE has 64 experts
        
        if self.enabled:
            self._open_file()
            print(f"[MoE Logger] Initialized. Logging layer 0 to {self.log_path}")
    
    def _open_file(self):
        """Open log file and write header."""
        try:
            self.file_handle = open(self.log_path, 'w')
            self._write_header()
        except Exception as e:
            print(f"Warning: Could not open MoE log file: {e}")
            self.enabled = False
    
    def _write_header(self):
        """Write the metadata header line."""
        if self.header_written:
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
        
        header = {
            "type": "meta",
            "model_id": os.environ.get('VLLM_MODEL_ID', 'allenai/OLMoE-1B-7B-0924'),
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "device": device,
            "seed": 1234,
            "layers_logged": self.layers_to_log,
            "top_k": self.top_k,
            "num_experts": self.num_experts
        }
        
        self.file_handle.write(json.dumps(header) + '\n')
        self.file_handle.flush()
        self.header_written = True
    
    def log_routing(self, layer_idx: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """Log routing decision for a batch of tokens."""
        if not self.enabled or layer_idx not in self.layers_to_log:
            return
        
        try:
            # topk_ids shape: [num_tokens, top_k]
            # topk_weights shape: [num_tokens, top_k]
            ids = topk_ids.detach().cpu().tolist()
            weights = topk_weights.detach().cpu().tolist()
            
            for i, (token_ids, token_weights) in enumerate(zip(ids, weights)):
                record = {
                    "type": "route",
                    "req_id": self.current_request_id,
                    "token_idx": self.token_counter,
                    "layer": layer_idx,
                    "topk_ids": token_ids,
                    "topk_weights": [round(w, 4) for w in token_weights]
                }
                self.file_handle.write(json.dumps(record) + '\n')
                self.token_counter += 1
            
            self.file_handle.flush()
        except Exception as e:
            print(f"[MoE Logger] Error logging: {e}")
    
    def new_request(self):
        """Signal start of a new request."""
        self.request_counter += 1
        self.current_request_id = f"r{self.request_counter}"
    
    def close(self):
        """Close the log file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            print(f"[MoE Logger] Closed. Logged {self.token_counter} token routings.")


def get_moe_logger() -> MoELogger:
    """Get the singleton MoE logger instance."""
    return MoELogger()


def reset_moe_logger():
    """Reset the singleton for testing purposes."""
    MoELogger._instance = None
