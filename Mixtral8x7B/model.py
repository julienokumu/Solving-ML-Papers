import torch
import torch.nn as nn

mixtral_8x7b_config = {
    "emb_dim": 4096,
    "hidden_dim": 14336,
    "vocab_size": 32000,
    "context_length": 32768,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "window_size": 4000,
    "num_experts": 8,
    "top_k_experts": 2
}

