import torch
import torch.nn as nn

mixtral_8x7b_config = {
    "vocab_size": 32000,
    "contex_length": 32768,
    "emb_dim": 4096,
    "hidden_dim": 14336,
    "window_size": 4000,
    "n_heads": 32,
    "n_layers": 32,
    "n_kv_heads": 8,
    "num_experts": 8,
    "top_k_experts": 2
}

class SWA(nn.Module):
    def __init__(self, emb_dim, n_heads, n_kv_heads, window_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.window_size = window_size
        self.head_dim = emb_dim // n_heads
        self.kv_head_dim = emb_dim // n_kv_heads

        self.q_proj = nn.Linear(emb_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, n_kv_heads * self.kv_head_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, n_kv_heads * self.kv_head_dim, bias=False)
        
        self.out_proj = nn.Linear(n_heads * self.head_dim, emb_dim, bias=False)
        self.scale = 1 / (self.head_dim ** 0.5)

    def forward(self, x, cache=None, position_idx=0):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.kv_head_dim).transpose(1, 2)

        if cache is not None:
            k_cache, v_cache = cache
            cache_idx = position_idx % self.window_size

            if cache_idx + L <= self.window_size:
                k_cache[:, :, cache_idx:cache_idx + L] = k
                v_cache[:, :, cache_idx:cache_idx + L] = v
            else:
                remaining = self.window_size - cache_idx
                k_cache[:, :, cache_idx:] = k[:, :, :remaining]
                v_cache[:, :, cache_idx:] = v[:, :, :remaining]
                k_cache[:, :, :L - remaining] = k[:, :, remaining:]
                v_cache[:, :, :L - remaining] = v[:, :, remaining:]

            cache_size = min(position_idx + 1, self.window_size)
            k = k_cache[:, :, :cache_size]
            v = v_cache[:, :, :cache_size]
        else:
            k = k[:, :, -self.window_size:]
            v = v[:, :, -self.window_size:]

        
        q_per_kv = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(q_per_kv, dim=1)
        v = v.repeat_interleave(q_per_kv, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        mask = torch.triu(torch.ones(L, k.shape[2], device=x.device), diagonal=1).bool()
        mask = mask | (torch.arange(k.shape[2], device=x.device).unsqueeze(0) <
                       torch.arange(L, device=x.device).unsqueeze(1) + position_idx - self.window_size)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        attn_weights = torch.sofmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.emb_dim)
        out = self.out_proj(out)
        return out, (k_cache, v_cache)
