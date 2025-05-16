import torch
import torch.nn as nn

# config dictionary
config = {
    "vocab_size": 32000,
    "context_length": 32000,
    "emb_dim": 4096,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_layers": 32,
    "n_kv_heads": 8,
    "window_size": 4096,
    "num_experts": 8,
    "top_k_experts": 2
}

# sliding window attention
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

        # rolling buffer cache
        if cache is not None:
            k_cache, v_cache = cache
            cache_idx = position_idx % self.window_size

            if cache_idx + L <= self.window_size:
                k_cache[:, :, cache_idx:cache_idx + L] = k
                v_cache[:, :, cache_idx:cache_idx + L] = v
            else:
                remaining = self.window_size - cache_idx
                k_cache[:, :, cache_idx:] = k[:, :, :remaining]
                v_cache[:, :, cache_idx:] = k[:, :, :remaining]
                k_cache[:, :, :L - remaining] = k[:, :, remaining:]
                v_cache[:, :, :L - remaining] = v[:, :, remaining:]
            cache_size = min(position_idx + 1, self.window_size)
            k = k_cache[:, :, :cache_size]
            v = v_cache[:, :, :cache_size]
        else:
            k = k[:, :, -self.window_size:]
            v = v[:, :, -self.window_size:]

        # grouped query attention
        q_per_kv = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(q_per_kv, dim=1)
        v = v.repeat_interleave(q_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        mask = torch.triu(torch.ones(L, k.shape[2], device=x.device), diagonal=1).bool()
        mask = mask | (torch.arange(k.shape[2], device=x.device).unsqueeze(0) <
                       torch.arange(L, device=x.device).unsqueeze(1) + position_idx - self.window_size)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.emb_dim)
        attn_out = self.out_proj(attn_out)

        return attn_out, (k_cache, v_cache)
    
# mixture of experts layer    
class MoE(nn.module):
    def __init__(self, emb_dim, hidden_dim, num_experts, top_k_experts):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts

        self.router = nn.Linear(emb_dim, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, emb_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, L, _ = x.shape
        router_logits = self.router(x)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k_experts, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)

        output = torch.zeros_like(x)

        for i in range(self.top_k_experts):
            expert_idx = top_k_indices[:, :, i]
            expert_weights = top_k_weights[:, :, i].unsqueeze(-1)
            expert_output = torch.zeros_like(x)

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    selected_tokens = x[mask]
                    expert_out = self.experts[e](selected_tokens)
                    expert_output[mask] = expert_out
            output += expert_weights * expert_output

        return output
    
# transformer block
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_heads, n_kv_heads, num_experts, top_k_experts, window_size):
        super().__init__()
        self.swa = SWA(emb_dim, n_heads, n_kv_heads, window_size)
        self.moe = MoE(emb_dim, hidden_dim, num_experts, top_k_experts)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, cache=None, position_idx=0):
        attn_out, new_cache = self.swa(self.norm1(x), cache, position_idx)
        x = x + attn_out
        moe_out = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, new_cache
    
# mixral 8x7b model
class Mixtral8x7B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.layers = nn.ModuleList([
            TransformerBlock(
                emb_dim=config["emb_dim"],
                hidden_dim=config["hidden_dim"],
                n_heads=config["n_heads"],
                n_kv_heads=config["n_kv_heads"],
                window_size=config["window_size"],
                num_expers=config["num_experts"],
                top_k_experts=config["top_k_experts"]
            ) for _ in range(config["n_layers"])
        ])
        self.norm = nn.LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, input_ids, cache=None, position_idx=0):
        if cache is None:
            cache = [(torch.zeros(input_ids.shape[0], self.config["n_kv_heads"], self.config["window_size"], self.config["emb_dim"] // self.config["n_heads"]),
                      torch.zeros((input_ids.shape[0], self.config["n_kv_heads"], self.config["window_size"], self.config["emb_dim"] // self.config["n_heads"])))
                    for _ in range(self.config["n_layers"])]

        x = self.embed(input_ids)

        new_cache = []
        for layer, layer_cache in zip(self.layers, cache):
            x, layer_cache = layer(x, layer_cache, position_idx)
            new_cache.append(layer_cache)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, new_cache


        



