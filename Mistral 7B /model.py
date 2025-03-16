import torch
import torch.nn as nn

MISTRAL_7B_CONFIG = {
    "vocab_size": int,
    "context_length": int,
    "emb_dim": int,
    "n_heads": int,
    "n_layers": int,
    "n_kv_heads": int,
    "window_size": int,
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

        # linear layers, input -> qkv
        self.q_proj = nn.Linear(emb_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, n_kv_heads * self.kv_head_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, n_kv_heads * self.kv_head_dim, bias=False)
        # linear layer, attn output -> emb
        self.o_proj = nn.Linear(n_heads * self.head_dim, emb_dim, bias=False)
        # scale to prevent large values
        self.scale = 1 / (self.head_dim ** 0.5)

    def forward(self, x, cache=None):
        B, L, _ = x.shape
        # input -> queries and reshape to [B, L, n_heads, head_dim], then transpose to [B, n_heads, L, head_dim]
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.kv_head_dim).transpose(1, 2)

        # cache, concatenate with prev k/v
        if cache is not None:
            k_cache, v_cache = cache
            # append new k/v, keep only last window_size tokens
            k = torch.cat([k_cache, k], dim=2)[:, :, -self.window_size:]
            v = torch.cat([v_cache, v], dim=2)[:, :, -self.window_size:]
        else:
            # if no cache trim k/v to last window_size tokens
            k = k[:, :, -self.window_size:]
            v = v[:, :, -self.window_size:]

        # query heads per k/v
        q_per_kv = self.n_heads // self.n_kv_heads
        # repeat key/value heads to match number of query heads
        k = k.repeat_interleave(q_per_kv, dim=1)
        v = v.repeat_interleave(q_per_kv, dim=1)

        # attn scores
        scores = torch.matmul(q, k.tranpose(-2, -1)) * self.scale

        # mask to prevent future peeking
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # extend mask for swa
        mask = mask | (torch.arange(L, device=x.device).unsqueeze(0) <
                       torch.arange(L, device=x.device).unsqueeze(1) - self.window_size)
        
        # mask to scores, setting masked positions to -inf
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        # softmax to get attn weights
        attn = torch.softmax(scores, dim=-1)
        # weighted sum of values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.emb_dim)
        # back to emb
        out = self.o_proj(out)

        return out, (k, v)
    
# transformer layer
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, n_kv_heads, window_size, ffn_dim=None):
        super().__init__()
        self.attn = SWA(emb_dim, n_heads, n_kv_heads, window_size)
        # ffn intermediate size
        ffn_dim = ffn_dim if ffn_dim is not None else 4 * emb_dim
        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, emb_dim)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), cache)
        x = x + attn_out # residual connection
        x = x + self.ffn(self.norm2(x))
        return x, new_cache
    
# mistral model
class Mistral7B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # emb layer converting token ids to dense vectors
        self.embed = nn.Embedding(config["vocab_size"], config["emb_dim"])
        # trasnformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                emb_dim=config["emb_dim"],
                n_heads=config["n_heads"],
                n_kv_heads=config["n_kv_heads"],
                window_size=config["window_size"]
            ) for _ in range(config["n_layers"])
        ])
        self.norm = nn.LayerNorm(config["emb_dim"])
        # back to vocab_size for logits
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, input_ids, cache=None):
        # if no cache intialize an empty cache list for each layer
        if cache is None:
            cache = [None] * self.config["n_layers"]
        # convert input ids to emb
        x = self.embed(input_ids)
        new_cache = [] # list to store updated caches for each layer

        # iterate over layers and their corresponding caches
        for layer, layer_cache in zip(self.layers, cache):
            x, layer_cache = layer(x, layer_cache) # forward pass through layer, update x and cache
            new_cache.append(layer_cache) # store updated cache for layer

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_cache
    

