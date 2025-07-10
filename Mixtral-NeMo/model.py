import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class ModelArgs():
    vocab_size: int = 2**17
    dim: int = 5120
    head_dim: int = 128
    hidden_dim: int = 14436
    n_layers: int = 40
    n_heads: int = 32
    n_kv_heads: int = 8
    base: float = 1000000.0
    eps: float = 1e-6

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        return self.gamma * x / rms

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        gate = F.silu(self.gate(x))
        value = self.linear1(x)

        return self.linear2(value * gate * self.beta)
    
class RoPE(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.dim = dim
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        positions = torch.arange(seq_len, device=x.device, dtype=self.theta.dtype)
        freqs = torch.einsum("i, j->ij", positions, self.theta)

        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        cos_emb = emb.cos()[None, :, None, :]
        sin_emb = emb.sin()[None, :, None, :]

        return cos_emb, sin_emb
    
def apply_rope(x : torch.Tensor, cos_emb, sin_emb) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    output = torch.zeros_like(x)
    output[..., ::2] = x1 * cos_emb - x2 * sin_emb
    output[..., 1::2] = x1 * sin_emb + x2 * cos_emb

    return output

class MHA(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dim = dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim)
        self.out_proj = nn.Linear(n_heads * head_dim, dim)

        self.rope = RoPE(head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        cos_emb, sin_emb = self.rope(q, seq_len)
        q = apply_rope(q, cos_emb, sin_emb)
        k = apply_rope(k, cos_emb, sin_emb)

        repeat_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.out_proj(output)
    
class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.attn = MHA(
            dim,
            n_heads,
            n_kv_heads,
            head_dim
        )
        self.laynorm1 = RMSNorm(dim)

        self.ffn = SwiGLUFFN(
            dim,
            hidden_dim
        )
        self.laynorm2 = RMSNorm(dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        x_norm = self.laynorm1(x)
        x = x + self.attn(x_norm, mask)
        output = self.laynorm2(x)
        x = x + self.ff(output)

        return x

class MixtralNeMo(nn.Module):
    def __init__(self, n_layers: int, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            Block(
                dim,
                n_heads,
                n_kv_heads,
                head_dim, 
                hidden_dim
            ) for _ in range(n_layers)
        ])
        self.layernorm = RMSNorm(dim)
        self.logits = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len = input_ids.shape

        x = self.emb(input_ids)

        if mask is not None:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.logits(x)

        return logits


