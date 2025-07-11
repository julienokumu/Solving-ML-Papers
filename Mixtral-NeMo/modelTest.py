import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class ModelArgs:
    vocab_size: int = 256
    dim: int = 512
    head_dim: int = 64
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 1024
    n_layers: int = 2
    base: float = 1000.0
    eps: float = 1e-6

args = ModelArgs()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        output = x / rms * self.gamma
        print("====RMSNorm Test====")
        print(f"input shape: {x.shape}, output shape: {output.shape}")
        return output
    
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
        output = self.linear2(value * gate * self.beta)
        print("====SwiGLUFFN Test====")
        print(f"input shape: {x.shape}, output shape: {output.shape}")
        return output

class RoPE(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.dim = dim
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)
    
    def forward(self, x: torch.Tensor, seq_len=None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device, dtype=self.theta.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.theta)  # [seq_len, head_dim/2]
        cos_emb = freqs.cos()[None, :, None, :]  # [1, seq_len, 1, head_dim/2]
        sin_emb = freqs.sin()[None, :, None, :]  # [1, seq_len, 1, head_dim/2]
        print("====RoPE Test====")
        print(f"input shape: {x.shape}, output shape 1: {cos_emb.shape}, output shape 2: {sin_emb.shape}")
        return cos_emb, sin_emb

def apply_rotary_emb(x, cos_emb, sin_emb):
    x1, x2 = x[..., ::2], x[..., 1::2]  # [batch, seq_len, n_heads/n_kv_heads, head_dim/2]
    output = torch.zeros_like(x)
    output[..., ::2] = x1 * cos_emb - x2 * sin_emb  # cos_emb, sin_emb: [1, seq_len, 1, head_dim/2]
    output[..., 1::2] = x1 * sin_emb + x2 * cos_emb
    print("====Apply Rope Test====")
    print(f"output shape: {output.shape}")
    return output

class MHA(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rotary_emb = RoPE(head_dim, args.base)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        cos_emb, sin_emb = self.rotary_emb(q, seq_len)
        q = apply_rotary_emb(q, cos_emb, sin_emb)
        k = apply_rotary_emb(k, cos_emb, sin_emb)
        repeat_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        print("====MHA Test====")
        print(f"input shape: {x.shape}, output shape: {output.shape}")
        return self.out_proj(output)
    
class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.attn = MHA(dim, n_heads, n_kv_heads, head_dim)
        self.layernorm1 = RMSNorm(dim, args.eps)
        self.ffn = SwiGLUFFN(dim, hidden_dim)
        self.layernorm2 = RMSNorm(dim, args.eps)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch, seq_len, dim = x.shape
        x_norm = self.layernorm1(x)
        x = x + self.attn(x_norm, mask)
        out = self.layernorm2(x)
        output = x + self.ffn(out)
        print("====Block Test====")
        print(f"input shape: {x.shape}, output shape: {output.shape}")
        return output

class MixtralNeMo(nn.Module):
    def __init__(self, vocab_size: int, dim: int, hidden_dim: int, head_dim: int, n_heads: int, n_kv_heads: int, n_layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, head_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim, args.eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        x = self.emb(input_ids)
        if mask is None:
            seq_len = input_ids.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        print("====MixtralNeMo Test====")
        print(f"input shape: {x.shape}, output shape: {logits.shape}")
        return logits
    
model = MixtralNeMo(
    vocab_size=args.vocab_size,
    dim=args.dim,
    hidden_dim=args.hidden_dim,
    head_dim=args.head_dim,
    n_heads=args.n_heads,
    n_kv_heads=args.n_kv_heads,
    n_layers=args.n_layers
)

if __name__ == "__main__":
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")