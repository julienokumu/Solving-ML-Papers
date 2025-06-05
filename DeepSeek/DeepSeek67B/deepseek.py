import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.gamma
        return x_norm

class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_heads, seq_len, head_dim = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(-1)
        angles = positions * self.theta
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)
        return rotated

class GQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        assert d_model == n_heads * self.head_dim, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, " n_heads must be divisible by n_kv_heads"
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model)
        self.rope = RoPE(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads * self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads * self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads * self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        repeat_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.out_proj(output)
        return output

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_intermed = int((8/9) * d_model)
        self.linear1 = nn.Linear(d_model, self.d_intermed)
        self.linear2 = nn.Linear(d_model, self.d_intermed)
        self.linear3 = nn.Linear(self.d_intermed, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        gate = F.silu(self.linear2(x))
        x = self.linear1(x)
        x = x * gate
        output = self.linear3(x)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.att_norm = RMSNorm(d_model)
        self.gqa = GQA(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        x = self.attn_norm(x)
        attn_out = self.gqa(x, mask)
        x = x + attn_out
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        output = x + ffn_out
        return output

class DeepSeek67B(nn.Module):
    def __init__(self, d_model: int = 8192, n_heads: int = 64, n_kv_heads: int = 8, context_length: int = 4096, n_layers: int = 95, vocab_size: int = 102400):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                n_kv_heads
            ) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len = x.shape
        x = self.embedding(x)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.output(x)
        return logits


