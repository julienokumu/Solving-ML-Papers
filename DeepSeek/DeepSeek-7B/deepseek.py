import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Paramter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normalized_x = x / rms
        return normalized_x * self.gamma

class RoPE(nn.Module):
    def __init__(self, head_dim: int, context_length: int = 4096, base: float = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.context_length = context_length
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor, context_length: int = 4096) -> torch.Tensor:
        positions = torch.arange(context_length, device=x.device).unsqueeze(-1)
        angles = positions * self.theta
        cos = torch.cos(angles).repeat(1, 2)
        sin = torch.sin(angles).repeat(1, 2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.usqueeze(0).unsqueeze(2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)
        return rotated
    
class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model is divisible by n_heads"
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, context_length, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q, context_length)
        k = self.rope(k, context_length)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2)
        output = output.contiguous().view(batch, context_length, d_model)
        output = self.out_proj(output)
        return output

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_intermed = int((8/3) * d_model)
        self.linear1 = nn.Linear(d_model, self.d_intermed)
        self.linear2 = nn.Linear(d_model, self.d_intermed)
        self.linear3 = nn.Linear(self.d_intermed, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.linear2(x))
        x = self.linear1(x)
        x = x * gate
        output = self.linear3(x)
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mha = MHA(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.attn_norm(x)
        attn_output = self.mha(x, mask)
        x = x + attn_output
        x = self.ffn_norm(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        return x

class DeepSeek7B(nn.Module):
    def __init__(self, d_model: int = 4096, n_heads: int = 32, n_layers: int = 32, context_length: int = 4096, vocab_size: int = 102400):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RoPE(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads
            ) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, context_length = x.shape
        x = self.embedding(x)
        mask = torch.tril(torch.ones(context_length, context_length, device=x.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.output(x)
        return logits
    


