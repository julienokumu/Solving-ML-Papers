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
        normalized_x = x / rms
        output = normalized_x * self.gamma
        assert x.shape == output.shape, f"input shape and output shape must match"
        print("\n===RMSNorm Test===")
        print(f"input shape: {x.shape}, output shape: {output.shape}")
        return output
    
class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_heads, seq_len, head_dim = x.shape
        assert head_dim == self.head_dim, "ensure head_dim matches"
        positions = torch.arange(seq_len, device=x.device).unsqueeze(-1)
        angles = positions * self.theta
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)
        assert x.shape == rotated.shape, "must match"
        print("\n===RoPE Test===")
        print(f"input shape: {x.shape}, output shape: {rotated.shape}")
        return rotated
    
class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RoPE(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.out_proj(output)
        assert x.shape == output.shape, "input shape must be same as output shape"
        print("\n===MHA Test===")
        print(f"input shape: {x.shape}, output shape: {output.shape}")
        return output
    
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_intermed = int((8/3 * d_model))
        self.linear1 = nn.Linear(d_model, self.d_intermed)
        self.linear2 = nn.Linear(d_model, self.d_intermed)
        self.linear3 = nn.Linear(self.d_intermed, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        gate = F.silu(self.linear2(x))
        x = self.linear1(x)
        x = x * gate
        output = self.linear3(x)
        
        print("\n===SwiGLUFFN Test===")
        print(f"input shape: {x.shape}, output shape: {output.shape}")
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mha = MHA(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        x = self.attn_norm(x)
        attn_output = self.mha(x, mask)
        x = x + attn_output
        x = self.ffn_norm(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        assert x.shape == x.shape, "shapes must match"
        print("\n===TransformerBlock Test===")
        print(f"input shape: {x.shape}, output shape: {x.shape}")
        return x

class DeepSeek7B(nn.Module):
    def __init__(self, d_model: int = 4096, n_heads: int = 32, n_layers: int = 32, context_length: int = 4096, vocab_size: int = 102400):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads
            ) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        x = self.embedding(x)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.output(x)
        print(f"\n===DeepSeek7B Test===")
        print(f"input shape: {x.shape}, output shape: {logits.shape}")
        return logits
    
try:
    print("starting model test....")
    model = DeepSeek7B(d_model=128, n_heads=8, n_layers=2, vocab_size=1000)
    print("model instantiated successfully")
    x = torch.randint(0, 1000, (2, 16))
    print(f"input shape: {x.shape}")
    device = torch.device("cpu")
    model = model.to(device)
    x = x.to(device)
    print(f"model and input moved to devie: {device}")
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"forward pass completed. Output shape: {logits.shape}")
except Exception as e:
    print(f"error occured: {str(e)}")
    import traceback
    traceback.print_exc()

