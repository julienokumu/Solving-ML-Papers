import torch
import torch.nn as nn
import math

# Updated Rotary Positional Embedding (RoPE)
class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000):
        super().__init__()
        # head_dim: dimension of each head (d_k)
        self.head_dim = head_dim
        # base: base frequency for RoPE
        self.base = base
        # Compute theta values: 1 / (base^(2i/head_dim))
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: input tensor [batch_size, num_heads, seq_len, head_dim]
        batch, num_heads, seq_len, head_dim = x.shape
        # Compute position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(-1)  # [seq_len, 1]
        # Compute angles: position * theta
        angles = positions * self.theta  # [seq_len, head_dim/2]
        # Compute cos and sin for rotation
        cos = torch.cos(angles)  # [seq_len, head_dim/2]
        sin = torch.sin(angles)  # [seq_len, head_dim/2]
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim/2]
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim/2]
        # Split input into even and odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]  # [batch_size, num_heads, seq_len, head_dim/2]
        # Apply rotation: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        rotated = torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)  # [batch_size, num_heads, seq_len, head_dim]
        return rotated

    def shape_check(self, x, output):
        print(f"RoPE - Input shape: {x.shape}, Output shape: {output.shape}")

# RMSNorm Layer
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.gamma
        return x_norm

    def shape_check(self, x, output):
        print(f"RMSNorm - Input shape: {x.shape}, Output shape: {output.shape}")

# SwiGLU Feed-Forward Network
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_int):
        super().__init__()
        self.d_model = d_model
        self.d_int = d_int
        self.w1 = nn.Linear(d_model, d_int, bias=True)
        self.w2 = nn.Linear(d_model, d_int, bias=True)
        self.w3 = nn.Linear(d_int, d_model, bias=True)
        self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        gate = self.swish(self.w1(x))
        x2 = self.w2(x)
        hidden = gate * x2
        output = self.w3(hidden)
        return output

    def shape_check(self, x, output):
        print(f"SwiGLU - Input shape: {x.shape}, Output shape: {output.shape}")

# Grouped-Query Attention (GQA) with Updated RoPE
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)
        # Use updated RoPE class
        self.rotary = RoPE(head_dim)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        # Compute queries, keys, values
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)  # [batch_size, seq_len, num_kv_heads, head_dim]
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)  # [batch_size, seq_len, num_kv_heads, head_dim]
        # Transpose for RoPE: [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]
        # Apply RoPE to queries and keys
        q = self.rotary(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self.rotary(k)  # [batch_size, num_kv_heads, seq_len, head_dim]
        # Transpose back: [batch_size, seq_len, num_heads, head_dim]
        q = q.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        k = k.transpose(1, 2)  # [batch_size, seq_len, num_kv_heads, head_dim]
        # Repeat keys and values to match num_heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)  # [batch_size, seq_len, num_heads, head_dim]
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)  # [batch_size, seq_len, num_heads, head_dim]
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # Softmax to get attention weights
        attn = torch.softmax(scores, dim=-1)
        # Compute attention output
        context = torch.matmul(attn, v)  # [batch_size, seq_len, num_heads, head_dim]
        # Reshape and project
        context = context.reshape(batch, seq_len, -1)  # [batch_size, seq_len, num_heads * head_dim]
        output = self.o_proj(context)  # [batch_size, seq_len, d_model]
        return output

    def shape_check(self, x, output):
        print(f"GQA - Input shape: {x.shape}, Output shape: {output.shape}")

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim, d_int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, head_dim)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_int)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        output = x + ffn_out
        return output

    def shape_check(self, x, output):
        print(f"TransformerBlock - Input shape: {x.shape}, Output shape: {output.shape}")

# DeepSeek LLM 67B Model
class DeepSeekLLM(nn.Module):
    def __init__(self, vocab_size=99, d_model=82, num_layers=9, num_heads=8, num_kv_heads=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads  # 128
        self.d_int = int((8/9) * d_model)  # ~7282
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, num_kv_heads, self.head_dim, self.d_int)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)  # [batch_size, seq_len, d_model]
        for block in self.blocks:
            x = block(x)  # [batch_size, seq_len, d_model]
        x = self.norm(x)  # [batch_size, seq_len, d_model]
        output = self.out_proj(x)  # [batch_size, seq_len, vocab_size]
        return output

    def shape_check(self, x, output):
        print(f"DeepSeekLLM - Input shape: {x.shape}, Output shape: {output.shape}")

# Test the model with a sample input
batch, seq_len = 2, 100
vocab_size = 70
model = DeepSeekLLM()
sample_input = torch.randint(0, vocab_size, (batch, seq_len))
output = model(sample_input)
model.shape_check(sample_input, output)