import torch
import torch.nn as nn
import math

# hyperparameters
d_model = 512
n_heads = 8
d_head = d_model // n_heads
n_layers = 6
seq_len = 1024
vocab_size = 50257

# rotary embedding
class RoPE(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        theta = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        self.register_buffer('sin', torch.sin(pos * theta).unsqueeze(2))
        self.register_buffer('cos', torch.cos(pos * theta).unsqueeze(2))

    def forward(self, x, seq_len):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rot_x_even = x_even * self.cos[:seq_len] - x_odd * self.sin[:seq_len]
        rot_x_odd = x_even * self.sin[:seq_len] + x_odd * self.cos[:seq_len]
        return torch.cat((rot_x_even, rot_x_odd), dim=-1)
    
# rms normalization
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.scale * x_norm

# feed forward block wih swiglu    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x_ff = self.linear1(x)
        x_ff = x_ff * self.silu(x_ff)
        return self.linear2(x_ff)

# grouped query attention
class GQA(nn.Module):
    def __init__(self, d_model, n_head, d_head, n_groups=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_groups = n_groups

        self.q_proj = nn.Linear(d_model, n_heads * d_head)
        self.k_proj = nn.Linear(d_model, (n_heads // n_groups) * d_head)
        self.v_proj = nn.Linear(d_model, (n_heads // n_groups) * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)

        self.register_buffer('k_cache', torch.zeros(1, seq_len, (n_heads // n_groups) * d_head))
        self.register_buffer('v_cache', torch.zeros(1, seq_len, (n_heads // n_groups) * d_head))

    def forward(self, x, step=None):
        batch_size, seq_len = x.size(0), x.size(1)

        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads // self.n_groups, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads // self.n_groups, self.d_head).transpose(1, 2)

        if step is not None:
            self.k_cache[:, step] = K[:, -1]
            self.v_cache[:, step] = V[:, -1]
            K = self.k_cache[:, :step + 1]
            V = self.v_cache[:, :step + 1]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.sofmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_head)
        return self.out_proj(out)
    
# llama transformer layer
class LLaMALayer(nn.Module):
    def __init__(self, d_model, n_heads, d_head, n_groups=4):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model)
        self.attn = GQA(d_model, n_heads, d_head, n_groups)
        self.rms_norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, step=None):
        x_norm = self.rms_norm1(x)
        x = x + self.attn(x_norm, step)
        x_ff = self.rms_norm2(x)
        x = x + self.ff(x_norm)
        return x
    
# full LLaMA model
class LLaMA(nn.Module):
    def __init__(self, d_model, n_heads, d_head, n_layers, vocab_size, n_groups=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_emb = RoPE(d_model)
        self.layers = nn.ModuleLis([LLaMALayer(d_model, n_heads, d_head, n_groups) for _ in range(n_layers)])
        self.rms_norm = RMSNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, step=None):
        x = self.embedding(x)
        x = self.rotary_emb(x, x.size(1))
        for layer in self.layers:
            x = layer(x, step)
        x = self.rms_norm(x)
        x = self.linear(x)
        return x


