import torch
import torch.nn as nn
import math

# hyperparameters
d_model = 4096
n_heads = 32
d_head = d_model // n_heads
n_layers = 32
seq_len = 2048
vocab_size = 32000

# rotary embeddings
class RoPE(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        theta = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pos = torch.arange(seq_len, dype=torch.float).unsqueeze(1)
        self.register_buffer('sin', torch.sin(pos * theta).unsqueeze(2))
        self.register_buffer('cos', torch.cos(pos * theta).unsqueeze(2))

    def forward(self, x, seq_len):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rot_x_even = x_even * self.cos[:seq_len] - x_odd * self.sin[:seq_len]
        rot_x_odd = x_even * self.sin[:seq_len] - x_odd * self.sin[:seq_len]
        x_rot = torch.stack((rot_x_even, rot_x_odd), dim=-1).reshape(x.shape)
        return x_rot
    
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
    
# feed forward network with swiglu
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = int((2/3 * 4 * d_model))
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear_gate = nn.Linear(d_model, d_ff)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x_ff = self.linear1(x)
        gate = self.linear_gate(x)
        x_ff = x_ff * self.silu(gate)
        return self.linear2(x_ff)
    
# standard multihead attention with kv cache and casual masking
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.q_proj = nn.Linear(d_model, n_heads * d_head)
        self.k_proj = nn.Linear(d_model, n_heads * d_head)
        self.v_proj = nn.Linear(d_model, n_heads * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.rotary_emb = RoPE(d_head)
        self.register_buffer('k_cache', torch.zeros(1, seq_len, n_heads * d_head))
        self.register_buffer('v_cache', torch.zeros(1, seq_len, n_heads * d_head))

    def forward(self, x, step=None):
        batch_size, seq_len = x.size(0), x.size(1)
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        Q = self.rotary_emb(Q, seq_len)
        K = self.rotary_emb(K, seq_len)

        if step is not None:
            self.k_cache[:, step:step + seq_len] = K
            self.v_cache[:, step:step + seq_len] = V
            K = self.k_cache[:, :step + seq_len]
            V = self.v_cache[:, :step + seq_len]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if step is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(scores.device)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.sofmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_head)
        return self.out_proj(out)
    
# llama transformer layer
class LLaMALayer(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, d_head)
        self.rms_norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model)
    
    def forward(self, x, step=None):
        x_norm = self.rms_norm1(x)
        x = x + self.attn(x_norm, step)
        x_norm = self.rms_norm2(x)
        x = x + self.ff(x_norm)
        return x
    
# full llama model
class LLaMA(nn.Module):
    def __init__(self, d_model, n_heads, d_head, vocab_size, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LLaMALayer(
                d_model,
                n_heads,
                d_head
            ) for _ in range(n_layers)
        ])
        self.rms_norm = RMSNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, step=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, step)
        x = self.rms_norm(x)
        x = self.linear(x)
        return x
    