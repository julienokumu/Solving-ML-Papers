import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class ModelArgs:
    vocab_size: int = 160
    dim: int = 320
    n_layers: int = 2
    n_heads: int = 5
    head_dim: int = 64
    n_experts: int = 32
    n_shared_experts: int = 1
    n_activated_experts: int = 4
    expert_scale: float = 0.25
    ffn_inter_dim: int = 2
    base: float = 10000.0
    eps: float = 1e-5
    drop_rate: float = 0.1

args = ModelArgs()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        norm_x = x / rms * self.gamma
        return norm_x
    
class RoPE(nn.Module):
    def __init__(self, head_dim: int, theta: torch.Tensor):
        super().__init__()
        self.head_dim = head_dim
        self.register_buffer("theta", theta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_heads, seq_len, head_dim = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(-1)
        angles = positions * self.theta
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        qk_rot = torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)
        return qk_rot

class ExpertFFN(nn.Module):
    def __init__(self, dim: int, inter_dim: int, drop_rate: float):
        super().__init__()
        self.layer1 = nn.Linear(dim, inter_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        self.layer2 = nn.Linear(inter_dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        output = self.layer2(x)
        return output
    
class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_shared_experts = args.n_shared_experts
        self.n_activated_experts = args.n_activated_experts - args.n_shared_experts
        self.n_routed_experts = args.n_experts - args.n_shared_experts
        standard_inter_dim = int(args.dim * args.ffn_inter_dim)
        expert_inter_dim = int(standard_inter_dim * args.expert_scale)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(
                args.dim,
                expert_inter_dim,
                args.drop_rate
            ) for _ in range(args.n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            ExpertFFN(
                args.dim,
                expert_inter_dim,
                args.drop_rate
            ) for _ in range(self.n_routed_experts)
        ])
        self.router = nn.Linear(args.dim, self.n_routed_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(x)
        logits = self.router(x)
        affinities = F.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(affinities, self.n_activated_experts, dim=-1)
        gates = torch.zeros_like(affinities).scatter_(-1, topk_indices, topk_values)
        routed_output = torch.zeros_like(x)
        for i in range(self.n_routed_experts):
            gate = gates[:, :, i].unsqueeze(-1)
            if gate.max() > 0:
                expert_output = self.routed_experts[i](x)
                routed_output += gate * expert_output
        output = shared_output + routed_output + x
        return output

class MHA(nn.Module):
    def __init__(self, args, theta: torch.Tensor):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.q_proj = nn.Linear(args.dim, args.n_heads * args.head_dim)
        self.k_proj = nn.Linear(args.dim, args.n_heads * args.head_dim)
        self.v_proj = nn.Linear(args.dim, args.n_heads * args.head_dim)
        self.out_proj = nn.Linear(args.n_heads * args.head_dim, args.dim)
        self.rope = RoPE(args.head_dim, theta)
        self.scale = 1.0 / math.sqrt(args.head_dim)
        self.dropout = nn.Dropout(args.drop_rate)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        mask = mask * float('-inf')
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(output)
        return output

class Block(nn.Module):
    def __init__(self, args, theta: torch.Tensor):
        super().__init__()
        self.layer_norm1 = RMSNorm(args.dim, args.eps)
        self.attn = MHA(args, theta)
        self.layer_norm2 = RMSNorm(args.dim, args.eps)
        self.moe = MoE(args)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        x = self.layer_norm1(x)
        attn_output = self.attn(x, positions) + x
        attn_norm = self.layer_norm2(attn_output)
        output = self.moe(attn_norm) + attn_output
        return output

class DeepSeekMoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args.vocab_size, args.dim)
        theta = 1.0 / (args.base ** (torch.arange(0, args.head_dim, 2).float() / args.head_dim))
        self.register_buffer("theta", theta)
        self.layers = nn.ModuleList([
            Block(
                args,
                theta
            ) for _ in range(args.n_layers)
        ])
        self.layer_norm = RMSNorm(args.dim, args.eps)
        self.out_proj = nn.Linear(args.dim, args.vocab_size)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len = x.shape
        if positions is None:
            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=x.device)
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.layer_norm(x)
        logits = self.out_proj(x)
        return logits

        
