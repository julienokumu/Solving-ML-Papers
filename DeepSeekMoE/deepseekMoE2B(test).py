import torch 
import torch.nn as nn
import torch.nn.functional as F
import dataclasses as dataclass
import math

@dataclass
class ModelArgs:
    vocab_size: int = 8000
    hidden_dim: int = 1280
    n_layers: int = 9
    n_heads: int = 10
    head_dim: int = 128
    n_experts: int = 64
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    expert_scale: float = 0.25
    ffn_intermed_scale: int = 4

args = ModelArgs()

class ExpertFFN(nn.Module):
    def __init__(self, hidden_dim, intermed_dim):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, intermed_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(intermed_dim, hidden_dim)
    
    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MoE(nn.Module):
    def __init__(self, hidden_dim, n_experts, n_shared_experts, n_activated_experts, expert_scale):
        super().__init__()
        standard_intermed_dim = int(hidden_dim * args.ffn_intermed_scale)
        expert_intermed_dim = int(standard_intermed_dim * expert_scale)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(
                hidden_dim,
                expert_intermed_dim
            ) for _ in range(n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            ExpertFFN(
                hidden_dim,
                expert_intermed_dim
            ) for _ in range(n_experts - n_shared_experts)
        ])
        self.router = nn.Linear(hidden_dim, n_experts - n_shared_experts)
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts - n_shared_experts
        self.n_routed_experts = n_experts - n_shared_experts
    
    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = expert(x) + shared_output
        logits = self.router(x)
        affinities = F.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(affinities, self.n_activated_experts, dim=-1)
        gates = torch.zeros_like(affinities).scatter_(-1, topk_indices, topk_values)
        routed_output = torch.zeros_like(x)
        for i in range(self.n_routed_experts):
            gate = gates[:, :, i].unsqueeze(-1)
            if gate.max() > 0:
                expert_output = self.routed_experts[i](x)
                routed_output = routed_output + gate * expert_output
        output = shared_output + routed_output + x
        return output

class MHA(nn.Module):
    def __init__(self, hidden_dim, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, n_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, n_heads * head_dim)
        self.out_proj = nn.Linear(n_heads * head_dim, hidden_dim)
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(context)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, head_dim, n_experts, n_shared_experts, n_activated_experts, expert_scale):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.mha = MHA(hidden_dim, n_heads, head_dim)
        self.layer_norm2 = nn.Linear(hidden_dim)
        self.moe = MoE(hidden_dim, n_experts, n_shared_experts, n_activated_experts, expert_scale)
    
    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape
        x_norm = self.layer_norm1(x)
        attn_output = self.mha(x_norm) + x
        attn_norm = self.layer_norm2(attn_output)
        output = self.moe(attn_norm) + attn_output
        return output

class DeepSeekMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.emb = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                args.hidden_dim,
                args.n_heads,
                args.head_dim,
                args.n_experts,
                args.n_shared_experts,
                args.n_activated_experts,
                args.expert_scale
            )
            for _ in range(args.n_layers)
        ])
        self.layer_norm = nn.LayerNorm(args.hidden_dim)
        self.out_proj = nn.Linear(args.hidden_dim, args.vocab_size)
    
    def forward(self, x):
        batch, seq_len = x.shape
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.out_proj(x)
        return logits


