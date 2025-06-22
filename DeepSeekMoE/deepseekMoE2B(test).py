import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExpertFFN(nn.Module):
    def __init__(self, intermed_dim, hidden_dim=1280):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, intermed_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(intermed_dim, hidden_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

class MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts, num_shared_experts, num_activated_experts, expert_scale, ffn_intermed_scale):
        super().__init__()
        standard_intermed_dim = int(hidden_dim * ffn_intermed_scale)
        expert_intermed_dim = int(standard_intermed_dim * expert_scale)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(
                hidden_dim, 
                expert_intermed_dim
            ) for _ in range(num_shared_experts)
        ])
        self.router_experts = nn.ModuleList([
            ExpertFFN(
                hidden_dim,
                expert_intermed_dim
            ) for _ in range(num_experts - num_shared_experts)
        ])
        self.router = nn.Linear(hidden_dim, num_experts - num_shared_experts)
        self.num_shared_experts = num_shared_experts
        self.num_activated_experts = num_activated_experts - num_shared_experts
        self.num_routed_experts = num_experts - num_shared_experts
    
    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(x)
        logits = self.router(x)
        affinities = F.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(affinities, self.num_activated_experts, dim=-1)
        gates = torch.zeros_like(affinities).scatter_(-1, topk_indices, topk_values)
        routed_output = torch.zeros_like(x)
        for i in range(self.num_routed_experts):
            gate = gates[:, :, i].unsqueeze(-1)
            if gate.max() > 0:
                expert_output = self.routed_experts[i](x)
                routed_output += gate * expert_output
        output = shared_output + routed_output + x
        return output
    
class MHA(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(context)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_dim, num_experts, num_shared_experts, num_activated_experts, experts_scale):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attn = MHA(hidden_dim, num_heads, head_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.moe = MoELayer(hidden_dim, num_experts, num_shared_experts, num_activated_experts, experts_scale)
    
    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape
        attn_output = self.attn(self.layer_norm1(x)) + x
        output = self.moe(self.layer_norm2(attn_output)) + attn_output
        return output

class DeepSeekMoE(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, head_dim, num_experts, num_shared_experts, num_activated_experts, expert_scale):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, head_dim, num_experts, num_shared_experts, num_activated_experts, expert_scale)
            for _ in range(num_layers)
        ])
        self.layer_norm_final = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        batch, seq_len = x.shape
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm_final(x)
        logits = self.out_proj(x)
        return logits

