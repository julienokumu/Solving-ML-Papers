import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    def __init__(self, emb_dim, n_heads, head_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(emb_dim, n_heads * head_dim)
        self.k_proj = nn.Linear(emb_dim, n_heads * head_dim)
        self.v_proj = nn.Linear(emb_dim, n_heads * head_dim)
        self.out_proj = nn.Linear(n_heads * head_dim, emb_dim)
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(context)
        return output

class ExpertFFN(nn.Module):
    def __init__(self, emb_dim, intermed_dim):
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, intermed_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(intermed_dim, emb_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MoELayer(nn.Module):
    def __init__(self, emb_dim, n_experts, n_shared_experts, n_activated_experts, expert_scale):
        super().__init__()
        standard_intermed_dim = int(emb_dim * 4)
        expert_intermed_dim = int(standard_intermed_dim * expert_scale)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(
                emb_dim,
                expert_intermed_dim
            ) for _ in range(n_shared_experts)
        ])
        self.router = nn.Linear(emb_dim, n_experts - n_shared_experts)
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts - n_shared_experts
        self.n_routed_experts = n_experts - n_shared_experts

    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
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
    
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, head_dim, n_experts, n_shared_experts, n_activated_experts, expert_scale):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.mha = MHA(emb_dim, n_heads, head_dim)
        self.layer_norm2 = nn.Linear(emb_dim)
        self.moe = MoELayer(emb_dim, n_experts, n_shared_experts, n_activated_experts, expert_scale)
    
    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
        x_norm = self.layer_norm1(x)
        attn_output = self.mha(x_norm) + x
        attn_norm = self.layer_norm2(attn_output)
        output = self.moe(attn_norm) + attn_output
        return output

class DeepSeekMoE(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_layers, n_heads, head_dim, n_experts, n_shared_experts, n_activated_experts, expert_scale):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                emb_dim,
                n_heads,
                head_dim,
                n_experts,
                n_shared_experts,
                n_activated_experts, expert_scale
            ) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.out_proj = nn.Linear(emb_dim, vocab_size)
    
    def forward(self, x):
        batch, seq_len = x.shape
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        self.layer_norm(x)
        logits = self.out_proj(x)
        return logits
    