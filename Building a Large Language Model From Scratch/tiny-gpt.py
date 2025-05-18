import torch
import torch.nn as nn
import math

class TinyGPTConfig:
    def __init__(self):
        self.vocab_size = 40000
        self.n_layers = 12
        self.n_heads = 12
        self.emb_dim = 768
        self.hidden_dim = 3072
        self.max_seq_len = 512
        self.dropout = 0.1

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.emb_dim = config.emb_dim
        self.head_dim = config.emb_dim // config.n_heads
        self.q_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.k_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.v_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, L, emb_dim = x.size()

        Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_weights = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, emb_dim)

        output = self.out_proj(attn_weights)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.emb_dim, config.hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.hidden_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.emb_dim)
        self.ff = FeedForward(config)
        self.norm2 = nn.LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        attn_output = self.mhsa(x, mask)
        x = x + attn_output
        x = self.norm1(x)

        ffn_output = self.ff(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x
    
class TinyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.transformer_block = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.final_layernorm = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)

        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.size()
        assert L <= self.config.max_seq_len, "L must not be larger than max_seq_len"
        
        if attention_mask is not None:
            attention_mask = torch.tril(torch.ones(L, L).view(1, 1, L, L)).to(input_ids.device)

        tok_emb = self.tok_emb(input_ids)
        pos_ids = torch.arange(0, L, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_ids)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x, attention_mask)

        x = self.final_layernorm(x)
        logits = self.lm_head(x)

        return logits



