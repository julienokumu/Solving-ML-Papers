import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float =0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        Q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn_output = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output
    
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float =0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        output = self.dropout(x)

        return output

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout=0.1):
        super(Block, self).__init__()
        self.attn = MultiHeadSelfAttention(
            d_model,
            n_heads,
            dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FFN(
            d_model,
            d_ff,
            dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        attn_output = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
    

class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 510, n_layers: int = 6, n_heads: int = 6, d_ff: int = 1020, max_seq_len: int = 512, dropout: float = 0.1):
        super(GPT, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            Block(
                d_model,
                n_heads,
                d_ff,
                dropout
            ) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, mask=None):
        batch, seq_len = input_ids.size()

        if mask is not None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        tok_emb = self.tok_emb(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)

        x = self.dropout(pos_emb + tok_emb)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.out_proj(x)

        return logits

    @staticmethod
    def get_tokenizer():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        return tokenizer
