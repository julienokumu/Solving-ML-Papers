import torch
import torch.nn as nn
import math

# token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # embedding layer for tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # scaling
    
# segment embeddings
class SegmentEmbedding(nn.Module):
    def __init__(self, num_segments, d_model):
        super().__init__()
        # embedding layer for segments
        self.embedding = nn.Embedding(num_segments, d_model)

    def forward(self, x):
        return self.embedding(x)
    
# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=int):
        super().__init__()
        # zero tensor for pe
        pe = torch.zeros(max_len, d_model)
        # position indices (0 to max_len-1)
        position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)
        # scaling terms
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        # fill even indices with sine of scaled positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # add batch dim to positional encodings
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :] # add pe up to seq len

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads
        # linear layers for q, k, v and o
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        # project, reshape and transpose the q, k, v
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2)
        K = self. W_k(x).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2)
        V = self. W_v(x).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2)
        # scores with scaling
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_heads)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # attn weights
        attn_weights = torch.softmax(scores, dim=-1)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # apply output to combine attn heads
        output = self.W_o(context)
        return output
    
# feed forward network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ff(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

# full bert model   
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model,num_heads, num_layers, num_segments, d_ff, max_len=int, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.segment_embedding = SegmentEmbedding(num_segments, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        token_emb = self.token_embedding(input_ids)
        segment_emb = self.segment_embedding(segment_ids)
        x = token_emb + segment_emb
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.norm(x)
    

