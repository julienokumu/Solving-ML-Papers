import torch
import torch.nn as nn
import math

# token embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
# segment embedding
class SegmentEmbedding(nn.Module):
    def __init__(self, num_segments, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_segments, d_model)

    def forward(self, x):
        return self.embedding(x)
    
# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # zero tensor for positional encoding
        pe = torch.zeros(max_len, d_model)
        # position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # scaling terms for sin/cos
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        # fill even indices with sine of scaled positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # fill odd indices with cosine of scaled positions
        pe[:, 1::2] = torch.cos(position * div_term)
        # add batch dimension to pe
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads

        # linear layers for q, k, v
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size
        # project to q, k, v and reshape and transpose
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2)
        # attn scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_heads)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output
    
#  position wise-feed forward network
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
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
    
# bert model
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, num_segments, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embeddings = TokenEmbedding(vocab_size, d_model)
        self.segment_embeddings = SegmentEmbedding(num_segments, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, segment_ids):
        tok_emb = self.token_embeddings(input_ids)
        seg_emb = self.segment_embeddings(segment_ids)
        x = tok_emb + seg_emb
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    

