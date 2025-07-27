import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, 1e-9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output

    def split_heads(self, x):
        batch, seq_len, d_model = x.size()
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        batch, _, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.q_proj(Q))
        K = self.split_heads(self.k_proj(K))
        V = self.split_heads(self.v_proj(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.out_proj(self.combine_heads(attn_output))

        return output


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, inter_dim):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, inter_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(inter_dim, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        theta = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, ::2] = torch.sin(position * theta)
        pe[:, 1::2] = torch.cos(position * theta)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, inter_dim, dropout):
        super(Encoder, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, inter_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(self.dropout(ffn_output))

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, inter_dim, dropout):
        super(Decoder, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, inter_dim)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm3(self.dropout(ffn_output))

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, inter_dim, max_len, dropout):
        super(Transformer, self).__init__()
        self.encoder_emb = nn.Embedding(src_vocab_size, d_model)
        self.decoder_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layer = nn.ModuleList([
            Encoder(
                d_model,
                n_heads,
                inter_dim,
                dropout
            ) for _ in range(n_layers)
        ])

        self.decoder_layer = nn.ModuleList([
            Decoder(
                d_model,
                n_heads,
                inter_dim,
                dropout
            ) for _ in range(n_layers)
        ])

        self.lm_head = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_emb = self.dropout(self.positional_encoding(self.encoder_emb(src)))
        tgt_emb = self.dropout(self.positional_encoding(self.decoder_emb(tgt)))

        enc_output = src_emb
        for enc_layer in self.encoder_layer:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_emb
        for dec_layer in self.decoder_layer:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.lm_head(dec_output)
        
        return output


