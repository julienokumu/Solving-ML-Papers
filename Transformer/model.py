import torch
import torch.nn as nn
import math 

TRANSFORMER_CONFIG =  {
    "vocab_size": int,
    "context_length": int,
    "emb_dim": int,
    "n_heads": int,
    "n_layers": int,
    "drop_rate": int,
    "qkv_bias": False
}

# mulit-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        emb_dim = config["emb_dim"]
        n_heads = config["n_heads"]
        drop_rate = config["drop_rate"]
        qkv_bias = config["qkv_bias"]

        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.head_dim = emb_dim // n_heads
        self.n_heads = n_heads
        self.emb_dim = emb_dim

        # linear layers to project input to q,k,v
        self.W_q = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        # output after combining heads
        self.w_o = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.scale = math.sqrt(self.head_dim) # prevent attn score explosions

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0) # number of sequences from query's first tensor dim
        # reshape q, k, v and split into heads
        query = self.W_q(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # attn scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        # mask to prevent future peeking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax for attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # weighted sum of values
        attn_output = torch.matmul(attn_weights, value)
        # reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, - 1, self.emb_dim)
        output = self.W_o(attn_output)
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionWiseFeedForward, self).__init__()
        emb_dim = config["emb_dim"]
        d_ff = emb_dim * 4 # inner layer
        drop_rate = config["drop_rate"]

        self.linear1 = nn.Linear(emb_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        emb_dim = config["emb_dim"]
        max_len = config["context_length"]

        # tensor to store pe
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # scaling factor for sin/cos waves
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000) / emb_dim))
        # fill even indices with sine values and odd indices with cosine values
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # add batch dimension
        self.register_buffer('pe', pe) # not a trainable parameter

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device) # add pe to x
        return x
    
# encoder block
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.norm2 = nn.LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["emb_dim"])

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask) # x's rep q,k,v
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["n_layers"])])
        self.norm = nn.LayerNorm(config["emb_dim"])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
# decoder block
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.enc_dec_attention = MultiHeadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.nomr2 = nn.LayerNorm(config["emb_dim"])
        self.norm3 = nn.LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.enc_dec_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config["n_layers"])])
        self.norm = nn.LayerNorm(config["emb_dim"])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
    
# transformer block
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.emb_dim = config["emb_dim"]

        # emb layer converting tokens to vectors
        self.embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_encoding = PositionalEncoding(config)
        self.dropout = nn.Dropout(config["drop_rate"])

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.output_layer = nn.Linear(config["emb_dim"], config["vocab_size"]) # token predictions

    def generate_square_subsequent_mask(self, sz): # prevent future peeking
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('inf')).masked_fill(mask == 1, float(0)) # converts to -inf for masked, 0 for unmasked
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.emb_dim)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        enc_output = self.encoder(src_emb, src_mask)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.emb_dim)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

        output = self.output_layer(dec_output)
        return output


