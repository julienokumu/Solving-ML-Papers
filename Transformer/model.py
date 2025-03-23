import torch
import torch.nn as nn
import math

TRANSFORMER_CONFIG =  {
    "vocab_size": int,
    "context_length": int,
    "emb_dim": int,
    "n_heads": int,
    "n_layers": int,
    "qkv_bias": False,
    "drop_rate": int,
}

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        # extract from config
        emb_dim = config["emb_dim"]
        n_heads = config["n_heads"]
        drop_rate = config["drop_rate"]
        qkv_bias = config["qkv_bias"]
        
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        # store
        self.head_dim = emb_dim // n_heads
        self.n_head = n_heads
        self.emb_dim = emb_dim

        # linear layers, input -> q, k, v
        self.W_q = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        # linear layer, output after attention
        self.W_o = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        # get sequence length from first query tensor dim
        batch_size = query.size(0)

        # project and reshape q, k, v, split into heads
        query = self.W_q(query).view(batch_size, -1, self.n_heads, self.head_dim).tranpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # attn scores
        scores = torch.matmul(query, value.transpose(-2, -1)) / self.scale
        # mask to prevent future peeking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # weighted sum of values
        attn_ouput = torch.matmul(attn_weights, value)
        # reshape back to emb_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)
        # apply output proj -> head outputs = final attn results
        output = self.W_o(attn_output)
        return output

# feed forward network    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionWiseFeedForward, self).__init__()
        emb_dim = config["emb_dim"]
        ff_dim = emb_dim * 4
        drop_rate = config["drop_rate"]

        self.linear1 = nn.Linear(emb_dim, ff_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
        self.linear2 = nn.Linear(ff_dim, emb_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
# positional encoding(pe)
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        emb_dim = config["emb_dim"]
        max_len = config["context_length"]
        # tensor of zeros to store pe's
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # scaling factors for sin/cos waves
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000) / emb_dim))
        # fill even indices with sin values, odd indices with cos values
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add pe to x, match seq len
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x
    
# encoder blocks
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.norm2 = nn.LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x, mask=None):
        # apply self attention to q, k, v
        attn_output = self.self_attention(x, x, x, mask)
        # add attn_output to input, apply dropout and norm
        x = self.norm1(x + self.dropout(attn_output))
        # apply feed forward network to normalized output
        ff_output = self.feed_forward(x)
        # add ff_output to input, apply dropout and norm
        x = self.norm2(x + self. dropout(ff_output))
        return x
    
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        # list of n_layer encoder layers
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
        # for target sequence
        self.self_attention = MultiHeadAttention(config)
        # connect encoder output to decoder
        self.enc_dec_attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.norm2 = nn.LayerNorm(config["emb_dim"])
        self.norm3 = nn.LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # apply masked sel-attention to target sequence
        attn_output = self.self_attention(x, x, x, tgt_mask)
        # add input to attn_output, dropout and norm
        x = self.norm1(x + self.dropout(attn_output))
        # attn to decoder output using x as query
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
        
#  complete transformer
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.emb_dim = config["emb_dim"]

        # embedding layer to convert tokens to vectors
        self.embeddinig = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_encoding = PositionalEncoding(config)
        self.dropout = nn.Dropout(config["drop_rate"])
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # linear layer to predict token probabilities
        self.output_layer = nn.Linear(config["emb_dim"], config["vocab_size"])

    def generate_square_subsequent_mask(self, sz):
        # prevent future peeking
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # converts -inf to mask, 0 for unmasked
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self. emb_dim)
        # add pe to src emb
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        # pass src emb throught encoder to get encoded rep
        enc_output = self.encoder(src_emb, src_mask)

        # target emb tokens and scaling
        tgt_emb = self.embedding(tgt) * math.sqrt(self.emb_dim)
        # add pe
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        # mask
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        # pass target through decoder with encoder output
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        # project decoder output to vocab size for token probabilities
        output = self.output_layer(dec_output)
        return output
