import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=768, in_channels=3):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) **2

        self.projection = nn.Conv2d(
            in_channels,
            emb_dim,
            stride=patch_size,
            kernel=patch_size
        )
    
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
    
class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, num_patches):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        x = x + self.pos_emb
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=768, mlp_dim=3072, num_heads=12, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.mhs_attention = nn.MulitheadAttention(
            emb_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layer_norm1(x)
        attn_output = self.mhs_attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        return x 