import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, emb_dim=768, in_channels=3, patch_size=16, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) **2

        self.projection = nn.Conv2d(
            emb_dim,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size
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
        self.mhs_attention = nn.MultiheadAttention(
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
        x = self.layer_norm2(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, emb_dim=768, mlp_dim=3072, in_channels=3, num_heads=12, depth=12, num_classes=1000, dropout=0.1, patch_size=16, img_size=224):
        super(VisionTransformer, self).__init__()
        self.num_patches = (img_size // patch_size) **2
        self.patch_emb = PatchEmbedding(
            emb_dim,
            in_channels,
            patch_size,
            img_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_emb = PositionalEmbedding(
            self.num_patches,
            emb_dim
        )
        self.pos_drop = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(
                emb_dim,
                num_heads,
                mlp_dim,
                dropout
            )
            for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.class_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        batch_size = x.view(0)
        x = self.patch_emb(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_emb(x)
        x = self.pos_drop(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.layer_norm(x)
        cls_output = x[:, 0]
        logits = self.class_head(cls_output)
        return logits