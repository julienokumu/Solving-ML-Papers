import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) **2

        self.projection = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
    
class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, emb_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        x = x + self.pos_emb
        return x