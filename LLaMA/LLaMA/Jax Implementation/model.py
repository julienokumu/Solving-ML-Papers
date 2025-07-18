import jax
import jax.numpy as jnp
import flax.linen as nn
import math

d_model: int = 4096
n_heads: int = 32
d_head: int = d_model // n_heads
n_layers: int = 32
seq_len: int = 2048
vocab_size: int = 32000
base: float = 10000 

class RoPE(nn.Module):
    d_head: int
    seq_len: int

    def setup(self):
        theta = 1.0 / (base ** (jnp.arange(0, self.d_head, 2).astype(jnp.float32) / self.d_head))
        pos = jnp.arange(self.seq_len, dtype=jnp.float32).reshape(-1, 1)
        self.sin = jnp.sin(pos * theta).reshape(self.seq_len, -1, 1)
        self.cos = jnp.cos(pos * theta).reshape(self.seq_len, -1, 1)

    def __call__(self, x, seq_len):
        

