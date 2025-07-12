import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
import time
from jax import random
import jax.profiler

@dataclass
class ModelArgs:
    vocab_size: int = 256
    dim: int = 512
    head_dim: int = 64
    hidden_dim: int = 1024
    n_heads: int = 8
    n_layers: int = 2
    n_kv_heads: int = 2
    base: float = 1000.0
    eps: float = 1e-6

args = ModelArgs()

class RMSNorm(nn.Module):
    dim: int
    eps: float

    def setup(self):
        self.gamma = self.param('gamma', nn.initializers.ones, (self.dim,))
    
    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.gamma

class SwiGLUFFN(nn.Module):
    dim: int
    hidden_dim: int

    def setup(self):
        self.linear1 = nn.Dense(self.hidden_dim)
        self.gate = nn.Dense(self.hidden_dim)
        self.linear2 = nn.Dense(self.dim)
        self.beta = self.param('beta', nn.initializers.ones, (1,))
    
    def __call__(self, x):
        gate = nn.silu(self.gate(x))
        value = self.linear1(x)
        return self.linear2(value * gate * self.beta)

class RoPE(nn.Module):
    dim: int
    base: float

    def setup(self):
        theta = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        self.theta = theta
    
    def __call__(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(positions, self.theta)
        cos_emb = jnp.cos(freqs)[None, :, None, :]
        sin_emb = jnp.sin(freqs)[None, :, None, :]
        return cos_emb, sin_emb

def apply_rotary_emb(x, cos_emb, sin_emb):
    x1, x2 = x[..., ::2], x[..., 1::2]
    output = jnp.zeros_like(x)
    output = output.at[..., ::2].set(x1 * cos_emb - x2 * sin_emb)
    output = output.at[..., 1::2].set(x1 * sin_emb + x2 * cos_emb)
    return output

class MHA(nn.Module):
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    def setup(self):
        self.q_proj = nn.Dense(self.n_heads * self.head_dim, use_bias=False)
        self.k_proj = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False)
        self.v_proj = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False)
        self.out_proj = nn.Dense(self.dim, use_bias=False)
        self.rotary_emb = RoPE(self.head_dim, args.base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x, mask=None):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        cos_emb, sin_emb = self.rotary_emb(q, seq_len)
        q = apply_rotary_emb(q, cos_emb, sin_emb)
        k = apply_rotary_emb(k, cos_emb, sin_emb)

        repeat_factor = self.n_heads // self.n_kv_heads
        k = jnp.repeat(k, repeat_factor, axis=2)
        v = jnp.repeat(v, repeat_factor, axis=2)

        q, k, v = [jnp.transpose(x, (0, 2, 1, 3)) for x in (q, k, v)]

        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = jnp.where(mask == 0, float('-inf'), scores)
        
        attn = nn.softmax(scores, axis=-1)
        output = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.out_proj(output)

class Block(nn.Module):
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int

    def setup(self):
        self.attn = MHA(
            self.dim,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim
        )
        self.laynorm1 = RMSNorm(
            self.dim,
            args.eps
        )

        self.ffn = SwiGLUFFN(
            self.dim,
            self.hidden_dim
        )
        self.laynorm2 = RMSNorm(
            self.dim,
            args.eps
        )
    
    def __call__(self, x, mask=None):
        x_norm = self.laynorm1(x)
        x = x + self.attn(x_norm, mask)
        output = self.laynorm2(x)
        return x + self.ffn(output)

class MixtralNeMo(nn.Module):
    vocab_size: int
    dim: int
    hidden_dim: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    n_layers: int

    def setup(self):
        self.emb = nn.Embed(self.vocab_size, self.dim)
        self.layers = [
            Block(
                self.dim,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.hidden_dim
            ) for _ in range(self.n_layers)
        ]
        self.norm = RMSNorm(self.dim, args.eps)
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)
    
    def __call__(self, input_ids, mask=None):
        batch, seq_len = input_ids.shape
    
        x = self.emb(input_ids)

        if mask is None:
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len)))
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

def inference():
    key = random.PRNGKey(0)
    model = MixtralNeMo(
        vocab_size=args.vocab_size,
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        head_dim=args.head_dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers
    )

    input_ids = jnp.ones((1, 128), dtype=jnp.int32)
    params = model.init(key, input_ids)

    @jax.jit
    def forward(params, input_ids, mask=None):
        return model.apply(params, input_ids, mask)
    
    _ = forward(params, input_ids)

    jax.profiler.start_trace("./jax=trace")
    start_time = time.time()

    logits = forward(params, input_ids)

    logits.block_until_ready()
    end_time = time.time()
    jax.profiler.stop_trace()

    print(f"inference time: {(end_time - start_time) * 1000:.2f} ms")
    print("profile data saved to ./jax-trace. view using tensorboard")

if __name__ == "__main__":
    inference()
    