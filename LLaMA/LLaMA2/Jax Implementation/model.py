import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import random, jit
import time
import math

d_model = 64
n_heads = 8
head_dim = d_model // n_heads
n_layers = 6
seq_len = 100
vocab_size = 100
n_groups = 4



def RoPE(d_model, seq_len):
    theta = 1 / (1000 ** (jnp.arange(0, d_model, 2).astype(jnp.bfloat16) / d_model))
    pos = jnp.arange(seq_len, dtype=jnp.bfloat16).reshape(-1, 1)
    
    angles = pos * theta
    sin = jnp.sin(angles)
    cos = jnp.cos(angles)

    return sin, cos

def apply_rope(x, sin, cos, seq_len):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    sin = sin[:seq_len, None, :]
    cos = cos[:seq_len, None, :]

    rot_x_even = x_even * cos - x_odd * sin
    rot_x_odd = x_even * sin + x_odd * cos

    return jnp.concatenate((rot_x_even, rot_x_odd), axis=-1)



def RMSNorm(x, scale, eps=1e-6):
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    x_norm = x / rms

    return x_norm * scale



def FFN(params, x):
    x_ff = jnp.dot(x, params['linear1'])
    x_ff = x_ff * jnn.silu(x_ff)

    return jnp.dot(x_ff, params['linear2'])



def GQA(params, x, k_cache, v_cache, step=None):
    batch, seq_len = x.shape[0], x.shape[1]

    Q = jnp.dot(x, params['q_proj']).reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    K = jnp.dot(x, params['k_proj']).reshape(batch, seq_len, n_heads // n_groups, head_dim).transpose(0, 2, 1, 3)
    V = jnp.dot(x, params['v_proj']).reshape(batch, seq_len, n_heads // n_groups, head_dim).transpose(0, 2, 1, 3)

    if step is not None:
        k_cache = k_cache.at[:, :, step].set(K[:, :, -1])
        v_cache = v_cache.at[:, :, step].set(V[:, :, -1])

        k = k_cache[:, :, :step+1]
        v = v_cache[:, :, :step+1]

    else:
        k = K
        v = V

    k = jnp.repeat(k, n_groups, axis=1)
    v = jnp.repeat(v, n_groups, axis=1)

    scores = jnp.matmul(Q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    attn = jnn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_heads * head_dim)
    out = jnp.dot(out, params['out_proj'])
    
    return out, k_cache, v_cache



def LlamaLayer(params, x, k_cache, v_cache, step=None):
    x_norm = RMSNorm(x, params['rms_norm1'])
    attn_out, k_cache, v_cache = GQA(params['attn'], x_norm, k_cache, v_cache, step)
    x = x + attn_out
    x_ff = RMSNorm(x, params['rms_norm2'])
    x = x + FFN(params['ff'], x_ff)

    return x, k_cache, v_cache



def Llama68M(params, x, sin, cos, k_caches, v_caches, step=None):
    x = params['embedding'][x]
    x = apply_rope(x, sin, cos, x.shape[1])

    for i in range(n_layers):
        x, k_caches[i], v_caches[i] = LlamaLayer(params['layers'][i], x, k_caches[i], v_caches[i], step)
    x = RMSNorm(x, params['rms_norm'])
    x = jnp.dot(x, params['linear'])

    return x, k_caches, v_caches



def init_params(key):
    key = random.split(key, 10)
    params = {
        'embedding': random.normal(key[0], (vocab_size, d_model), dtype=jnp.bfloat16) * 0.02,
        'rms_norm': jnp.ones(d_model, dtype=jnp.bfloat16),
        'linear': random.normal(key[1], (d_model, vocab_size), dtype=jnp.bfloat16) * 0.02,
        'layers': []
    }
    for i in range(n_layers):
        layer_key = random.split(key[i + 2], 6)
        layer_params = {
            'rms_norm1': jnp.ones(d_model, dtype=jnp.bfloat16),
            'attn': {
                'q_proj': random.normal(layer_key[0], (d_model, n_heads * head_dim), dtype=jnp.bfloat16) * 0.02,
                'k_proj': random.normal(layer_key[1], (d_model, (n_heads // n_groups) * head_dim), dtype=jnp.bfloat16) * 0.02,
                'v_proj': random.normal(layer_key[2], (d_model, (n_heads // n_groups) * head_dim), dtype=jnp.bfloat16) * 0.02,
                'out_proj': random.normal(layer_key[3], (n_heads * head_dim, d_model), dtype=jnp.bfloat16) * 0.02
            },
            'rms_norm2': jnp.ones(d_model, dtype=jnp.bfloat16),
            'ff': {
                'linear1': random.normal(layer_key[4], (d_model, 640), dtype=jnp.bfloat16) * 0.02,
                'linear2': random.normal(layer_key[5], (640, d_model), dtype=jnp.bfloat16) * 0.02
            }
        }
        params['layers'].append(layer_params)
    return params



def init_caches(batch):
    k_caches = [jnp.zeros((batch, n_heads // n_groups, seq_len, head_dim), dtype=jnp.bfloat16) for _ in range(n_layers)]
    v_caches = [jnp.zeros((batch, n_heads // n_groups, seq_len, head_dim), dtype=jnp.bfloat16) for _ in range(n_layers)]
    return k_caches, v_caches



Llama68M_JIT = jit(Llama68M, static_argnums=(6,))

def main():
    key = random.PRNGKey(0)
    params = init_params(key)
    sin, cos = RoPE(seq_len, d_model)
    k_caches, v_caches = init_caches(batch=1)
    input_ids = jnp.ones((1, 20), dtype=jnp.int32)
    num_runs = 10
    output, k_caches, v_caches = Llama68M_JIT(params, input_ids, sin, cos, k_caches, v_caches)
    output.block_until_ready()

    start_time = time.time()
    for _ in range(num_runs):
        output, k_caches, v_caches = Llama68M_JIT(params, input_ids, sin, cos, k_caches, v_caches)
        output.block_until_ready()
    avg_time = (time.time() - start_time) / num_runs

    print(f"average inference time over {num_runs} runs: {avg_time:.4f} seconds")

if __name__ == "__main__":
    main()


