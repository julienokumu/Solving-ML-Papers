import torch
import jax  # Import JAX for high-performance numerical computations and automatic differentiation
import jax.numpy as jnp  # Import JAX's NumPy-like API for tensor operations
import flax.linen as nn  # Import Flax's neural network module for defining model layers
from transformers import GPT2Tokenizer  # Import GPT-2 tokenizer from Hugging Face for text tokenization
import math  # Import math for mathematical operations like square root

class MHSA(nn.Module):
    # Multi-Head Self-Attention module for attending to different parts of the input sequence
    d_model: int  # Model dimension (input/output size)
    n_heads: int  # Number of attention heads
    dropout: float = 0.1  # Dropout rate for regularization
    deterministic: bool = False  # Flag to disable dropout during inference

    def setup(self):
        # Initialize layers and parameters for the MHSA module
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"  # Ensure d_model is divisible by n_heads
        self.head_dim = self.d_model // self.n_heads  # Calculate dimension per attention head
        self.q_proj = nn.Dense(self.d_model, kernel_init=nn.initializers.normal(stddev=0.02))  # Linear layer for query projection
        self.k_proj = nn.Dense(self.d_model, kernel_init=nn.initializers.normal(stddev=0.02))  # Linear layer for key projection
        self.v_proj = nn.Dense(self.d_model, kernel_init=nn.initializers.normal(stddev=0.02))  # Linear layer for value projection
        self.out_proj = nn.Dense(self.d_model, kernel_init=nn.initializers.normal(stddev=0.02))  # Linear layer for output projection
        self.scale = 1.0 / math.sqrt(self.head_dim)  # Scaling factor for attention scores to stabilize gradients
        self.dropout_layer = nn.Dropout(self.dropout)  # Dropout layer for attention weights

    def __call__(self, x, attention_mask=None, key=None):
        # Forward pass for multi-head self-attention, with explicit key for dropout
        batch, seq_len, d_model = x.shape  # Extract batch size, sequence length, and model dimension
        # Project input to queries and reshape for multi-head attention (batch, n_heads, seq_len, head_dim)
        Q = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Project input to keys and reshape
        K = self.k_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Project input to values and reshape
        V = self.v_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Compute attention scores (Q * K^T) and apply scaling
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        # Handle attention mask for padding and causal modeling
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].astype(jnp.bool_)  # Reshape mask for broadcasting
        else:
            padding_mask = jnp.ones((batch, 1, 1, seq_len), dtype=jnp.bool_)  # Default mask of ones if none provided
        # Create causal mask to prevent attending to future tokens
        causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
        # Combine causal and padding masks
        combined_mask = causal_mask & padding_mask
        # Apply mask to scores, setting masked positions to a large negative value
        scores = jnp.where(combined_mask, scores, -1e9)
        # Apply softmax to convert scores to probabilities
        attn = nn.softmax(scores, axis=-1)
        # Apply dropout to attention weights, using deterministic flag for inference
        attn = self.dropout_layer(attn, deterministic=self.deterministic, rng=key)
        # Compute attention output by multiplying weights with values
        attn_output = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        # Project attention output back to d_model dimension
        output = self.out_proj(attn_output)
        return output  # Return the final attention output

class FFN(nn.Module):
    # Feed-Forward Network module for processing attention outputs
    d_model: int  # Model dimension
    d_ff: int  # Feed-forward hidden dimension
    dropout: float = 0.1  # Dropout rate for regularization
    deterministic: bool = False  # Flag to disable dropout during inference

    def setup(self):
        # Initialize layers for the FFN module
        self.linear1 = nn.Dense(self.d_ff, kernel_init=nn.initializers.normal(stddev=0.02))  # First linear layer to expand dimension
        self.linear2 = nn.Dense(self.d_model, kernel_init=nn.initializers.normal(stddev=0.02))  # Second linear layer to project back
        self.dropout_layer = nn.Dropout(self.dropout)  # Dropout layer for regularization

    def __call__(self, x, key=None):
        # Forward pass for the feed-forward network
        x = self.linear1(x)  # Apply first linear transformation to expand dimension
        x = nn.gelu(x)  # Apply GELU activation for non-linearity
        x = self.linear2(x)  # Apply second linear transformation to project back
        x = self.dropout_layer(x, deterministic=self.deterministic, rng=key)  # Apply dropout
        return x  # Return the feed-forward output

class Block(nn.Module):
    # Transformer block combining MHSA and FFN with residual connections
    d_model: int  # Model dimension
    n_heads: int  # Number of attention heads
    d_ff: int  # Feed-forward hidden dimension
    dropout: float = 0.1  # Dropout rate for regularization
    deterministic: bool = False  # Flag to disable dropout during inference

    def setup(self):
        # Initialize components of the transformer block
        self.attn = MHSA(self.d_model, self.n_heads, self.dropout)  # Initialize multi-head self-attention
        self.norm1 = nn.LayerNorm()  # Layer normalization after attention
        self.ffn = FFN(self.d_model, self.d_ff, self.dropout)  # Initialize feed-forward network
        self.norm2 = nn.LayerNorm()  # Layer normalization after FFN
        self.dropout_layer = nn.Dropout(self.dropout)  # Dropout for residual connections

    def __call__(self, x, attention_mask=None, key=None):
        # Forward pass for the transformer block
        subkey1, subkey2, subkey3 = jax.random.split(key, 3) if key is not None else (None, None, None)  # Split random key for dropout
        attn_output = self.attn(x, attention_mask, key=subkey1)  # Apply multi-head self-attention
        x = self.norm1(x + self.dropout_layer(attn_output, deterministic=self.deterministic, rng=subkey2))  # Residual connection and normalization
        ffn_output = self.ffn(x, key=subkey3)  # Apply feed-forward network
        x = self.norm2(x + self.dropout_layer(ffn_output, deterministic=self.deterministic, rng=subkey2))  # Residual connection and normalization
        return x  # Return the block output

class GPT(nn.Module):
    # GPT model for autoregressive language modeling
    vocab_size: int  # Vocabulary size for token embeddings
    d_model: int = 256  # Model dimension, reduced to prevent overfitting
    n_layers: int = 4  # Number of transformer blocks, reduced for small dataset
    n_heads: int = 4  # Number of attention heads, adjusted for d_model
    d_ff: int = 512  # Feed-forward hidden dimension, scaled with d_model
    context_length: int = 512  # Maximum sequence length, consistent with training
    dropout: float = 0.1  # Dropout rate for regularization
    deterministic: bool = False  # Flag to disable dropout during inference

    def setup(self):
        # Initialize components of the GPT model
        self.tok_emb = nn.Embed(self.vocab_size, self.d_model, embedding_init=nn.initializers.normal(stddev=0.02))  # Token embedding layer
        self.pos_emb = nn.Embed(self.context_length, self.d_model, embedding_init=nn.initializers.normal(stddev=0.02))  # Positional embedding layer
        self.dropout_layer = nn.Dropout(self.dropout)  # Dropout for embeddings
        self.layers = [Block(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)]  # List of transformer blocks
        self.norm = nn.LayerNorm()  # Final layer normalization
        self.out_proj = nn.Dense(self.vocab_size, kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros)  # Output projection to vocabulary

    def __call__(self, input_ids, attention_mask=None, key=None):
        # Forward pass for the GPT model
        batch, seq_len = input_ids.shape  # Extract batch size and sequence length
        subkeys = jax.random.split(key, self.n_layers + 1) if key is not None else [None] * (self.n_layers + 1)  # Split random key for layers
        tok_emb = self.tok_emb(input_ids)  # Convert input IDs to token embeddings
        positions = jnp.arange(0, seq_len)[None]  # Create position indices (0 to seq_len-1)
        pos_emb = self.pos_emb(positions)  # Convert position indices to positional embeddings
        x = self.dropout_layer(tok_emb + pos_emb, deterministic=self.deterministic, rng=subkeys[0])  # Combine embeddings with dropout
        for i, layer in enumerate(self.layers):  # Iterate through transformer blocks
            x = layer(x, attention_mask, key=subkeys[i + 1])  # Apply each transformer block
        x = self.norm(x)  # Apply final layer normalization
        logits = self.out_proj(x)  # Project to vocabulary size for next-token prediction
        return logits  # Return logits for each token

    @staticmethod
    def get_tokenizer():
        # Static method to load the GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Load pre-trained GPT-2 tokenizer
        return tokenizer  # Return the tokenizer
