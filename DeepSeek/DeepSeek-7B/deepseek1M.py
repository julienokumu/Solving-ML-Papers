import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import requests
from torch.utils.data import Dataset, DataLoader
import os
import torch.optim as optim

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normalized_x = x / rms
        output = normalized_x * self.gamma
        return output
    
class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_heads, seq_len, head_dim = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(-1)
        angles = positions * self.theta
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)
        return rotated

class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model == n_heads * self.head_dim, "d_model must be divisible by n_heads"
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.out_proj(output)
        return output

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_intermed = int((8/3) * d_model)
        self.linear1 = nn.Linear(d_model, self.d_intermed)
        self.linear2 = nn.Linear(d_model, self.d_intermed)
        self.linear3 = nn.Linear(self.d_intermed, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        gate = F.silu(self.linear2(x))
        x = self.linear1(x)
        x = x * gate
        output = self.linear3(x)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mha = MHA(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        x = self.attn_norm(x)
        attn_output = self.mha(x, mask)
        x = x + attn_output
        x = self.ffn_norm(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        return x

class DeepSeek1M(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, context_length: int = 128, vocab_size: int = 65):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads
            ) for _ in range(n_layers)
        ]) 
        self.final_norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.context_length = context_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError(f"seq_len {seq_len} exceeds context_length {self.context_length}")
        x = self.embedding(x)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.output(x)
        return logits
    
class Tokenizer:
    def __init__(self, text):
        self.char = sorted(list(set(text)))
        self.vocab_size = len(self.char)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.char)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.char)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join(self.idx_to_char[i] for i in indices)
    
class TinyShakespeareDataset(Dataset):
    def __init__(self, text, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.data = torch.tensor(self.tokenizer.encode(text))

    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1: idx + self.context_length + 1]
        return x, y
    
def download_tinyshakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("tinyshakespeare.txt"):
        response = requests.get(url)
        with open("tinyshakespeare.txt", "w") as f:
            f.write(response.text)
    with open("tinyshakespeare.txt", "r") as f:
        text = f.read()
    return text

def generate_text(model, tokenizer, prompt, max_length=100, device="cpu"):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            if generated.shape[1] > model.context_length:
                generated = generated[:, -model.context_length:]
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    return generated_text

def train_model():
    batch = 64
    context_length = 64
    num_epochs = 20
    learning_rate = 4.2e-4
    grad_accum_steps = 4
    text = download_tinyshakespeare()
    tokenizer = Tokenizer(text)
    dataset = TinyShakespeareDataset(text, tokenizer, context_length)
    dataloader = DataLoader(dataset, batch_size=batch)
    model = DeepSeek1M(
        d_model=128,
        n_heads=4,
        n_layers=4,
        context_length=context_length,
        vocab_size=tokenizer.vocab_size
    )
    device = torch.device("cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = logits.view(-1, tokenizer.vocab_size)
            y = y.view(-1)
            loss = criterion(logits, y)
            loss = loss / grad_accum_steps
            loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            batch_count += 1

            if (i + 1) % 100 == 0:
                print(f"epoch {epoch + 1}, batch {i + 1}, avg loss { total_loss / batch_count:.4f}")
            
        avg_loss = total_loss / batch_count
        print(f"epoch {epoch + 1} completed, avg loss {avg_loss:.4f}")
    
    print("\n===Training Complete!===")
    print("Enter Prompt to generate text(type 'quit' to exit)")
    while True:
        prompt = input("User: ")
        if prompt.lower() == "quit":
            break
        generated_text = generate_text(model, tokenizer, prompt, max_length=100, device=device)
        print(f"DeepSeek1M: {generated_text}\n")

if __name__ == "__main__":
    print("DeepSeek-1M is training.....")
    train_model()
    

