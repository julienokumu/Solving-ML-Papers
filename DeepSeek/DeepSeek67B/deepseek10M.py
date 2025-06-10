import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import requests
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from tqdm import tqdm
import os.path as osp

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        norm_x = x / rms
        output = norm_x * self.gamma
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
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)
        return rotated

class GQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        assert d_model == n_heads * self.head_dim, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, " n_heads must be divisible by n_kv_heads"
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        repeat_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.out_proj(output)
        return output
    
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_intermed = int((8/9) * d_model)
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
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.gqa = GQA(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch, seq_len, d_model = x.shape
        x = self.attn_norm(x)
        attn_out = self.gqa(x, mask)
        x = x + attn_out
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        output = x + ffn_out
        return output

class DeepSeek10M(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_kv_heads: int = 2, n_layers: int = 6, context_length: int = 128, vocab_size: int = 65):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                n_kv_heads
            ) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.context_length = context_length
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError(f"seq_len {seq_len}, exceeds context_length {self.context_length}")
        x = self.embedding(x)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(1)
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
    
class TinyShakespeareDataset:
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
    file_path = "/kaggle/input/tinyshakespeare-dataset/tinyshakespeare.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            text = f.read()
        return text
    else:
        raise FileNotFoundError("Tiny Shakespeare dataset not found. Please upload it to Kaggle.")

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

def save_checkpoint(model, optimizer, epoch, batch_idx, loss, checkpoint_dir="/kaggle/working/checkpoints"):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = osp.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_latest_checkpoint(model, optimizer, checkpoint_dir="/kaggle/working/checkpoints"):
    
    if not os.path.exists(checkpoint_dir):
        return 0, 0, None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    if not checkpoints:
        return 0, 0, None
    latest_checkpoint = max(checkpoints, key=lambda x: (int(x.split('_')[2]), int(x.split('_')[4].split('.')[0])))
    checkpoint_path = osp.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch_idx']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {epoch+1}, batch {batch_idx+1}")
    return epoch, batch_idx, loss

def train_model():
    batch_size = 32  
    context_length = 64
    num_epochs = 20
    learning_rate = 4.2e-4
    grad_accum_steps = 1
    num_workers = 0 
    checkpoint_dir = "/kaggle/working/checkpoints"
    checkpoint_frequency = 10000

    text = download_tinyshakespeare()
    tokenizer = Tokenizer(text)
    dataset = TinyShakespeareDataset(text, tokenizer, context_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    model = DeepSeek10M(
        d_model=256,
        n_heads=8,
        n_kv_heads=2,
        n_layers=6,
        context_length=context_length,
        vocab_size=tokenizer.vocab_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch, start_batch, last_loss = load_latest_checkpoint(model, optimizer, checkpoint_dir)
    model.train()

   
    with tqdm(range(start_epoch, num_epochs), unit="epoch", desc="Training Progress") as tepoch:
        for epoch in tepoch:
            total_loss = 0
            batch_count = 0
            optimizer.zero_grad()

            for i, (x, y) in enumerate(dataloader):
                if epoch == start_epoch and i <= start_batch:
                    continue  
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

                
                if (i + 1) % checkpoint_frequency == 0:
                    save_checkpoint(model, optimizer, epoch, i, total_loss / batch_count, checkpoint_dir)

            
            avg_loss = total_loss / batch_count

            
            tepoch.set_postfix(avg_loss=avg_loss)

            
            print(f"Epoch {epoch + 1} completed, Avg Loss {avg_loss:.4f}")

            
            save_checkpoint(model, optimizer, epoch, i, avg_loss, checkpoint_dir)

    print("\n===Training Complete!===")
    print("Enter prompt to generate text (type 'quit' to exit):")
    while True:
        prompt = input("User: ")
        if prompt.lower() == "quit":
            break
        generated_text = generate_text(model, tokenizer, prompt, max_length=100, device=device)
        print(f"DeepSeek10M: {generated_text}\n")

if __name__ == "__main__":
    print("DeepSeek10M is training.....")
    train_model()