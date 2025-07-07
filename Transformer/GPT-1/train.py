import torch
import torch.nn as nn
from torch.optim import Adam
from datasets import load_dataset
from tqdm import tqdm
import os
from model import GPT
import torch.nn.functional as F
import math

batch_size = 8
context_length = 512
num_epochs = 10
learning_rate = 2.5e-4
checkpoint_dir = "/kaggle/working/checkpoints"
checkpoint_frequency = 60000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(tokenizer, split="train"):
    dataset_path = f"/kaggle/input/tinystories-narrative-classification/{'train.csv' if split == 'train' else 'validation.csv'}"
    dataset = load_dataset("csv", data_files={split: dataset_path})[split]
    def tokenize_function(examples):
        cleaned_texts = [str(text) if text is not None else "" for text in examples["text"]]
        return tokenizer(cleaned_texts, truncation=True, padding="max_length", max_length=context_length)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def get_dataloader(tokenized_dataset, batch_size, shuffle=True):
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item["input_ids"], dtype=torch.long) for item in batch])
        mask = torch.stack([torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch])
        return {"input_ids": input_ids, "attention_mask": mask}
    return torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def save_checkpoint(model, optimizer, epoch, batch_idx, loss):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"gpt1_tinystories_epoch{epoch+1}_batch{batch_idx+1}.pt")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_latest_checkpoint(model, optimizer):
    if not os.path.exists(checkpoint_dir):
        return 0, 0, None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("gpt1_tinystories_epoch")]
    if not checkpoints:
        return 0, 0, None
    latest_checkpoint = max(checkpoints, key=lambda x: (int(x.split('_')[2].replace('epoch', '')), int(x.split('_')[3].replace('batch', '').split('.')[0])))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch_idx']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {epoch+1}, batch {batch_idx+1}")
    return epoch, batch_idx, loss

def generate_text(model, tokenizer, prompt, max_length=100, device="cpu"):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            if generated.shape[1] > context_length:
                generated = generated[:, -context_length:]
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    return generated_text

def train_model():
    torch.cuda.empty_cache()
    tokenizer = GPT.get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    model = GPT(vocab_size=vocab_size, max_seq_len=context_length)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataset = prepare_data(tokenizer, split="train")
    val_dataset = prepare_data(tokenizer, split="validate")
    train_dataloader = get_dataloader(train_dataset, batch_size)
    val_dataloader = get_dataloader(val_dataset, batch_size, shuffle=False)

    start_epoch, start_batch, last_loss = load_latest_checkpoint(model, optimizer)
    model.train()

    with tqdm(range(start_epoch, num_epochs), unit="epoch", desc="Training Progress") as tepoch:
        for epoch in tepoch:
            total_loss = 0
            batch_count = 0
            optimizer.zero_grad()

            for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                if epoch == start_epoch and i <= start_batch:
                    continue
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(input_ids, mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                batch_count += 1

                if (i + 1) % checkpoint_frequency == 0:
                    save_checkpoint(model, optimizer, epoch, i, loss.item())

            avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
            tepoch.set_postfix(avg_loss=avg_train_loss)
            print(f"Epoch {epoch + 1} completed, Avg Train Loss: {avg_train_loss:.4f}")

            model.eval()
            val_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validating"):
                    input_ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    logits = model(input_ids, mask)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    val_loss += loss.item()
                    val_batch_count += 1
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            val_perplexity = math.exp(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
            model.train()

            save_checkpoint(model, optimizer, epoch, i, avg_train_loss)

    print("\n=== Training Complete! ===")
    print("Enter prompt to generate text (type 'quit' to exit):")
    while True:
        prompt = input("User: ")
        if prompt.lower() == "quit":
            break
        generated_text = generate_text(model, tokenizer, prompt, max_length=300, device=device)
        print(f"GPT-Stories: {generated_text}\n")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("GPT-Stories is training...")
    train_model()
    
