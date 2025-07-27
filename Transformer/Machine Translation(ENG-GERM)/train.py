import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer
import math

dataset = [
    ("I eat an apple", "Ich esse einen Apfel"),
    ("She reads a book", "Sie liest ein Buch"),
    ("He runs fast", "Er läuft schnell"),
    ("We go to school", "Wir gehen zur Schule"),
    ("They play football", "Sie spielen Fußball"),
    ("The cat sleeps", "Die Katze schläft"),
    ("I drink water", "Ich trinke Wasser"),
    ("You write a letter", "Du schreibst einen Brief"),
    ("The dog barks", "Der Hund bellt"),
    ("She sings a song", "Sie singt ein Lied"),
    ("He drives a car", "Er fährt ein Auto"),
    ("We watch a movie", "Wir schauen einen Film"),
    ("They eat bread", "Sie essen Brot"),
    ("The bird flies", "Der Vogel fliegt"),
    ("I read a newspaper", "Ich lese eine Zeitung"),
    ("You run to the park", "Du läufst zum Park"),
    ("The sun shines", "Die Sonne scheint"),
    ("He writes a story", "Er schreibt eine Geschichte"),
    ("She dances well", "Sie tanzt gut"),
    ("We play chess", "Wir spielen Schach"),
    ("They drink coffee", "Sie trinken Kaffee"),
    ("The horse runs", "Das Pferd läuft"),
    ("I see a star", "Ich sehe einen Stern"),
    ("You sing a song", "Du singst ein Lied"),
    ("The tree grows", "Der Baum wächst"),
    ("He paints a picture", "Er malt ein Bild"),
    ("She walks to school", "Sie geht zur Schule"),
    ("We read books", "Wir lesen Bücher"),
    ("They write letters", "Sie schreiben Briefe"),
    ("The moon shines", "Der Mond scheint"),
]


def build_vocab(sentences, lang):
    word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>"}
    current_idx = 3
    for sentence in sentences:
        tokens = sentence.lower().split()
        for token in tokens:
            if token not in word2idx:
                word2idx[token] = current_idx
                idx2word[current_idx] = token
                current_idx += 1
    return word2idx, idx2word


en_sentences = [pair[0] for pair in dataset]
de_sentences = [pair[1] for pair in dataset]
en_word2idx, en_idx2word = build_vocab(en_sentences, 'en')
de_word2idx, de_idx2word = build_vocab(de_sentences, 'de')


def sentence_to_ids(sentence, word2idx, max_len):
    tokens = sentence.lower().split()
    ids = [word2idx["<sos>"]] + [word2idx.get(token, 0) for token in tokens] + [word2idx["<eos>"]]
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids[:max_len], dtype=torch.long)


max_len = max(max(len(pair[0].split()), len(pair[1].split())) for pair in dataset) + 2
src_data = torch.stack([sentence_to_ids(pair[0], en_word2idx, max_len) for pair in dataset])
tgt_data = torch.stack([sentence_to_ids(pair[1], de_word2idx, max_len) for pair in dataset])


d_model = 16
n_heads = 2
inter_dim = 32
n_layers = 1
dropout = 0.1
src_vocab_size = len(en_word2idx)
tgt_vocab_size = len(de_word2idx)
epochs =  150000
warmup_steps = 1000
learning_rate = 0.0001


model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    inter_dim=inter_dim,
    max_len=max_len,
    dropout=dropout
)


criterion = nn.CrossEntropyLoss(ignore_index=de_word2idx["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)


class WarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self._step_count
        lr = (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [lr for _ in self.base_lrs]


scheduler = WarmupLR(optimizer, d_model, warmup_steps)


model.train()
total_steps = 0
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(dataset)):
        optimizer.zero_grad()
        src = src_data[i].unsqueeze(0)
        tgt = tgt_data[i].unsqueeze(0)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model(src, tgt_input)
        loss = criterion(output.view(-1, tgt_vocab_size), tgt_output.view(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_steps += 1
        total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'average loss: {avg_loss:.4f}')


def translate_sentence(model, sentence, en_word2idx, de_word2idx, de_idx2word, max_len, device="cpu"):
    model.eval()
    src = sentence_to_ids(sentence, en_word2idx, max_len).unsqueeze(0).to(device)
    tgt = torch.tensor([[de_word2idx["<sos>"]]], dtype=torch.long).to(device)

    for _ in range(max_len -1):
        with torch.no_grad():
            output = model(src, tgt)
        next_token = output[:, -1, :].argmax(dim=-1).item()
        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        if next_token == de_word2idx["<eos>"]:
            break

    translated = [de_idx2word[idx.item()] for idx in tgt[0] if idx.item() not in {de_word2idx["<sos>"], de_word2idx["<pad>"]}]
    return " ".join(translated)


test_sentences = [
    "I eat an apple",
    "She reads a book",
    "He runs fast",
    "The cat sleeps",
    "They eat bread",
    "The horse runs",
    "I see a star",
    "The dog barks",
    "He drives a car",
    "The moon shines"
]

print("\nTransformer ENG-GER Translations:")
for sentence in test_sentences:
    translation = translate_sentence(model, sentence, en_word2idx, de_word2idx, de_idx2word, max_len)
    print(f"English: {sentence}")
    print(f"German: {translation}\n")
