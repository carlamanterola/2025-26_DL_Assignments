import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import numpy as np
import re

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_VOCAB_SIZE = 10_000
EMBEDDING_DIM  = 100
HIDDEN_DIM     = 64
BATCH_SIZE     = 64
MAX_SEQ_LEN    = 200
EPOCHS         = 8
WEIGHT_DECAY   = 1e-4
LR             = 5e-4


# ── Tokeniser ─────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


# ── Vocabulary builder ────────────────────────────────────────────────────────
def build_vocab(train_raw, max_vocab_size: int = MAX_VOCAB_SIZE):
    print("Building vocabulary …")
    counter: Counter = Counter()
    for item in train_raw:
        counter.update(tokenize(item["text"]))

    specials    = ["<pad>", "<unk>"]
    vocab_tokens = [tok for tok, _ in counter.most_common(max_vocab_size)]
    vocab       = specials + vocab_tokens
    stoi        = {tok: i for i, tok in enumerate(vocab)}
    print(f"Vocabulary size: {len(vocab)}")
    return vocab, stoi


# ── Encoder ───────────────────────────────────────────────────────────────────
def make_encoder(stoi: dict, unk_idx: int = 1, max_len: int = MAX_SEQ_LEN):
    def encode(text: str) -> list[int]:
        tokens = tokenize(text)[:max_len]
        return [stoi.get(tok, unk_idx) for tok in tokens]
    return encode


# ── GloVe loader ──────────────────────────────────────────────────────────────
def load_glove(path: str, stoi: dict, pad_idx: int, dim: int = 100) -> torch.Tensor:
    print(f"Loading GloVe from {path} …")
    embeddings = torch.zeros(len(stoi), dim)
    nn.init.uniform_(embeddings, -0.01, 0.01)
    embeddings[pad_idx] = torch.zeros(dim)

    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if word in stoi:
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                embeddings[stoi[word]] = vec
                found += 1

    print(f"GloVe: {found}/{len(stoi)} vocab words matched")
    return embeddings


# ── Dataset ───────────────────────────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, hf_dataset, encode_fn):
        self.data      = hf_dataset
        self.encode_fn = encode_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        ids   = self.encode_fn(item["text"])
        label = item["label"]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)


def collate_fn(batch, pad_idx: int = 0):
    texts, labels = zip(*batch)
    max_len = max(t.size(0) for t in texts)
    padded  = torch.full((max_len, len(texts)), pad_idx, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[:len(t), i] = t
    return padded, torch.stack(labels)


def make_collate_fn(pad_idx: int):
    def _collate(batch):
        return collate_fn(batch, pad_idx=pad_idx)
    return _collate


def build_dataloaders(train_raw, test_raw, encode_fn, pad_idx: int,
                      batch_size: int = BATCH_SIZE):
    train_ds = IMDBDataset(train_raw, encode_fn)
    test_ds  = IMDBDataset(test_raw,  encode_fn)
    _collate = make_collate_fn(pad_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=_collate)
    return train_loader, test_loader


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm      = nn.LSTM(embedding_dim, hidden_dim,
                                 num_layers=2, bidirectional=True, dropout=0.5)
        self.fc        = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(0.5)

    def forward(self, x):
        emb        = self.dropout(self.embedding(x))
        _, (h, _)  = self.lstm(emb)
        h          = torch.cat([h[-2], h[-1]], dim=1)
        h          = self.dropout(h)
        return self.fc(h)


# ── Metrics ───────────────────────────────────────────────────────────────────
def binary_acc(preds: torch.Tensor, y: torch.Tensor) -> float:
    rounded = torch.round(torch.sigmoid(preds))
    correct = torch.eq(rounded, y).float()
    return (correct.sum() / len(correct)).item()


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0
    for i, (text, label) in enumerate(iterator):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)
        acc  = binary_acc(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc  += acc
        if i % 20 == 0:
            print(f"  Step {i:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# ── Evaluation loop ───────────────────────────────────────────────────────────
def evaluate_epoch(model, iterator, criterion, device):
    model.eval()
    epoch_loss, epoch_acc = 0.0, 0.0
    with torch.no_grad():
        for text, label in iterator:
            text, label = text.to(device), label.to(device)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, label)
            acc  = binary_acc(predictions, label)
            epoch_loss += loss.item()
            epoch_acc  += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)