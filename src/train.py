import os
import zipfile
import collections
import random

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

EMBED_DIM    = 100
WINDOW_SIZE  = 5
NEG_SAMPLES  = 5
BATCH_SIZE   = 8192
EPOCHS       = 3
LR           = 0.025
MIN_COUNT    = 5
MAX_VOCAB    = 20000
MAX_TOKENS   = 5_000_000
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def download_text8():
    if not os.path.exists("text8"):
        print("Downloading text8...")
        url = "http://mattmahoney.net/dc/text8.zip"
        r = requests.get(url, stream=True)
        with open("text8.zip", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile("text8.zip") as z:
            z.extractall()
        print("Done.")
    with open("text8", "r") as f:
        text = f.read().split()
    text = text[:MAX_TOKENS]
    print(f"Using {len(text):,} tokens")
    return text

def build_vocab(words, min_count=MIN_COUNT, max_vocab=MAX_VOCAB):
    counter = collections.Counter(words)
    vocab = [w for w, c in counter.most_common(max_vocab) if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    encoded = [word2idx[w] for w in words if w in word2idx]
    freq = np.array([counter[w] ** 0.75 for w in vocab])
    freq = freq / freq.sum()
    return encoded, word2idx, idx2word, freq

class SkipGramDataset(Dataset):
    def __init__(self, encoded, freq, window=WINDOW_SIZE, neg_samples=NEG_SAMPLES):
        self.data = encoded
        self.freq = freq
        self.window = window
        self.neg_samples = neg_samples
        self.vocab_size = len(freq)

        print("Building training pairs...")
        self.pairs = []
        for i, center in enumerate(tqdm(self.data)):
            r = random.randint(1, self.window)
            start = max(0, i - r)
            end   = min(len(self.data), i + r + 1)
            for j in range(start, end):
                if j != i:
                    self.pairs.append((center, self.data[j]))
        print(f"Total training pairs: {len(self.pairs):,}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        negs = np.random.choice(self.vocab_size, size=self.neg_samples, p=self.freq)
        return (
            torch.tensor(center,  dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(negs,    dtype=torch.long),
        )

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed  = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        # FIXED: both matrices initialized with small random values, NOT zeros
        nn.init.uniform_(self.in_embed.weight,  -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.uniform_(self.out_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)

    def forward(self, center, context, negatives):
        v_center  = self.in_embed(center)
        v_context = self.out_embed(context)
        v_negs    = self.out_embed(negatives)

        pos_score = torch.sum(v_center * v_context, dim=1)
        pos_loss  = torch.nn.functional.logsigmoid(pos_score)

        neg_score = torch.bmm(v_negs, v_center.unsqueeze(2)).squeeze(2)
        neg_loss  = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

def train(model, dataset, epochs=EPOCHS):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_steps = len(loader)
    step = 0

    for epoch in range(epochs):
        total_loss = 0
        for center, context, negs in tqdm(loader, desc=f"Epoch {epoch+1}"):
            center  = center.to(DEVICE)
            context = context.to(DEVICE)
            negs    = negs.to(DEVICE)

            lr = 0.001



            optimizer.zero_grad()
            loss = model(center, context, negs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    return model

def save_checkpoint(model, word2idx, path="embeddings.pt"):
    checkpoint = {
        'embeddings': model.in_embed.weight.detach().cpu(),
        'word2idx': word2idx,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

if __name__ == "__main__":
    words = download_text8()
    encoded, word2idx, idx2word, freq = build_vocab(words)
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size:,} | Corpus tokens: {len(encoded):,}")

    dataset = SkipGramDataset(encoded, freq)
    model   = SkipGram(vocab_size, EMBED_DIM).to(DEVICE)

    model = train(model, dataset)
    save_checkpoint(model, word2idx)
