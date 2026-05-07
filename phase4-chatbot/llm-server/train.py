# ============================================
# TRAIN.PY — PyTorch Version
# Real backpropagation, trains properly
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random

# ── Read training data
with open("data/stocks.txt", "r") as f:
    text = f.read()

# ── Build vocabulary
vocab      = sorted(set(text))
vocab_size = len(vocab)
c2i        = { ch:i for i,ch in enumerate(vocab) }
i2c        = { i:ch for i,ch in enumerate(vocab) }

def encode(s): return [c2i[c] for c in s]
def decode(l): return ''.join([i2c[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

print(f"Vocab size: {vocab_size}")
print(f"Training chars: {len(data)}")

# ── Hyperparameters — tune these
BLOCK_SIZE  = 128     # context window (chars)
EMBED_SIZE  = 128    # embedding dimensions
HEAD_SIZE   = 32     # attention head size
N_HEADS     = 4      # number of attention heads
N_LAYERS    = 4      # transformer blocks
DROPOUT     = 0.1    # dropout rate
BATCH_SIZE  = 64     # examples per step
N_STEPS     = 10000   # training steps
LEARN_RATE  = 3e-4   # learning rate
EVAL_EVERY  = 500    # print loss every N steps

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")


# ============================================
# MODEL — Full Transformer in PyTorch
# ============================================

class Head(nn.Module):
    """ One attention head """
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.query = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]
        # Attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v


class MultiHead(nn.Module):
    """ Multiple attention heads in parallel """
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_heads * head_size, EMBED_SIZE)
        self.drop  = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
    """ Two-layer MLP after attention """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_SIZE, 4 * EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(4 * EMBED_SIZE, EMBED_SIZE),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ One full transformer block """
    def __init__(self):
        super().__init__()
        head_size  = EMBED_SIZE // N_HEADS
        self.sa    = MultiHead(N_HEADS, head_size)
        self.ff    = FeedForward()
        self.ln1   = nn.LayerNorm(EMBED_SIZE)
        self.ln2   = nn.LayerNorm(EMBED_SIZE)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # attention + residual
        x = x + self.ff(self.ln2(x))   # feedforward + residual
        return x


class TinyLLM(nn.Module):
    """ Full character-level transformer """
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_SIZE)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_SIZE)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln      = nn.LayerNorm(EMBED_SIZE)
        self.head    = nn.Linear(EMBED_SIZE, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok  = self.tok_emb(idx)                              # [B,T,C]
        pos  = self.pos_emb(torch.arange(T, device=device))  # [T,C]
        x    = self.blocks(tok + pos)
        x    = self.ln(x)
        logits = self.head(x)                                 # [B,T,vocab_size]

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx, n_chars, temperature=1.0):
        for _ in range(n_chars):
            idx_crop   = idx[:, -BLOCK_SIZE:]
            logits, _  = self(idx_crop)
            logits     = logits[:, -1, :] / temperature
            probs      = F.softmax(logits, dim=-1)
            next_idx   = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat((idx, next_idx), dim=1)
        return idx


# ============================================
# TRAINING LOOP
# ============================================
def get_batch():
    ix  = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([data[i   : i+BLOCK_SIZE  ] for i in ix])
    y   = torch.stack([data[i+1 : i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)


model     = TinyLLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)
n_params  = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}\n")
print("Training...\n")

for step in range(N_STEPS):
    xb, yb    = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()          # real backprop — updates ALL weights
    optimizer.step()

    if (step + 1) % EVAL_EVERY == 0:
        print(f"Step {step+1:5d}/{N_STEPS} | Loss: {loss.item():.4f}")

        # Generate a sample to see progress
        seed     = torch.tensor([encode("Stock prices")], dtype=torch.long).to(device)
        sample   = model.generate(seed, n_chars=60, temperature=0.8)
        generated = decode(sample[0].tolist()[len("Stock prices"):])
        print(f"Sample: 'Stock prices{generated}'\n")


# ── Save weights
print("\nSaving weights...")
torch.save(model.state_dict(), "weights_torch.pt")

# Also save vocab so app.py can load it
with open("vocab.json", "w") as f:
    json.dump({ "vocab": vocab, "vocab_size": vocab_size }, f)

print("Saved weights_torch.pt and vocab.json")
print(f"\nFinal loss: {loss.item():.4f}")
print("Done! Now update app.py to load the PyTorch model.")