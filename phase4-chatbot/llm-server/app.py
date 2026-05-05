# ============================================
# APP.PY — Python Flask Server (PyTorch Version)
# Full model classes included — no imports needed
# Run with: python app.py
# Listens on: http://localhost:5000
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

app = Flask(__name__)
CORS(app)


# ============================================
# STEP 1 — Load vocab from vocab.json
# (created by train.py)
# ============================================
with open("vocab.json", "r") as f:
    vocab_data = json.load(f)

vocab      = vocab_data["vocab"]
vocab_size = vocab_data["vocab_size"]
c2i        = { ch: i for i, ch in enumerate(vocab) }
i2c        = { i: ch for i, ch in enumerate(vocab) }

def encode(s): return [c2i.get(c, 0) for c in s]
def decode(l): return ''.join([i2c[i] for i in l])

print(f"Vocab loaded: {vocab_size} characters")


# ============================================
# STEP 2 — Model hyperparameters
# Must match EXACTLY what you used in train.py
# ============================================
BLOCK_SIZE = 64
EMBED_SIZE = 64
HEAD_SIZE  = 32
N_HEADS    = 4
N_LAYERS   = 3
DROPOUT    = 0.1


# ============================================
# STEP 3 — Model classes
# Exact same classes as train.py
# ============================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.query = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k   = self.key(x)
        q   = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v


class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_heads * head_size, EMBED_SIZE)
        self.drop  = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
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
    def __init__(self):
        super().__init__()
        head_size = EMBED_SIZE // N_HEADS
        self.sa   = MultiHead(N_HEADS, head_size)
        self.ff   = FeedForward()
        self.ln1  = nn.LayerNorm(EMBED_SIZE)
        self.ln2  = nn.LayerNorm(EMBED_SIZE)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_SIZE)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_SIZE)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln      = nn.LayerNorm(EMBED_SIZE)
        self.head    = nn.Linear(EMBED_SIZE, vocab_size)

    def forward(self, idx, targets=None):
        B, T   = idx.shape
        tok    = self.tok_emb(idx)
        pos    = self.pos_emb(torch.arange(T))
        x      = self.blocks(tok + pos)
        x      = self.ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, n_chars, temperature=1.0):
        for _ in range(n_chars):
            idx_crop  = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_crop)
            logits    = logits[:, -1, :] / temperature
            probs     = F.softmax(logits, dim=-1)
            next_idx  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat((idx, next_idx), dim=1)
        return idx


# ============================================
# STEP 4 — Load trained weights
# ============================================
model = TinyLLM()
model.load_state_dict(torch.load("weights_torch.pt", map_location='cpu'))
model.eval()   # puts model in inference mode (disables dropout)
print("PyTorch model loaded successfully!")


# ============================================
# STEP 5 — Helper: generate a response
# ============================================
def generate_response(seed_text, n_chars=150, temperature=1.0):
    """
    Takes seed text, runs the model, returns generated continuation.
    """
    # Filter to known characters only
    safe_seed = ''.join(c for c in seed_text if c in c2i)
    if not safe_seed:
        safe_seed = "Stock"

    # Encode seed to token IDs
    token_ids = encode(safe_seed)
    idx       = torch.tensor([token_ids], dtype=torch.long)

    # Generate with no gradient tracking (faster, less memory)
    with torch.no_grad():
        output = model.generate(idx, n_chars=n_chars, temperature=temperature)

    # Decode only the NEW characters (skip the seed)
    generated_ids = output[0].tolist()[len(token_ids):]
    return decode(generated_ids)


# ============================================
# ROUTES
# ============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":     "ok",
        "vocab_size": vocab_size,
        "model":      "TinyLLM PyTorch"
    })


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({ "error": "Missing 'message' in request body" }), 400

    message     = data.get('message', '')
    temperature = float(data.get('temperature', 1.0))
    n_chars     = int(data.get('n_chars', 150))

    print(f"Received: '{message}'")

    response = generate_response(message, n_chars=n_chars, temperature=temperature)

    print(f"Generated: '{response[:60]}...'")

    return jsonify({
        "response": response,
        "seed":     message
    })


# ============================================
# START
# ============================================
if __name__ == '__main__':
    print("\nFlask LLM server running at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)