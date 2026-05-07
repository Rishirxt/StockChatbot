# ============================================
# APP.PY — Flask Server with RAG Pipeline
# RAG = Retrieval Augmented Generation
#
# Flow:
#   1. User sends question
#   2. Search knowledge base for best match
#   3a. If good match found → return the answer
#   3b. If no match → fall back to TinyLLM generation
#   4. Return response to Node.js → browser
#
# Run with: python app.py
# Listens on: http://localhost:5000
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from knowledge_base import KnowledgeBase
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "Data", "qa_pairs.json")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights_torch.pt")


# ============================================
# STEP 1 — Load Knowledge Base
# ============================================
kb = KnowledgeBase(KB_PATH)


# ============================================
# STEP 2 — Load Vocab
# ============================================
with open(VOCAB_PATH, "r") as f:
    vocab_data = json.load(f)

vocab      = vocab_data["vocab"]
vocab_size = vocab_data["vocab_size"]
c2i        = { ch: i for i, ch in enumerate(vocab) }
i2c        = { i: ch for i, ch in enumerate(vocab) }

def encode(s): return [c2i.get(c, 0) for c in s]
def decode(l): return ''.join([i2c[i] for i in l])

print(f"Vocab loaded: {vocab_size} characters")


# ============================================
# STEP 3 — Model Hyperparameters
# Must match train.py exactly
# ============================================
BLOCK_SIZE = 64
EMBED_SIZE = 64
HEAD_SIZE  = 32
N_HEADS    = 4
N_LAYERS   = 3
DROPOUT    = 0.1


def infer_model_config(state_dict):
    """
    Infer the training-time architecture from the saved weights.
    This keeps app.py aligned with the current checkpoint.
    """
    vocab_size_from_weights, embed_size = state_dict["tok_emb.weight"].shape
    block_size, _ = state_dict["pos_emb.weight"].shape

    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    head_ids = {
        int(key.split(".")[4])
        for key in state_dict
        if key.startswith("blocks.0.sa.heads.") and key.split(".")[4].isdigit()
    }

    return {
        "vocab_size": vocab_size_from_weights,
        "embed_size": embed_size,
        "block_size": block_size,
        "n_layers": max(layer_ids) + 1 if layer_ids else 0,
        "n_heads": max(head_ids) + 1 if head_ids else 1,
    }


# ============================================
# STEP 4 — PyTorch Model Classes
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
        loss   = None
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
# STEP 5 — Load Trained Weights
# ============================================
llm_available = False
model = None

try:
    state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
    config = infer_model_config(state_dict)

    BLOCK_SIZE = config["block_size"]
    EMBED_SIZE = config["embed_size"]
    N_HEADS = config["n_heads"]
    N_LAYERS = config["n_layers"]
    HEAD_SIZE = EMBED_SIZE // N_HEADS

    if config["vocab_size"] != vocab_size:
        raise RuntimeError(
            f"Checkpoint vocab size ({config['vocab_size']}) does not match vocab.json ({vocab_size})."
        )

    model = TinyLLM()
    model.load_state_dict(state_dict)
    model.eval()
    llm_available = True
    print("PyTorch LLM loaded successfully!")
except FileNotFoundError:
    print("weights_torch.pt not found — LLM fallback disabled.")
    print("RAG knowledge base will still answer known questions.")
except RuntimeError as e:
    print(f"Could not load weights_torch.pt â€” LLM fallback disabled. {e}")
    print("RAG knowledge base will still answer known questions.")


# ============================================
# STEP 6 — LLM Generation Helper
# ============================================
def llm_generate(seed_text, n_chars=120, temperature=1.0):
    """
    Generate text from the TinyLLM as a fallback
    when no knowledge base match is found.
    """
    if not llm_available:
        return "I don't have specific information about that in my knowledge base. Try asking about stocks, dividends, ETFs, inflation, or other common investment topics."

    safe_seed = ''.join(c for c in seed_text if c in c2i)
    if not safe_seed:
        safe_seed = "Stock"

    token_ids = encode(safe_seed)
    idx = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        output = model.generate(idx, n_chars=n_chars, temperature=temperature)

    generated_ids = output[0].tolist()[len(token_ids):]
    return decode(generated_ids)


# ============================================
# STEP 7 — RAG Pipeline
# The core of the upgrade — combines KB + LLM
# ============================================
def rag_pipeline(user_message, temperature=1.0):
    """
    Full RAG pipeline:

    1. Search knowledge base for best answer
    2a. High confidence match (score > 0.2) → return KB answer directly
    2b. Medium confidence (0.1-0.2) → return KB answer with note
    2c. No match (score < 0.1) → fall back to LLM generation

    Returns: (response_text, source)
    source = 'knowledge_base' or 'llm_generation'
    """

    # Step 1: Search knowledge base
    answer, score = kb.get_answer(user_message, threshold=0.10)

    print(f"KB search score: {score:.4f}")

    # Step 2a: Strong match → use KB answer directly
    if score >= 0.20:
        print(f"→ Strong KB match (score {score:.4f})")
        return answer, "knowledge_base"

    # Step 2b: Weak match — still use it but flag it
    if score >= 0.10:
        print(f"→ Weak KB match (score {score:.4f})")
        return answer, "knowledge_base"

    # Step 2c: No match → fall back to LLM
    print(f"→ No KB match — using LLM fallback")
    response = llm_generate(user_message, n_chars=120, temperature=temperature)
    return response, "llm_generation"


# ============================================
# ROUTES
# ============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":       "ok",
        "vocab_size":   vocab_size,
        "llm_loaded":   llm_available,
        "kb_size":      len(kb.qa_pairs),
        "model":        "TinyLLM + RAG"
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Main chat endpoint.
    Receives user message → runs RAG pipeline → returns response.
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({ "error": "Missing 'message' in request" }), 400

    message     = data.get('message', '').strip()
    temperature = float(data.get('temperature', 1.0))

    if not message:
        return jsonify({ "error": "Message is empty" }), 400

    print(f"\nUser: '{message}'")

    # Run the RAG pipeline
    response, source = rag_pipeline(message, temperature)

    print(f"Source: {source}")
    print(f"Response: '{response[:80]}...'")

    return jsonify({
        "response": response,
        "source":   source,      # tells UI where answer came from
        "seed":     message
    })


@app.route('/search', methods=['POST'])
def search():
    """
    Debug endpoint — shows top 3 KB matches for a query.
    Useful for testing the knowledge base.
    """
    data    = request.get_json()
    query   = data.get('query', '')
    results = kb.search(query, top_k=3, threshold=0.0)
    return jsonify({ "query": query, "results": results })


# ============================================
# START
# ============================================
if __name__ == '__main__':
    print(f"\nStockAI RAG Server running at http://localhost:5000")
    print(f"Knowledge base: {len(kb.qa_pairs)} Q&A pairs")
    print(f"LLM fallback: {'enabled' if llm_available else 'disabled'}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
