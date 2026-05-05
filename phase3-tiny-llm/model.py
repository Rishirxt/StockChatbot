# ============================================
# MODEL.PY
# Full Tiny Transformer — stacks all layers together
# This is Concept 4 from Phase 3
# ============================================

import numpy as np
from tokenizer import Tokenizer
from embeddings import EmbeddingTable
from attention import SelfAttention


class FeedForward:
    """
    A simple two-layer neural network (same as Phase 2!)
    Applied to each token independently after attention.

    Structure:
        input [embed_size] → hidden [embed_size × 4] → output [embed_size]
    """
    def __init__(self, size):
        self.W1 = np.random.randn(size, size * 4) * 0.01
        self.W2 = np.random.randn(size * 4, size) * 0.01
        self.b1 = np.zeros(size * 4)
        self.b2 = np.zeros(size)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # x shape: [seq_len × size]
        hidden = self.relu(x @ self.W1 + self.b1)   # expand + activate
        return hidden @ self.W2 + self.b2            # contract back


class LayerNorm:
    """
    Normalizes the values in each token's vector.
    Keeps numbers from exploding or vanishing as they flow through layers.
    """
    def __init__(self, size, eps=1e-5):
        self.gamma = np.ones(size)
        self.beta  = np.zeros(size)
        self.eps   = eps

    def forward(self, x):
        mean       = x.mean(axis=-1, keepdims=True)
        std        = x.std(axis=-1, keepdims=True)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta


class TransformerBlock:
    """
    One full transformer block:
        1. Self-attention         (tokens look at each other)
        2. Project back to embed_size (head_size → embed_size)
        3. Add & Normalize        (residual connection + layer norm)
        4. Feed forward           (process each token independently)
        5. Add & Normalize        (residual connection + layer norm)
    """
    def __init__(self, embed_size, head_size):
        self.attention = SelfAttention(embed_size, head_size)
        self.ff        = FeedForward(embed_size)
        self.norm1     = LayerNorm(embed_size)
        self.norm2     = LayerNorm(embed_size)

        # Projects attention output from head_size → embed_size
        # so the residual addition shapes match
        self.proj = np.random.randn(head_size, embed_size) * 0.01

    def forward(self, x):
        # Step 1: Attention
        attn_out, _ = self.attention.forward(x)

        # Step 2: Project attn_out from head_size → embed_size
        attn_projected = attn_out @ self.proj       # [seq_len × embed_size]

        # Step 3: Residual add + normalize (shapes match now ✅)
        x = self.norm1.forward(x + attn_projected)

        # Step 4: Feed forward
        ff_out = self.ff.forward(x)

        # Step 5: Residual add + normalize
        x = self.norm2.forward(x + ff_out)

        return x


class TinyLLM:
    """
    The full tiny language model.

    Architecture:
        Input token IDs
            ↓
        Token Embeddings     [vocab_size → embed_size]
            ↓
        Positional Encoding  [adds position info to each token]
            ↓
        Transformer Block 1  [attention + feedforward]
            ↓
        Transformer Block 2  [attention + feedforward]
            ↓
        Output projection    [embed_size → vocab_size]
            ↓
        Softmax              [probabilities for each next character]
    """
    def __init__(self, vocab_size, embed_size=32, head_size=16, n_blocks=2, block_size=64):
        self.vocab_size  = vocab_size
        self.embed_size  = embed_size
        self.head_size   = head_size
        self.block_size  = block_size

        # Token embedding table
        self.embedding = EmbeddingTable(vocab_size, embed_size)

        # Positional encoding — tells the model WHERE each token is
        self.pos_embedding = np.random.randn(block_size, embed_size) * 0.01

        # Stack of transformer blocks
        self.blocks = [TransformerBlock(embed_size, head_size) for _ in range(n_blocks)]

        # Output projection: maps embed_size → vocab_size
        self.output_proj = np.random.randn(embed_size, vocab_size) * 0.01
        self.output_bias = np.zeros(vocab_size)

        print(f"\nTiny LLM created!")
        print(f"  Vocab size:    {vocab_size}")
        print(f"  Embed size:    {embed_size}")
        print(f"  Head size:     {head_size}")
        print(f"  Blocks:        {n_blocks}")
        print(f"  Block size:    {block_size} (max sequence length)")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, token_ids):
        """
        Full forward pass through the model.

        token_ids = list of integers e.g. [19, 45, 32, 18]
        returns   = probability distribution over vocabulary for next token
        """
        seq_len = len(token_ids)

        # Step 1: Token embeddings — shape [seq_len × embed_size]
        token_emb = self.embedding.lookup(token_ids)

        # Step 2: Add positional encoding
        pos_emb = self.pos_embedding[:seq_len]
        x = token_emb + pos_emb       # shape: [seq_len × embed_size]

        # Step 3: Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Step 4: Take only the LAST token's output
        last_token = x[-1]            # shape: [embed_size]

        # Step 5: Project to vocab size to get raw scores (logits)
        logits = last_token @ self.output_proj + self.output_bias  # [vocab_size]

        # Step 6: Softmax → probability for each character
        probs = self.softmax(logits)

        return probs

    def predict_next(self, token_ids):
        """
        Given a sequence of token IDs, predict the most likely next token.
        """
        probs      = self.forward(token_ids)
        next_token = np.argmax(probs)
        return next_token, probs


# ============================================
# TEST — run this file directly to see it work
# ============================================
if __name__ == "__main__":

    with open("data/stocks.txt", "r") as f:
        text = f.read()

    tokenizer = Tokenizer(text)

    print("\n=== FULL MODEL TEST ===")

    # Build the model
    model = TinyLLM(
        vocab_size = tokenizer.vocab_size,
        embed_size = 32,
        head_size  = 16,
        n_blocks   = 2,
        block_size = 64
    )

    # Test forward pass
    sample    = "Stock prices"
    token_ids = tokenizer.encode(sample)

    print(f"\nInput: '{sample}'")
    print(f"Token IDs: {token_ids}")

    next_token, probs = model.predict_next(token_ids)
    next_char = tokenizer.decode([next_token])

    print(f"\nPredicted next character: '{next_char}'")
    print(f"Model confidence: {probs[next_token]:.4f}")

    # Show top 5 predictions
    top5 = np.argsort(probs)[-5:][::-1]
    print(f"\nTop 5 predictions:")
    for idx in top5:
        ch = tokenizer.decode([idx])
        print(f"  '{ch}' → {probs[idx]:.4f}")

    print("\nNote: predictions are random now — train.py will fix this!")