# ============================================
# ATTENTION.PY
# The Q/K/V self-attention mechanism
# This is Concept 3 from Phase 3
# ============================================

import numpy as np

class SelfAttention:
    def __init__(self, embed_size, head_size):
        """
        Self-attention layer.

        embed_size = size of each token's vector coming in  (e.g. 32)
        head_size  = size of Q, K, V projections            (e.g. 16)

        Every token gets projected into 3 vectors:
            Q (Query)  = "what am I looking for?"
            K (Key)    = "what do I contain?"
            V (Value)  = "what do I give out?"

        These projections are learned weight matrices:
            W_q shape = [embed_size × head_size]
            W_k shape = [embed_size × head_size]
            W_v shape = [embed_size × head_size]
        """
        self.head_size = head_size

        # Weight matrices for Q, K, V — learned during training
        # Start random, get updated via backpropagation
        self.W_q = np.random.randn(embed_size, head_size) * 0.01
        self.W_k = np.random.randn(embed_size, head_size) * 0.01
        self.W_v = np.random.randn(embed_size, head_size) * 0.01

    def softmax(self, x):
        """
        Softmax turns raw scores into probabilities that sum to 1.
        Example: [1.2, 0.5, 3.1] → [0.09, 0.04, 0.87]
        We subtract max(x) first for numerical stability (prevents overflow).
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, x):
        """
        Run the attention mechanism.

        x = input embeddings, shape [seq_len × embed_size]
            seq_len   = number of characters in the input sequence
            embed_size = size of each character's vector

        Returns output of shape [seq_len × head_size]

        Step by step:
        1. Project x into Q, K, V
        2. Compute attention scores = Q × K^T
        3. Scale scores (divide by sqrt of head_size)
        4. Apply causal mask (can't look into the future)
        5. Softmax → attention weights
        6. Multiply weights by V → output
        """
        seq_len = x.shape[0]

        # Step 1: Project into Q, K, V
        # Each is shape [seq_len × head_size]
        Q = x @ self.W_q   # @ = matrix multiply in numpy
        K = x @ self.W_k
        V = x @ self.W_v

        # Step 2: Compute attention scores
        # Q @ K.T gives a [seq_len × seq_len] matrix
        # Each value = how much token i should attend to token j
        scores = Q @ K.T

        # Step 3: Scale scores
        # Divide by sqrt(head_size) to prevent scores from getting too large
        # Large scores → softmax becomes too sharp → gradients vanish
        scores = scores / np.sqrt(self.head_size)

        # Step 4: Causal mask
        # A token should only attend to PREVIOUS tokens, not future ones
        # (when predicting char 5, we cant peek at chars 6, 7, 8...)
        # We fill future positions with -infinity so softmax gives them 0
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # upper triangle = future
        scores = scores - mask * 1e9   # -1e9 ≈ -infinity

        # Step 5: Softmax → attention weights
        # Each row now sums to 1.0 — a probability distribution
        weights = self.softmax(scores)

        # Step 6: Weighted sum of Values
        # output[i] = weighted combination of all V vectors
        #           = mostly the V of the most relevant past token
        output = weights @ V

        return output, weights   # return weights too so we can inspect them


# ============================================
# TEST — run this file directly to see it work
# ============================================
if __name__ == "__main__":
    from tokenizer import Tokenizer
    from embeddings import EmbeddingTable

    with open("data/stocks.txt", "r") as f:
        text = f.read()

    tokenizer  = Tokenizer(text)

    print("\n=== ATTENTION TEST ===\n")

    embed_size = 16
    head_size  = 8
    sample     = "Stock rose"

    # Tokenize and embed the sample
    token_ids  = tokenizer.encode(sample)
    embed_table = EmbeddingTable(tokenizer.vocab_size, embed_size)
    embeddings  = embed_table.lookup(token_ids)

    print(f"Input text:        '{sample}'")
    print(f"Sequence length:    {len(token_ids)} characters")
    print(f"Embeddings shape:  {embeddings.shape}  ← [seq_len × embed_size]")

    # Run attention
    attention   = SelfAttention(embed_size, head_size)
    output, weights = attention.forward(embeddings)

    print(f"Attention output:  {output.shape}  ← [seq_len × head_size]")
    print(f"Attention weights: {weights.shape}  ← [seq_len × seq_len]")

    print(f"\nAttention weights for last character '{sample[-1]}':")
    print(f"(how much it attends to each previous character)")
    for i, (ch, w) in enumerate(zip(sample, weights[-1])):
        bar = '█' * int(w * 50)
        print(f"  '{ch}' → {w:.4f} {bar}")

    print("\nNote: weights are random now — training will make them meaningful!")