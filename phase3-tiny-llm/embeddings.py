# ============================================
# EMBEDDINGS.PY
# Turns token numbers into vectors (lists of floats)
# This is Concept 2 from Phase 3
# ============================================

import numpy as np

class EmbeddingTable:
    def __init__(self, vocab_size, embed_size):
        """
        Create the embedding table.

        vocab_size  = how many unique characters (e.g. 65)
        embed_size  = how many floats per character (e.g. 32)

        The table is a matrix of shape [vocab_size × embed_size]
        Each row = the learned vector for one character

        Example with vocab_size=65, embed_size=32:
            table shape = [65 × 32]
            table[19]   = vector for character 19 ('S')
                        = [0.21, -0.54, 0.87, 0.33, ...]  32 numbers
        """
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # Initialize with small random numbers
        # Scale by 0.01 so values start small — prevents exploding gradients
        self.table = np.random.randn(vocab_size, embed_size) * 0.01

        print(f"Embedding table created: [{vocab_size} × {embed_size}]")
        print(f"Each character → vector of {embed_size} floats")

    def lookup(self, token_ids):
        """
        Look up the embedding vector(s) for one or more token IDs.

        token_ids = a list of integers e.g. [19, 45, 32, 18, 24]
        returns   = a 2D array of shape [len(token_ids) × embed_size]

        Think of this like: for each token number, grab that row from the table.
        """
        return self.table[token_ids]  # numpy lets us index with a list directly

    def get_vocab_size(self):
        return self.vocab_size

    def get_embed_size(self):
        return self.embed_size


# ============================================
# TEST — run this file directly to see it work
# ============================================
if __name__ == "__main__":
    from tokenizer import Tokenizer

    # Load text and build tokenizer
    with open("data/stocks.txt", "r") as f:
        text = f.read()

    tokenizer = Tokenizer(text)

    print("\n=== EMBEDDING TABLE TEST ===\n")

    # Create embedding table
    # vocab_size = from tokenizer, embed_size = 8 (small for testing)
    embed_table = EmbeddingTable(vocab_size=tokenizer.vocab_size, embed_size=8)

    # Encode a word
    sample = "Stock"
    token_ids = tokenizer.encode(sample)
    print(f"\nWord: '{sample}'")
    print(f"Token IDs: {token_ids}")

    # Look up embeddings
    embeddings = embed_table.lookup(token_ids)
    print(f"\nEmbedding shape: {embeddings.shape}  ← [5 chars × 8 floats]")
    print(f"\nEmbedding for 'S' (token {token_ids[0]}):")
    print(f"  {embeddings[0].round(4)}")
    print(f"\nEmbedding for 't' (token {token_ids[1]}):")
    print(f"  {embeddings[1].round(4)}")

    print("\nNotice: all values are small random numbers.")
    print("Training will turn these into meaningful vectors!")