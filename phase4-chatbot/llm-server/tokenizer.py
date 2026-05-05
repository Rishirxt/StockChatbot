# ============================================
# TOKENIZER.PY
# Converts text → numbers and numbers → text
# This is Concept 1 from Phase 3
# ============================================

class Tokenizer:
    def __init__(self, text):
        """
        Build vocabulary from the training text.
        Vocabulary = every unique character found in the text.
        """

        # Step 1: Find every unique character in the text
        # sorted() makes it consistent every run
        self.vocab = sorted(set(text))

        # Step 2: Vocabulary size = how many unique characters exist
        self.vocab_size = len(self.vocab)

        # Step 3: Build lookup tables
        # char_to_int: turn a character into a number  e.g. 'a' → 3
        # int_to_char: turn a number back into a char  e.g. 3 → 'a'
        self.char_to_int = { ch: i for i, ch in enumerate(self.vocab) }
        self.int_to_char = { i: ch for i, ch in enumerate(self.vocab) }

        print(f"Vocabulary size: {self.vocab_size} unique characters")
        print(f"Vocabulary: {''.join(self.vocab)}")

    def encode(self, text):
        """
        Convert a string of text into a list of integers.
        Example: "Stock" → [19, 45, 32, 18, 24]
        """
        return [self.char_to_int[ch] for ch in text]

    def decode(self, integers):
        """
        Convert a list of integers back into a string.
        Example: [19, 45, 32, 18, 24] → "Stock"
        """
        return ''.join([self.int_to_char[i] for i in integers])


# ============================================
# TEST — run this file directly to see it work
# ============================================
if __name__ == "__main__":

    # Load the training text
    with open("data/stocks.txt", "r") as f:
        text = f.read()

    print("=== TOKENIZER TEST ===\n")

    # Build tokenizer from text
    tokenizer = Tokenizer(text)

    # Test encoding
    sample = "Stock prices rose"
    encoded = tokenizer.encode(sample)
    print(f"\nOriginal text:  '{sample}'")
    print(f"Encoded:         {encoded}")

    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded back:   '{decoded}'")

    # Show a few character mappings
    print(f"\nSample mappings:")
    for ch in ['S', 't', 'o', 'c', 'k', ' ']:
        print(f"  '{ch}' → {tokenizer.char_to_int[ch]}")