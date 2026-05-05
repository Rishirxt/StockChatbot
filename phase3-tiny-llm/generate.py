# ============================================
# GENERATE.PY
# Uses the trained model to generate new stock text
# Run AFTER train.py has saved weights.json
# ============================================

import numpy as np
import json
from tokenizer import Tokenizer
from model import TinyLLM
from train import load_weights


def generate(model, tokenizer, seed_text, n_chars=200, temperature=1.0):
    """
    Generate new text character by character.

    seed_text   = starting text to prompt the model  e.g. "Stock prices"
    n_chars     = how many new characters to generate
    temperature = controls randomness of output
                  low  (0.5) = more predictable, repetitive
                  high (1.5) = more creative, sometimes nonsensical
                  1.0        = balanced

    How generation works:
        1. Encode seed text into token IDs
        2. Run forward pass → get probability distribution
        3. Sample next character from that distribution
        4. Append to sequence
        5. Repeat from step 2 using the growing sequence
    """
    # Encode the seed text
    token_ids = tokenizer.encode(seed_text)
    generated = seed_text

    print(f"Seed: '{seed_text}'")
    print(f"Generating {n_chars} characters...\n")
    print("─" * 50)
    print(generated, end="", flush=True)

    for _ in range(n_chars):
        # Only use the last block_size tokens (model's max context window)
        context = token_ids[-model.block_size:]

        # Forward pass → probabilities for next character
        probs = model.forward(context)

        # Apply temperature scaling
        # temperature < 1 → sharpen distribution (more confident)
        # temperature > 1 → flatten distribution (more random)
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()   # renormalize to sum to 1

        # Sample from the distribution
        # np.random.choice picks a character weighted by its probability
        next_token = np.random.choice(len(probs), p=probs)

        # Decode and print the new character
        next_char = tokenizer.decode([next_token])
        print(next_char, end="", flush=True)

        # Append to our growing sequence
        token_ids.append(next_token)
        generated += next_char

    print("\n" + "─" * 50)
    return generated


# ============================================
# MAIN — run generation with different seeds
# ============================================
if __name__ == "__main__":

    print("=== STOCK LLM TEXT GENERATION ===\n")

    # Load training text and build tokenizer
    with open("data/stocks.txt", "r") as f:
        text = f.read()

    tokenizer = Tokenizer(text)

    # Rebuild model with same settings as training
    model = TinyLLM(
        vocab_size = tokenizer.vocab_size,
        embed_size = 32,
        head_size  = 16,
        n_blocks   = 2,
        block_size = 32
    )

    # Load the trained weights from train.py
    try:
        load_weights(model)
        print("Trained weights loaded successfully!\n")
    except FileNotFoundError:
        print("⚠️  No weights.json found — run train.py first!")
        print("Running with random weights for demonstration...\n")

    print("\n" + "=" * 50)
    print("GENERATION 1 — Low temperature (focused)")
    print("=" * 50)
    generate(model, tokenizer,
             seed_text   = "Stock prices",
             n_chars     = 150,
             temperature = 0.5)

    print("\n" + "=" * 50)
    print("GENERATION 2 — High temperature (creative)")
    print("=" * 50)
    generate(model, tokenizer,
             seed_text   = "The market",
             n_chars     = 150,
             temperature = 1.2)

    print("\n" + "=" * 50)
    print("GENERATION 3 — Balanced temperature")
    print("=" * 50)
    generate(model, tokenizer,
             seed_text   = "Investors are",
             n_chars     = 150,
             temperature = 1.0)

    print("\n✅ Generation complete!")
    print("\nTip: run train.py with more steps (n_steps=500+) for better output")