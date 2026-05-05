# ============================================
# TRAIN.PY
# Training loop — teaches the model to predict text
# This is Concept 5 from Phase 3
# ============================================

import numpy as np
import json
import random
from tokenizer import Tokenizer
from model import TinyLLM


def cross_entropy_loss(probs, target_id):
    """
    Cross entropy loss measures how wrong the prediction is.

    probs     = probability distribution over all characters [vocab_size]
    target_id = the correct next character's token ID

    Formula: loss = -log(probability of correct character)

    If model is confident and right:  prob=0.9 → loss = -log(0.9) = 0.10  ✅ low
    If model is wrong:                prob=0.01 → loss = -log(0.01) = 4.6  ❌ high
    """
    correct_prob = probs[target_id]
    correct_prob = np.clip(correct_prob, 1e-9, 1.0)  # prevent log(0)
    return -np.log(correct_prob)


def get_batch(token_ids, block_size, batch_size=4):
    """
    Grab random chunks of text to train on.

    For each chunk:
        input  = characters 0 to block_size-1
        target = characters 1 to block_size   (shifted by 1)

    Example with block_size=5:
        full text:  [S, t, o, c, k,  , r, o, s, e]
        input:      [S, t, o, c, k]
        target:     [t, o, c, k,  ]
                     ↑ each target is the next char after its input

    We train the model to predict target[i] given input[0..i]
    """
    inputs  = []
    targets = []

    for _ in range(batch_size):
        # Pick a random starting position
        start = random.randint(0, len(token_ids) - block_size - 1)

        # Input = block of characters starting at 'start'
        inp = token_ids[start : start + block_size]

        # Target = same block but shifted 1 position forward
        tgt = token_ids[start + 1 : start + block_size + 1]

        inputs.append(inp)
        targets.append(tgt)

    return inputs, targets


def numerical_gradient(model, token_ids, target_id, param, idx, epsilon=1e-4):
    """
    Numerical gradient — estimates how much changing one weight affects the loss.

    This is a simplified version of backpropagation.
    Real backprop calculates this analytically (faster).
    Numerical gradient calculates it by brute force:

        gradient ≈ (loss(w + ε) - loss(w - ε)) / (2ε)

    We nudge the weight up, measure loss.
    We nudge the weight down, measure loss.
    The difference tells us the slope — which direction to move.
    """
    # Save original value
    original = param.flat[idx]

    # Nudge weight up
    param.flat[idx] = original + epsilon
    probs_up = model.forward(token_ids)
    loss_up  = cross_entropy_loss(probs_up, target_id)

    # Nudge weight down
    param.flat[idx] = original - epsilon
    probs_down = model.forward(token_ids)
    loss_down  = cross_entropy_loss(probs_down, target_id)

    # Restore original
    param.flat[idx] = original

    # Gradient = slope of loss at this weight
    return (loss_up - loss_down) / (2 * epsilon)


def update_embedding(model, token_ids, target_id, learning_rate=0.01):
    """
    Update the embedding table weights for the tokens in our input.
    Only updates the rows that were actually used (the input tokens).
    """
    for pos, tok in enumerate(token_ids[:-1]):
        for dim in range(min(4, model.embed_size)):  # update first 4 dims for speed
            grad = numerical_gradient(
                model, token_ids, target_id,
                model.embedding.table, tok * model.embed_size + dim
            )
            model.embedding.table[tok, dim] -= learning_rate * grad


def update_output_proj(model, token_ids, target_id, learning_rate=0.01):
    """
    Update the output projection weights.
    These directly affect which character the model predicts next.
    """
    # Only update weights connected to top few vocab items for speed
    probs = model.forward(token_ids)
    top_indices = np.argsort(probs)[-5:]

    for vocab_idx in top_indices:
        for dim in range(min(4, model.head_size)):
            grad = numerical_gradient(
                model, token_ids, target_id,
                model.output_proj, dim * model.vocab_size + vocab_idx
            )
            model.output_proj[dim, vocab_idx] -= learning_rate * grad


def save_weights(model, path="weights.json"):
    """
    Save the learned weights to a JSON file so we can load them later
    without retraining from scratch.
    """
    weights = {
        "embedding": model.embedding.table.tolist(),
        "pos_embedding": model.pos_embedding.tolist(),
        "output_proj": model.output_proj.tolist(),
        "output_bias": model.output_bias.tolist(),
    }
    with open(path, "w") as f:
        json.dump(weights, f)
    print(f"Weights saved to {path}")


def load_weights(model, path="weights.json"):
    """
    Load previously saved weights back into the model.
    """
    with open(path, "r") as f:
        weights = json.load(f)

    model.embedding.table = np.array(weights["embedding"])
    model.pos_embedding    = np.array(weights["pos_embedding"])
    model.output_proj      = np.array(weights["output_proj"])
    model.output_bias      = np.array(weights["output_bias"])
    print(f"Weights loaded from {path}")


# ============================================
# TRAINING LOOP
# ============================================
if __name__ == "__main__":

    print("=== TRAINING TINY STOCK LLM ===\n")

    # Load and encode training text
    with open("data/stocks.txt", "r") as f:
        text = f.read()

    tokenizer  = Tokenizer(text)
    token_ids  = tokenizer.encode(text)

    print(f"Training text: {len(text)} characters")
    print(f"Token IDs:     {len(token_ids)} integers")

    # Build model
    model = TinyLLM(
        vocab_size = tokenizer.vocab_size,
        embed_size = 32,
        head_size  = 16,
        n_blocks   = 2,
        block_size = 32
    )

    # Training settings
    n_steps      = 50          # number of training steps
                               # (increase to 500+ for better results, but slower)
    block_size   = 16          # characters per training example
    learning_rate = 0.05       # how big each weight update is

    print(f"\nTraining for {n_steps} steps...")
    print(f"Block size: {block_size} chars per example")
    print(f"Learning rate: {learning_rate}\n")

    losses = []

    for step in range(n_steps):
        # Pick a random position in the training text
        start      = random.randint(0, len(token_ids) - block_size - 2)
        inp        = token_ids[start : start + block_size]
        target_id  = token_ids[start + block_size]  # the next character

        # Forward pass — get current prediction
        probs = model.forward(inp)
        loss  = cross_entropy_loss(probs, target_id)
        losses.append(loss)

        # Update weights (simplified gradient descent)
        update_embedding(model, inp, target_id, learning_rate)
        update_output_proj(model, inp, target_id, learning_rate)

        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:])
            predicted_char = tokenizer.decode([np.argmax(probs)])
            actual_char    = tokenizer.decode([target_id])
            print(f"Step {step+1:3d}/{n_steps} | Loss: {avg_loss:.4f} | "
                  f"Predicted: '{predicted_char}' | Actual: '{actual_char}'")

    print(f"\nTraining complete!")
    print(f"Starting loss: {losses[0]:.4f}")
    print(f"Ending loss:   {losses[-1]:.4f}")

    if losses[-1] < losses[0]:
        print("Loss went DOWN ✅ — the model is learning!")
    else:
        print("Loss needs more steps — try increasing n_steps to 500")

    # Save the trained weights
    save_weights(model)