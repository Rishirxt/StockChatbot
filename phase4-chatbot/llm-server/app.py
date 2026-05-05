# ============================================
# APP.PY — Python Flask Server
# Wraps your Phase 3 LLM in a REST API
# Run with: python app.py
# Listens on: http://localhost:5000
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys

# Add parent directory so we can import Phase 3 files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenizer import Tokenizer
from model import TinyLLM
from train import load_weights

app = Flask(__name__)
CORS(app)  # allows Node.js server to call this server

# ============================================
# LOAD MODEL ON STARTUP (do this once only)
# ============================================
print("Loading model...")

with open("data/stocks.txt", "r") as f:
    text = f.read()

tokenizer = Tokenizer(text)

model = TinyLLM(
    vocab_size = tokenizer.vocab_size,
    embed_size = 32,
    head_size  = 16,
    n_blocks   = 2,
    block_size = 32
)

try:
    load_weights(model)
    print("Weights loaded successfully!")
except FileNotFoundError:
    print("No weights.json found — using random weights.")
    print("Run train.py first for better results.")

print("Model ready!\n")


# ============================================
# HELPER — Generate text from the LLM
# ============================================
def generate_response(seed_text, n_chars=120, temperature=1.0):
    """
    Takes a user message as seed text and generates a response.

    seed_text   = the user's message
    n_chars     = how many characters to generate
    temperature = randomness (0.5 = focused, 1.5 = creative)
    """
    import numpy as np

    # Only use characters the model knows
    # Filter out unknown characters from the seed
    known_chars = set(tokenizer.vocab)
    safe_seed = ''.join(c for c in seed_text if c in known_chars)

    # Fallback if seed has no known characters
    if not safe_seed:
        safe_seed = "Stock"

    token_ids = tokenizer.encode(safe_seed)
    generated = safe_seed

    for _ in range(n_chars):
        # Use last block_size tokens as context
        context = token_ids[-model.block_size:]

        # Get probability distribution for next character
        probs = model.forward(context)

        # Apply temperature
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()

        # Sample next character
        next_token = np.random.choice(len(probs), p=probs)
        next_char  = tokenizer.decode([next_token])

        token_ids.append(next_token)
        generated += next_char

    # Return only the generated part (not the seed)
    return generated[len(safe_seed):]


# ============================================
# ROUTE — Health check
# ============================================
@app.route('/health', methods=['GET'])
def health():
    """
    Simple endpoint to confirm the server is running.
    Browser or Node.js can call: GET http://localhost:5000/health
    """
    return jsonify({
        "status": "ok",
        "vocab_size": tokenizer.vocab_size,
        "model": "TinyLLM Phase 3"
    })


# ============================================
# ROUTE — Generate a response
# ============================================
@app.route('/generate', methods=['POST'])
def generate():
    """
    Main endpoint. Receives a message and returns a generated response.

    Request body:  { "message": "What is a stock?", "temperature": 1.0 }
    Response body: { "response": "...", "seed": "..." }
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({ "error": "Missing 'message' in request body" }), 400

    message     = data.get('message', '')
    temperature = float(data.get('temperature', 1.0))
    n_chars     = int(data.get('n_chars', 120))

    print(f"Received message: '{message}'")

    response = generate_response(
        seed_text   = message,
        n_chars     = n_chars,
        temperature = temperature
    )

    print(f"Generated: '{response[:50]}...'")

    return jsonify({
        "response": response,
        "seed": message
    })


# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    print("Starting Flask LLM server on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health    — check if server is running")
    print("  POST /generate  — generate text from the LLM\n")
    app.run(host='0.0.0.0', port=5000, debug=False)