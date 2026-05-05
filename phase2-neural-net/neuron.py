# ============================================
# NEURON.PY — A Single Neuron
# ============================================

import math

# --- Activation Functions ---
def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# --- One Neuron ---
def neuron(inputs, weights, bias):
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    return weighted_sum + bias

# --- Test it ---
if __name__ == "__main__":
    stock_inputs = [0.8, 0.5, 0.9]  # price_change, volume, volatility
    weights      = [0.4, 0.3, 0.6]
    bias         = 0.1

    raw = neuron(stock_inputs, weights, bias)
    print(f"Raw neuron output:   {raw:.4f}")
    print(f"After ReLU:          {relu(raw):.4f}")
    print(f"After Sigmoid (0-1): {sigmoid(raw):.4f}")