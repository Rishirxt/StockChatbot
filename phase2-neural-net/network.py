# ============================================
# NETWORK.PY — Tiny Neural Network
# Stock Risk Predictor
# ============================================

from neuron import neuron, relu, sigmoid

# --- The Network's Weights (random to start — learned during training) ---
network = {
    "hidden": [
        { "weights": [0.4, 0.3, 0.6], "bias": 0.1 },  # neuron 1
        { "weights": [0.2, 0.8, 0.1], "bias": 0.0 },  # neuron 2
        { "weights": [0.7, 0.1, 0.5], "bias": 0.2 },  # neuron 3
    ],
    "output": { "weights": [0.5, 0.4, 0.6], "bias": 0.1 }
}

# --- Forward Pass ---
# Data flows: inputs → hidden layer → output layer → risk score
def forward_pass(inputs):
    # Step 1: Pass inputs through each hidden neuron + apply ReLU
    hidden_outputs = [
        relu(neuron(inputs, n["weights"], n["bias"]))
        for n in network["hidden"]
    ]

    # Step 2: Pass hidden outputs through output neuron + apply Sigmoid
    raw = neuron(hidden_outputs, network["output"]["weights"], network["output"]["bias"])
    return sigmoid(raw)  # returns a score between 0 and 1

# --- Loss Function (Mean Squared Error) ---
# Measures how wrong the prediction is
def loss(predicted, actual):
    return (predicted - actual) ** 2

# --- Test with 3 stocks ---
if __name__ == "__main__":
    high_risk_stock   = [0.9, 0.8, 0.95]  # high change, volume, volatility
    low_risk_stock    = [0.1, 0.2, 0.05]  # stable, quiet stock
    medium_risk_stock = [0.5, 0.5, 0.50]  # average stock

    print("============================================")
    print("   STOCK RISK NEURAL NETWORK")
    print("   Inputs: [price_change, volume, volatility]")
    print("============================================\n")

    print("--- High Risk Stock [0.9, 0.8, 0.95] ---")
    high_pred = forward_pass(high_risk_stock)
    print(f"Risk Score: {high_pred:.4f}")
    print(f"Loss vs actual 0.9: {loss(high_pred, 0.9):.4f}")

    print("\n--- Low Risk Stock [0.1, 0.2, 0.05] ---")
    low_pred = forward_pass(low_risk_stock)
    print(f"Risk Score: {low_pred:.4f}")
    print(f"Loss vs actual 0.1: {loss(low_pred, 0.1):.4f}")

    print("\n--- Medium Risk Stock [0.5, 0.5, 0.5] ---")
    med_pred = forward_pass(medium_risk_stock)
    print(f"Risk Score: {med_pred:.4f}")
    print(f"Loss vs actual 0.5: {loss(med_pred, 0.5):.4f}")

    print("\n============================================")
    print("NOTE: Scores are inaccurate because weights")
    print("are random. Training will fix this in Phase 3!")
    print("============================================")