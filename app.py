
import joblib
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# Load Random Forest
rf_model     = joblib.load("water_potability_model.pkl")
scaler       = joblib.load("scaler.pkl")
rf_threshold = joblib.load("threshold.pkl")

# Define PyTorch architecture
class WaterMLP_v1(nn.Module):
    def __init__(self):
        super(WaterMLP_v1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# Load PyTorch model
nn_model = WaterMLP_v1()
nn_model.load_state_dict(torch.load("pytorch_model.pth", map_location=torch.device("cpu")))
nn_model.eval()
nn_threshold = 0.7

@app.route("/", methods=["GET"])
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Water Potability Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; background: #f0f8ff; }
        h1 { color: #1A5F7A; }
        .sdg { color: #2E9E6B; font-weight: bold; }
        input { width: 80px; padding: 5px; margin: 5px; border: 1px solid #ccc; border-radius: 4px; }
        button { background: #1A5F7A; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        button:hover { background: #2E9E6B; }
        .result { margin-top: 20px; padding: 15px; border-radius: 8px; display: none; }
        .safe { background: #d4edda; border: 1px solid #2E9E6B; }
        .unsafe { background: #f8d7da; border: 1px solid #e05252; }
        .label { font-size: 24px; font-weight: bold; }
        .detail { margin-top: 10px; font-size: 14px; color: #555; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        td { padding: 5px 10px; border-bottom: 1px solid #eee; }
    </style>
</head>
<body>
    <h1>Water Potability Prediction</h1>
    <p class="sdg">SDG 6: Clean Water and Sanitation</p>
    <p>Enter 9 water quality measurements to predict potability.</p>

    <table>
        <tr><td>pH</td><td><input type="number" id="ph" step="0.01" placeholder="7.2"></td></tr>
        <tr><td>Hardness</td><td><input type="number" id="hardness" step="0.01" placeholder="204.0"></td></tr>
        <tr><td>Solids</td><td><input type="number" id="solids" step="0.01" placeholder="20791.0"></td></tr>
        <tr><td>Chloramines</td><td><input type="number" id="chloramines" step="0.01" placeholder="7.3"></td></tr>
        <tr><td>Sulfate</td><td><input type="number" id="sulfate" step="0.01" placeholder="368.0"></td></tr>
        <tr><td>Conductivity</td><td><input type="number" id="conductivity" step="0.01" placeholder="564.0"></td></tr>
        <tr><td>Organic Carbon</td><td><input type="number" id="organic_carbon" step="0.01" placeholder="10.3"></td></tr>
        <tr><td>Trihalomethanes</td><td><input type="number" id="trihalomethanes" step="0.01" placeholder="86.0"></td></tr>
        <tr><td>Turbidity</td><td><input type="number" id="turbidity" step="0.01" placeholder="2.96"></td></tr>
    </table>

    <button onclick="predict()">Predict</button>

    <div class="result" id="result">
        <div class="label" id="label"></div>
        <div class="detail" id="detail"></div>
    </div>

    <script>
        async function predict() {
            const features = [
                parseFloat(document.getElementById("ph").value),
                parseFloat(document.getElementById("hardness").value),
                parseFloat(document.getElementById("solids").value),
                parseFloat(document.getElementById("chloramines").value),
                parseFloat(document.getElementById("sulfate").value),
                parseFloat(document.getElementById("conductivity").value),
                parseFloat(document.getElementById("organic_carbon").value),
                parseFloat(document.getElementById("trihalomethanes").value),
                parseFloat(document.getElementById("turbidity").value)
            ];

            if (features.some(isNaN)) {
                alert("Please fill in all 9 fields.");
                return;
            }

            const response = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({features: features})
            });

            const data = await response.json();
            const result = document.getElementById("result");
            const label = document.getElementById("label");
            const detail = document.getElementById("detail");

            result.style.display = "block";
            if (data.recommended.safe) {
                result.className = "result safe";
                label.innerHTML = "✅ POTABLE (Safe to drink)";
            } else {
                result.className = "result unsafe";
                label.innerHTML = "❌ NON-POTABLE (Not safe to drink)";
            }

            detail.innerHTML = `
                <b>Recommended model:</b> ${data.recommended.model}<br>
                <b>Neural Network:</b> ${data.neural_network.label} (confidence: ${data.neural_network.confidence}, threshold: ${data.neural_network.threshold})<br>
                <b>Random Forest:</b> ${data.random_forest.label} (confidence: ${data.random_forest.confidence}, threshold: ${data.random_forest.threshold})
            `;
        }
    </script>
</body>
</html>
"""
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": True,
        "nn_threshold":  nn_threshold,
        "rf_threshold":  float(rf_threshold)
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Input validation
        if features.shape[1] != 9:
            return jsonify({"error": "Expected exactly 9 features"}), 400

        features_sc = scaler.transform(features)

        # Random Forest prediction
        rf_prob  = rf_model.predict_proba(features_sc)[0][1]
        rf_pred  = int(rf_prob >= rf_threshold)

        # PyTorch prediction
        tensor   = torch.FloatTensor(features_sc)
        with torch.no_grad():
            nn_prob = nn_model(tensor).squeeze().item()
        nn_pred  = int(nn_prob >= nn_threshold)

        # Dynamic recommendation
        nn_confidence = abs(nn_prob - 0.5)
        rf_confidence = abs(rf_prob - 0.5)
        recommended_is_nn = nn_confidence > rf_confidence

        result = {
            "random_forest": {
                "prediction": rf_pred,
                "label":      "Potable" if rf_pred == 1 else "Non-potable",
                "confidence": round(float(rf_prob), 3),
                "threshold":  rf_threshold,
                "safe":       bool(rf_pred == 1)
            },
            "neural_network": {
                "prediction": nn_pred,
                "label":      "Potable" if nn_pred == 1 else "Non-potable",
                "confidence": round(float(nn_prob), 3),
                "threshold":  nn_threshold,
                "safe":       bool(nn_pred == 1)
            },
            "recommended": {
                "model":      "Neural Network (PyTorch v1)" if recommended_is_nn else "Random Forest",
                "reason": "Higher certainty on this specific sample (distance from 0.5)",
                "prediction": nn_pred if recommended_is_nn else rf_pred,
                "label":      ("Potable" if nn_pred == 1 else "Non-potable") if recommended_is_nn else ("Potable" if rf_pred == 1 else "Non-potable"),
                "safe":       bool(nn_pred == 1) if recommended_is_nn else bool(rf_pred == 1)
            }
        }

        # Log prediction
        logging.info(
            f"Input: {features.tolist()} | "
            f"NN: {result['neural_network']['label']} ({result['neural_network']['confidence']}) | "
            f"RF: {result['random_forest']['label']} ({result['random_forest']['confidence']}) | "
            f"Recommended: {result['recommended']['model']}"
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
