
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
    return jsonify({
        "name":     "Water Potability Prediction API",
        "version":  "2.0",
        "sdg":      "SDG 6 - Clean Water and Sanitation",
        "models":   ["Random Forest (threshold 0.6)", "PyTorch MLP v1 (threshold 0.7)"],
        "endpoint": "/predict",
        "method":   "POST"
    })

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
        recommended_is_nn = nn_prob > rf_prob

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
                "reason":     "Higher confidence on this specific sample",
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
