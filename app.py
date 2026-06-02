
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model, scaler and threshold
model     = joblib.load("water_potability_model.pkl")
scaler    = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Water Potability Prediction API",
        "version": "1.0",
        "sdg": "SDG 6 - Clean Water and Sanitation",
        "endpoint": "/predict",
        "method": "POST"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        features_sc  = scaler.transform(features)
        probability  = model.predict_proba(features_sc)[0][1]
        prediction   = int(probability >= threshold)
        return jsonify({
            "prediction":  prediction,
            "label":       "Potable" if prediction == 1 else "Non-potable",
            "confidence":  round(float(probability), 3),
            "threshold":   threshold,
            "safe":        bool(prediction == 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
