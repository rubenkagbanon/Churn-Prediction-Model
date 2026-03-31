"""
predict.py — API Flask pour prédictions de churn en temps réel
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.database import save_prediction

app = Flask(__name__)
CORS(app)

MODEL_PATH   = os.path.join("models", "voting_classifier.pkl")
SCALER_PATH  = os.path.join("models", "scaler.pkl")
ENCODER_PATH = os.path.join("models", "encoders.pkl")

model    = None
scaler   = None
encoders = None

FEATURES = [
    "gender", "senior_citizen", "partner", "dependents", "tenure",
    "phone_service", "multiple_lines", "internet_service", "online_security",
    "tech_support", "contract", "paperless_billing", "payment_method",
    "monthly_charges", "total_charges"
]


def load_artifacts():
    global model, scaler, encoders
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    print("[API] Artifacts chargés.")


def preprocess_input(data: dict) -> np.ndarray:
    df = pd.DataFrame([data])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    df = df[FEATURES]
    return scaler.transform(df)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "voting_classifier_v1.0"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data  = request.json
        X     = preprocess_input(data)
        proba = float(model.predict_proba(X)[0][1])
        pred  = "Churn" if proba >= 0.5 else "No Churn"

        save_prediction(
            customer_id=data.get("customer_id", "unknown"),
            churn_prob=proba,
            prediction=pred,
            model_version="v1.0"
        )

        return jsonify({
            "customer_id": data.get("customer_id", "unknown"),
            "prediction":  pred,
            "churn_probability": round(proba, 4),
            "risk_level":  "High" if proba > 0.7 else ("Medium" if proba > 0.4 else "Low")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        records = request.json.get("customers", [])
        results = []
        for record in records:
            X     = preprocess_input(record)
            proba = float(model.predict_proba(X)[0][1])
            pred  = "Churn" if proba >= 0.5 else "No Churn"
            results.append({
                "customer_id":      record.get("customer_id", "?"),
                "prediction":       pred,
                "churn_probability": round(proba, 4)
            })
        return jsonify({"results": results, "total": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import sys
    if "--serve" in sys.argv or True:
        load_artifacts()
        app.run(host="0.0.0.0", port=5000, debug=False)