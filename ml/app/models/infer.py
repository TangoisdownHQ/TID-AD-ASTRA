# ml/app/models/infer.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import StandardScaler
from app.data.prep import load_kepler_dataset

MODEL_PATH = "app/models/kepler_xgb_model.pkl"
SCALER_PATH = "app/models/scaler.pkl"

def load_model():
    """Load trained model and scaler from disk."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_exoplanet(features: dict):
    """
    Make a prediction from a single JSON payload of features.
    Example input:
      {"koi_period": 12.5, "koi_prad": 1.2, "koi_impact": 0.45, "koi_insol": 250.0}
    """
    model, scaler = load_model()
    X_input = pd.DataFrame([features])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    proba = np.max(model.predict_proba(X_scaled))
    return {"prediction": str(pred), "confidence": float(proba)}

