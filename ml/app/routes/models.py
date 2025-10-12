from fastapi import APIRouter, HTTPException
from pathlib import Path
import numpy as np
import joblib
import json
import os
from datetime import datetime
from app.models.classifier import explain_prediction, predict, train_model
from app.schemas import PredictRequest

router = APIRouter(tags=["Models", "Registry"])

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"
META_FILE = MODEL_DIR / "registry.json"


@router.get("/artifacts")
def list_models():
    """
    List all trained model artifacts and metadata.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = sorted(MODEL_DIR.glob("*.joblib"), key=os.path.getmtime, reverse=True)
    models = [
        {
            "name": a.name,
            "path": str(a),
            "created_at": datetime.fromtimestamp(a.stat().st_mtime).isoformat(),
            "size_kb": round(a.stat().st_size / 1024, 2),
        }
        for a in artifacts
    ]
    return {"count": len(models), "models": models}


@router.get("/latest")
def get_latest_model():
    """
    Return the latest trained model info from registry.json.
    Includes path, timestamp, and performance metrics.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not META_FILE.exists():
        return {"error": "No registry.json found — train a model first."}

    try:
        with open(META_FILE, "r") as f:
            registry = json.load(f)
    except json.JSONDecodeError:
        return {"error": "Registry file is corrupted or unreadable."}

    if not registry:
        return {"error": "Registry is empty — no models logged yet."}

    latest_entry = sorted(
        registry, key=lambda x: x.get("created_at", ""), reverse=True
    )[0]

    meta = {
        "name": os.path.basename(latest_entry["path"]),
        "path": latest_entry["path"],
        "created_at": latest_entry.get("created_at", "unknown"),
        "metrics": latest_entry.get("metrics", {}),
        "hash": latest_entry.get("hash", "unknown"),
    }

    return meta


@router.post("/load")
def load_latest_model():
    """
    Load the latest model into memory and confirm it's valid.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = sorted(MODEL_DIR.glob("*.joblib"), key=os.path.getmtime, reverse=True)
    if not artifacts:
        return {"error": "No models found"}

    latest = artifacts[0]
    model = joblib.load(latest)
    return {
        "message": f"✅ Model {latest.name} loaded successfully",
        "n_features": getattr(model, "n_features_in_", "unknown"),
    }


@router.post("/predict")
def predict_exoplanet(features: dict):
    """
    Use the latest trained model to predict exoplanet classification.
    Example input:
    {
        "features": [0.23, 1.5, 0.98, ...]
    }
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load registry and latest model
    if not META_FILE.exists():
        raise HTTPException(status_code=404, detail="No registry.json found — train a model first.")

    with open(META_FILE, "r") as f:
        registry = json.load(f)

    if not registry:
        raise HTTPException(status_code=404, detail="Registry is empty — no models logged yet.")

    latest = sorted(registry, key=lambda x: x.get("created_at", ""), reverse=True)[0]
    model_path = latest["path"]

    # Load model
    model = joblib.load(model_path)

    # Validate input
    if "features" not in features:
        raise HTTPException(status_code=400, detail="Missing 'features' in request body.")
    
    X = np.array(features["features"]).reshape(1, -1)

    try:
        y_pred = model.predict(X)[0]
        y_proba = getattr(model, "predict_proba", lambda X: [[None]])(X)[0]
        confidence = float(max(y_proba)) if y_proba[0] is not None else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {
        "model": latest["hash"],
        "predicted_label": int(y_pred),
        "confidence": confidence
    }


@router.get("/lineage")
def get_model_lineage():
    """
    Returns dataset sources for each model (NASA or fallback).
    """
    if not META_FILE.exists():
        return {"error": "No registry.json found"}
    with open(META_FILE) as f:
        data = json.load(f)
    lineage = [
        {
            "hash": e["hash"],
            "dataset_source": e.get("dataset_source", "unknown"),
            "dataset_path": e.get("dataset_path", "unknown"),
            "created_at": e["created_at"]
        }
        for e in data
    ]
    return {"count": len(lineage), "lineage": lineage}


# ============================================================
# 🧠 Safe Explainability Endpoint
# ============================================================
@router.post("/explain")
def explain(req: PredictRequest):
    """
    Provide an interpretable explanation for model predictions.
    Robust to missing or malformed data, and includes reason for fallback.
    """
    try:
        features = getattr(req, "features", None)
        planet_name = getattr(req, "planet_name", None)
        missing_fields = []

        # ✅ Handle missing or invalid features gracefully
        if features is None or not isinstance(features, (list, tuple)) or not features:
            features = [0.5] * 10  # safe fallback vector
            missing_fields.append("features")

        # ✅ Ensure all floats are finite
        clean_features = []
        for x in features:
            if isinstance(x, (float, int)) and np.isfinite(x):
                clean_features.append(float(x))
            else:
                clean_features.append(0.0)
                missing_fields.append("non-finite-value")

        # ✅ Run the explanation safely
        try:
            result = explain_prediction(clean_features, planet_name)
        except Exception as e:
            result = {
                "planet": planet_name or "Unknown",
                "habitability_index": np.nan,
                "confidence": None,
                "predicted_label": 0,
                "summary": f"Explanation unavailable: {str(e)}"
            }

        # ✅ Detect missing data inside planet_info (if present)
        planet_info = result.get("planet_info", {})
        if planet_info:
            for field, value in planet_info.items():
                if value in [None, "", [], {}, np.nan]:
                    missing_fields.append(field)

        # ✅ Add reason field for context
        if missing_fields:
            result["reason"] = (
                "Missing or incomplete data for fields: "
                + ", ".join(sorted(set(missing_fields)))
                + ". Default estimates were used where possible."
            )
        else:
            result["reason"] = "All required planetary parameters available."

        # ✅ Sanitize output for JSON safety
        def sanitize(obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            else:
                return obj

        return sanitize(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability failed: {str(e)}")

