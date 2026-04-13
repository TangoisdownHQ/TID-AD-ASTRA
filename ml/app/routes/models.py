from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import numpy as np
import joblib
import json
import os
from datetime import datetime
from app.models.classifier import explain_prediction, predict, train_model
from app.schemas import PredictRequest

router = APIRouter(tags=["Models", "Registry"])

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "artifacts"
META_FILE = MODEL_DIR / "registry.json"
print("Using registry:", META_FILE)


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

    if not META_FILE.exists():
        raise HTTPException(status_code=404, detail="No registry.json found — train a model first.")

    with open(META_FILE, "r") as f:
        registry = json.load(f)

    if not registry:
        raise HTTPException(status_code=404, detail="Registry is empty — no models logged yet.")

    latest = sorted(registry, key=lambda x: x.get("created_at", ""), reverse=True)[0]
    model_path = latest["path"]

    model = joblib.load(model_path)

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
# 🧠 Explainability Endpoints (POST + GET)
# ============================================================

@router.post("/explain")
def explain(req: PredictRequest):
    """
    Primary explain endpoint for programmatic clients.
    """
    try:
        features = req.features or []
        planet_name = getattr(req, "planet_name", None)

        result = explain_prediction(features, planet_name)

        if not result.get("summary"):
            result["summary"] = (
                f"{planet_name} could not be fully analyzed due to missing model "
                "or incomplete observational data. This does not mean the planet "
                "is uninteresting — only that current datasets are limited."
            )

        result.setdefault("confidence", None)
        result.setdefault("habitability_index", None)
        result.setdefault("model", "Unavailable")
        result.setdefault("top_features", {})

        return result

    except Exception as e:
        return {
            "model": "Unavailable",
            "predicted_label": None,
            "confidence": None,
            "habitability_index": None,
            "top_features": {},
            "summary": (
                f"Analysis for {req.planet_name} is currently unavailable. "
                "This may be due to missing datasets, model artifacts, or incomplete "
                "observational parameters. The system will improve as more data is added."
            ),
            "error": str(e)
        }


@router.get("/explain")
def explain_get(planet: str = Query(None), features: str = Query("")):
    """
    Convenience GET endpoint for browser/curl usage.
    Example:
      /explain?planet=Kepler-452b&features=0,0,0,0,0,0,0,0,0,0,0,0
    """
    try:
        feature_list = [float(x) for x in features.split(",") if x.strip()] if features else []
        result = explain_prediction(feature_list, planet_name=planet)

        if not result.get("summary"):
            result["summary"] = (
                f"{planet} could not be fully analyzed due to missing model "
                "or incomplete observational data."
            )

        return result

    except Exception as e:
        return {
            "model": "Unavailable",
            "predicted_label": None,
            "confidence": None,
            "habitability_index": None,
            "top_features": {},
            "summary": (
                f"Analysis for {planet} is currently unavailable. "
                "This may be due to missing datasets or model artifacts."
            ),
            "error": str(e)
        }

@router.get("/compare")
def compare_models(
    planet_a: str = Query(..., description="First planet name"),
    planet_b: str = Query(..., description="Second planet name"),
    features: str = Query("", description="Comma-separated feature vector"),
):
    """
    Compare two planets using the latest model + habitability layer.

    Example:
      /models/compare?planet_a=Kepler-452b&planet_b=Kepler-10b&features=0,0,0,0,0,0,0,0,0,0,0,0
    """

    # ---- Parse features safely ----
    try:
        feats = [float(x) for x in features.split(",") if x.strip()] if features else []
    except Exception:
        feats = []

    # ---- Run explanations safely ----
    try:
        a = explain_prediction(feats, planet_name=planet_a)
    except Exception as e:
        a = {
            "model": "Unavailable",
            "predicted_label": None,
            "confidence": None,
            "habitability_index": None,
            "top_features": {},
            "planet_info": {},
            "summary": f"Failed to analyze {planet_a}: {str(e)}",
            "error": str(e),
        }

    try:
        b = explain_prediction(feats, planet_name=planet_b)
    except Exception as e:
        b = {
            "model": "Unavailable",
            "predicted_label": None,
            "confidence": None,
            "habitability_index": None,
            "top_features": {},
            "planet_info": {},
            "summary": f"Failed to analyze {planet_b}: {str(e)}",
            "error": str(e),
        }

    # ---- Comparison logic ----
    ha = a.get("habitability_index") or 0.0
    hb = b.get("habitability_index") or 0.0
    delta = round(ha - hb, 3)

    if ha == 0 and hb == 0:
        verdict = "Both planets lack sufficient data for a strong habitability comparison."
    elif delta > 0:
        verdict = f"{planet_a} appears more potentially habitable than {planet_b}."
    elif delta < 0:
        verdict = f"{planet_b} appears more potentially habitable than {planet_a}."
    else:
        verdict = f"{planet_a} and {planet_b} appear similarly habitable based on current data."

    # ---- Feature-level diff (if SHAP worked for either) ----
    diff_features = []
    fa = set(a.get("top_features", {}).keys())
    fb = set(b.get("top_features", {}).keys())
    diff_features = list((fa ^ fb))[:5]  # symmetric diff, capped

    return {
        "planet_a": planet_a,
        "planet_b": planet_b,
        "prediction_a": a,
        "prediction_b": b,
        "comparison": {
            "habitability_a": ha,
            "habitability_b": hb,
            "habitability_delta": delta,
            "key_differences": diff_features,
            "summary": verdict,
        }
    }

