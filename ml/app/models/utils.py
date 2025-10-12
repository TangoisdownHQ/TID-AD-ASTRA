# app/models/utils.py
import json
import joblib
from pathlib import Path
from app.system.selfaware import AWARENESS_FILE

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"
REGISTRY_FILE = ARTIFACT_DIR / "registry.json"

def load_latest_model():
    """Load the latest trained model from registry."""
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError("No model registry found — train a model first.")
    with open(REGISTRY_FILE, "r") as f:
        registry = json.load(f)
    if not registry:
        raise ValueError("Registry is empty — train a model first.")
    latest_entry = registry[-1]
    model_path = Path(latest_entry["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    return model, latest_entry

def get_feature_names():
    """Return list of model feature names from awareness state."""
    if not AWARENESS_FILE.exists():
        return []
    with open(AWARENESS_FILE, "r") as f:
        state = json.load(f)
    return state.get("feature_names", [])

def get_model_hash():
    """Return latest model hash from awareness."""
    if not AWARENESS_FILE.exists():
        return None
    with open(AWARENESS_FILE, "r") as f:
        state = json.load(f)
    return state.get("last_model_hash")

