# ml/app/models/classifier.py
import joblib
import json
import pandas as pd
import numpy as np
import hashlib
import shap
import logging
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.data.prep import load_kepler_dataset
from pathlib import Path
from datetime import datetime
from app.system.selfaware import update_awareness_state, AWARENESS_FILE  # Awareness tracker
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from app.system.planet_knowledge import get_planet_info, compute_habitability
from app.models.utils import load_latest_model, get_feature_names, get_model_hash

# =========================================================
# 🧾 Logging Setup
# =========================================================
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "explain.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

def log_event(message: str):
    """Helper to log events to both console and file."""
    print(message)
    logging.info(message)


# =========================================================
# 📂 Directories
# =========================================================
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "models" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = ARTIFACT_DIR / "registry.json"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# =========================================================
# 🔍 Dataset Selection
# =========================================================
def get_latest_dataset() -> tuple[str, Path]:
    if AWARENESS_FILE.exists():
        try:
            with open(AWARENESS_FILE, "r") as f:
                state = json.load(f)
            fetched = state.get("fetched_files", [])
            if fetched:
                latest = max(fetched, key=lambda p: Path(p).stat().st_mtime)
                latest_path = Path(latest)
                if latest_path.exists():
                    log_event(f"🧩 Using freshest dataset: {latest_path}")
                    return ("fetched", latest_path)
        except Exception as e:
            log_event(f"⚠️ Failed to parse awareness file: {e}")

    log_event("🪐 Using fallback Kepler/NASA dataset.")
    return ("kepler", None)


# =========================================================
# 🧠 TRAINING
# =========================================================
def train_model():
    source_type, dataset_path = get_latest_dataset()

    if dataset_path and dataset_path.exists():
        try:
            df = pd.read_csv(dataset_path)
            df = df.dropna(axis=0, thresh=int(0.5 * len(df.columns)))

            X = df.select_dtypes(include=[np.number])
            y = df.iloc[:, -1]

            if y.dtype == "object" or isinstance(y.iloc[0], str):
                le = LabelEncoder()
                y = le.fit_transform(y)
                label_map = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
                log_event(f"🧠 Encoded label classes: {label_map}")
                update_awareness_state(label_mapping=label_map)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            log_event(f"🧩 Training complete — model expects {X.shape[1]} features: {list(X.columns)}")
            update_awareness_state(feature_names=list(X.columns))

        except Exception as e:
            log_event(f"⚠️ Failed to load external dataset: {e}")
            df, X_train, X_test, y_train, y_test, scaler, source_type, dataset_path = load_kepler_dataset()

    else:
        data = load_kepler_dataset()
        if not isinstance(data, tuple) or len(data) < 8:
            raise ValueError("❌ Expected full dataset tuple from load_kepler_dataset, got something else.")

        df, X_train, X_test, y_train, y_test, scaler, source_type, dataset_path = data
        log_event(f"🧩 Training complete — model expects {X_train.shape[1]} features (Kepler fallback).")
        update_awareness_state(feature_names=list(X_train.columns) if hasattr(X_train, "columns") else [])

    update_awareness_state(
        last_trained_dataset=str(dataset_path),
        dataset_source=source_type,
        dataset_shape=df.shape
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    average_type = "macro" if len(np.unique(y)) > 2 else "binary"

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average=average_type))
    prec = float(precision_score(y_test, y_pred, average=average_type))
    rec = float(recall_score(y_test, y_pred, average=average_type))

    dataset_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:8]
    model_name = f"model_{dataset_hash}.joblib"
    model_path = ARTIFACT_DIR / model_name
    joblib.dump(model, model_path)

    entry = {
        "hash": dataset_hash,
        "path": str(model_path),
        "created_at": datetime.now().isoformat(),
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "dataset_source": source_type,
        "dataset_path": str(dataset_path),
    }

    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                registry = json.load(f)
            if not isinstance(registry, list):
                registry = [registry]
        except json.JSONDecodeError:
            registry = []
    else:
        registry = []

    registry.append(entry)

    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)

    update_awareness_state(
        last_model_path=str(model_path),
        last_model_metrics={"accuracy": acc, "f1": f1, "precision": prec, "recall": rec},
        last_model_hash=dataset_hash
    )

    log_event(f"✅ Model saved: {model_path}")
    log_event(f"✅ Registry updated: {REGISTRY_FILE}")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "model_hash": dataset_hash,
        "source": source_type
    }


# =========================================================
# 🔮 PREDICTION
# =========================================================
def predict(features: list[float]):
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError("❌ No registry.json found — train a model first.")

    with open(REGISTRY_FILE, "r") as f:
        registry = json.load(f)

    if not registry:
        raise ValueError("❌ Registry is empty — train a model first.")

    latest_entry = registry[-1]
    model_path = Path(latest_entry["path"])
    model_hash = latest_entry["hash"]

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = joblib.load(model_path)

    expected_features = None
    if AWARENESS_FILE.exists():
        try:
            with open(AWARENESS_FILE, "r") as f:
                state = json.load(f)
                expected_features = state.get("feature_names", None)
        except json.JSONDecodeError:
            log_event("⚠️ Awareness file unreadable, skipping feature validation.")

    if expected_features:
        expected_len = len(expected_features)
        if len(features) != expected_len:
            raise ValueError(
                f"❌ Feature shape mismatch — expected {expected_len} features "
                f"({expected_features[:5]}...) but got {len(features)}."
            )

    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]
    confidence = float(np.max(model.predict_proba(X)))

    return {
        "model": model_hash,
        "predicted_label": int(pred),
        "confidence": confidence,
        "expected_features": expected_features or "unknown"
    }


# =========================================================
# 🧠 EXPLANATION
# =========================================================
def explain_prediction(features: list[float], planet_name: str | None = None):
    model, meta = load_latest_model()
    feature_names = get_feature_names()
    log_event("\n🚀 [Explain] Starting planetary prediction explanation...")

    # =========================================================
    # 🧩 Validate and align input
    # =========================================================
    expected_n = getattr(model, "n_features_in_", len(features))
    if len(features) < expected_n:
        diff = expected_n - len(features)
        log_event(f"⚠️ [Align] Padding input with {diff} zeros → expected {expected_n} features.")
        features = features + [0.0] * diff
    elif len(features) > expected_n:
        log_event(f"⚠️ [Align] Truncating {len(features) - expected_n} extra features → expected {expected_n}.")
        features = features[:expected_n]
    else:
        log_event(f"✅ [Align] Feature vector length matches model ({expected_n} features).")

    X = np.array(features).reshape(1, -1)

    # =========================================================
    # 🔮 Prediction
    # =========================================================
    log_event("🧠 [Predict] Running model inference...")
    pred = model.predict(X)[0]
    confidence = float(np.max(model.predict_proba(X)))
    log_event(f"✅ [Predict] Prediction complete — Class: {pred}, Confidence: {confidence:.3f}")

    # =========================================================
    # 🌌 SHAP Explainability (safe + robust)
    # =========================================================
    log_event("🌌 [Explain] Computing SHAP feature importances...")
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # Handle possible multi-dimensional SHAP outputs
        if isinstance(shap_values.values, list):
            shap_array = np.abs(shap_values.values[0])
        elif shap_values.values.ndim == 3:
            shap_array = np.abs(shap_values.values).mean(axis=(0, 1))  # average over classes/samples
        elif shap_values.values.ndim == 2:
            shap_array = np.abs(shap_values.values[0])
        else:
            shap_array = np.abs(shap_values.values)

        feature_importances = np.array(shap_array).flatten()

        # Sanity check for feature name alignment
        if not feature_names or len(feature_names) != len(feature_importances):
            log_event("⚠️ [Explain] Feature name/importance mismatch — using generic indices.")
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]

        top_idx = np.argsort(feature_importances)[::-1][:5]
        top_features = {
            str(feature_names[int(i)]): float(feature_importances[int(i)]) for i in top_idx
        }

        log_event(f"🔬 [Explain] Top features: {list(top_features.keys())}")
    except Exception as e:
        log_event(f"❌ [Explain] SHAP computation failed: {e}")
        top_features = {}

    # =========================================================
    # 🌍 Planetary Metadata
    # =========================================================
    log_event("🪐 [Planet] Retrieving planetary metadata...")

    try:
        planet_info = get_planet_info(planet_name) if planet_name else {}
        if not isinstance(planet_info, dict) or planet_info is None:
            planet_info = {}
    except Exception as e:
        log_event(f"❌ [Planet] Error retrieving metadata: {e}")
        planet_info = {}

    if planet_info:
        log_event(f"✅ [Planet] Found data for {planet_info.get('planet_name', planet_name)}.")
    else:
        log_event(f"⚠️ [Planet] No metadata found for {planet_name} — continuing with defaults.")

    distance = planet_info.get("distance_from_earth_ly", "Unknown")
    host_star = planet_info.get("host_star", {})

    
    # =========================================================
    # 🌡️ Habitability
    # =========================================================
    log_event("🧬 [Habitability] Computing habitability index...")
    try:
        temp = features[10] if len(features) > 10 else 288
        radius = features[2]
        semimajoraxis = features[4]
        ecc = features[5]
        habitability_index = compute_habitability(temp, radius, semimajoraxis, ecc)
        log_event(f"✅ [Habitability] Habitability Index: {habitability_index:.3f}")
    except Exception as e:
        log_event(f"❌ [Habitability] Failed to compute index: {e}")
        habitability_index = 0.0

    # =========================================================
    # 🧭 Contextual Reasoning
    # =========================================================
    reasons = []
    if habitability_index > 0.7:
        reasons.append("Stable orbital and thermal conditions conducive to life.")
    elif habitability_index > 0.4:
        reasons.append("Moderate habitability — potential for microbial or extremophile life.")
    else:
        reasons.append("Extreme environment; unlikely to support Earth-like life.")

    if host_star.get("temperature") and host_star["temperature"] < 6000:
        reasons.append("Host star emits balanced radiation — supports stable climates.")
    if distance != "Unknown" and isinstance(distance, (int, float)) and distance < 50:
        reasons.append("Close proximity to Earth makes observation easier.")
    if host_star.get("spectral_type"):
        reasons.append(f"Host star spectral type {host_star['spectral_type']} affects radiation balance.")

    reasoning = " ".join(reasons)
    log_event("🧩 [Reasoning] " + reasoning)

    # =========================================================
    # 🧾 Summary
    # =========================================================
    summary = (
        f"{planet_name or 'This planet'} is predicted as class {int(pred)} "
        f"with confidence {confidence:.2f}. "
        f"Top influencing factors include {', '.join(list(top_features.keys())[:3]) or 'N/A'}. "
        f"Habitability index: {habitability_index:.2f}. {reasoning}"
    )

    log_event("✅ [Explain] Explanation complete.\n")

    # =========================================================
    # 📦 Return Structured Response
    # =========================================================
    return {
        "model": meta.get("hash"),
        "dataset_source": meta.get("dataset_source"),
        "predicted_label": int(pred),
        "confidence": confidence,
        "top_features": top_features,
        "habitability_index": habitability_index,
        "planet_info": planet_info,
        "summary": summary,
    }

