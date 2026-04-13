# ml/app/models/classifier.py
import joblib
import json
import pandas as pd
import numpy as np
import hashlib
import logging
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.data.prep import load_kepler_dataset
from pathlib import Path
from datetime import datetime
from app.system.selfaware import update_awareness_state, AWARENESS_FILE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from app.system.planet_knowledge import (
    get_planet_info,
    compute_habitability_index,
    narrative_summary,
)
from app.models.utils import load_latest_model, get_feature_names

# =========================================================
# 🧾 Logging Setup
# =========================================================
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "explain.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

def log_event(message: str):
    print(message)
    logging.info(message)


def log_debug(message: str):
    logging.info(message)


# =========================================================
# 📂 Directories
# =========================================================
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "models" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = ARTIFACT_DIR / "registry.json"


# =========================================================
# 🔍 Dataset Selection
# =========================================================
def get_latest_dataset():
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
            df = pd.read_csv(dataset_path, comment="#", on_bad_lines="skip")
            df = df.dropna(axis=0, thresh=int(0.5 * len(df.columns)))

            X = df.select_dtypes(include=[np.number]).dropna()
            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]

            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)
                update_awareness_state(label_mapping=dict(zip(le.classes_, le.transform(le.classes_))))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            update_awareness_state(feature_names=list(X.columns))

        except Exception as e:
            log_event(f"⚠️ External dataset failed: {e}")
            df, X_train, X_test, y_train, y_test, scaler, source_type, dataset_path = load_kepler_dataset()

    else:
        df, X_train, X_test, y_train, y_test, scaler, source_type, dataset_path = load_kepler_dataset()
        update_awareness_state(feature_names=list(range(X_train.shape[1])))

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
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    unique_classes = np.unique(y_train)
    average_type = "macro" if len(unique_classes) > 2 else "binary"

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average=average_type))
    prec = float(precision_score(y_test, y_pred, average=average_type))
    rec = float(recall_score(y_test, y_pred, average=average_type))

    dataset_hash = hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()[:8]

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

    registry = []
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                registry = json.load(f) or []
        except Exception:
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

    return entry


# =========================================================
# 🔮 PREDICTION
# =========================================================
def predict(features):
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)

    latest = registry[-1]
    model = joblib.load(latest["path"])

    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]
    confidence = float(np.max(model.predict_proba(X)))

    return {
        "model": latest["hash"],
        "predicted_label": int(pred),
        "confidence": confidence,
    }


def _feature_labels(count: int):
    feature_names = get_feature_names() or []
    labels = []
    for i in range(count):
        if i < len(feature_names):
            value = feature_names[i]
            labels.append(str(value))
        else:
            labels.append(f"feature_{i}")
    return labels


def _native_xgb_contributions(model, X, labels):
    booster = model.get_booster()
    contribs = booster.predict(
        xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(X.shape[1])]),
        pred_contribs=True,
        validate_features=False,
    )
    contrib_array = np.abs(np.asarray(contribs)[0][:-1])  # drop bias term
    top_idx = np.argsort(contrib_array)[::-1][:5]
    return {
        labels[i]: float(contrib_array[i])
        for i in top_idx
    }


def _status_from_habitability(habitability_index: float | None):
    if habitability_index is None:
        return "unknown"
    if habitability_index >= 0.7:
        return "habitable"
    if habitability_index >= 0.3:
        return "marginal"
    return "inhospitable"


def metadata_only_explanation(planet_name=None, error: str | None = None):
    try:
        planet_info = get_planet_info(planet_name) if planet_name else {}
    except Exception as exc:
        planet_info = {}
        error = error or str(exc)

    habitability_index = compute_habitability_index(planet_info) if planet_info else None
    summary = narrative_summary(planet_info) if planet_info else (
        f"There is limited publicly available data for {planet_name or 'this planet'} right now."
    )

    response = {
        "model": "metadata-only",
        "dataset_source": "planet_knowledge",
        "predicted_label": _status_from_habitability(habitability_index),
        "confidence": None,
        "top_features": {},
        "habitability_index": habitability_index,
        "planet_info": planet_info,
        "summary": summary,
        "reason": (
            "Returned a metadata-only analysis because the trained model was unavailable "
            "or could not explain this planet."
        ),
    }
    if error:
        response["error"] = error
    return response


# =========================================================
# 🧠 EXPLANATION
# =========================================================
def explain_prediction(features, planet_name=None):
    try:
        model, meta = load_latest_model()
    except Exception as exc:
        return metadata_only_explanation(planet_name=planet_name, error=str(exc))

    expected_n = getattr(model, "n_features_in_", len(features))
    features = (features + [0.0] * expected_n)[:expected_n]
    labels = _feature_labels(expected_n)

    # Always force numeric
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)

    try:
        pred = model.predict(X)[0]
        confidence = float(np.max(model.predict_proba(X)))
    except Exception as exc:
        return metadata_only_explanation(planet_name=planet_name, error=str(exc))

    # =========================================================
    # 🔍 Native XGBoost contributions with quiet fallback
    # =========================================================
    top_features = {}

    try:
        top_features = _native_xgb_contributions(model, X, labels)
    except Exception as e:
        log_debug(f"Native XGBoost contribution explanation failed: {e}")

        try:
            booster = model.get_booster()
            scores = booster.get_score(importance_type="gain")

            sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            top_features = {
                labels[int(k[1:])] if k.startswith("f") and int(k[1:]) < len(labels) else k: float(v)
                for k, v in sorted_feats
            }

        except Exception as e2:
            log_debug(f"XGBoost gain fallback failed: {e2}")
            top_features = {}

    # =========================================================
    # 🌍 Planet knowledge layer
    # =========================================================
    try:
        planet_info = get_planet_info(planet_name) if planet_name else {}
    except Exception as e:
        log_event(f"⚠️ Planet info lookup failed: {e}")
        planet_info = {}

    habitability_index = compute_habitability_index(planet_info)
    narrative = narrative_summary(planet_info)

    summary = (
        f"{planet_name or 'This planet'} is predicted as class {int(pred)} "
        f"with confidence {confidence:.2f}. "
        f"Top features: {', '.join(list(top_features.keys())[:3]) or 'N/A'}. "
        f"Habitability index: {habitability_index:.2f}. "
        f"{narrative}"
    )

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

# =========================================================
# 🖥️ CLI ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TID-AD-ASTRA model")
    parser.add_argument("--train", type=str, help="Path to dataset CSV")
    args = parser.parse_args()

    if args.train:
        log_event("🧠 CLI training invoked")
        log_event(f"📁 Dataset path: {args.train}")
        metrics = train_model()
        log_event(f"📊 Metrics: {metrics}")
