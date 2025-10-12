# app/models/classifier.py
import joblib
import json
import numpy as np
import hashlib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.data.prep import load_kepler_dataset
from pathlib import Path
from datetime import datetime

# =========================================================
# 📂 Directories
# =========================================================
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "app" / "models" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = ARTIFACT_DIR / "registry.json"


def train_model():
    """
    Train a model on the Kepler/NASA dataset and update the registry.
    """
    data = load_kepler_dataset()
    if not isinstance(data, tuple) or len(data) < 8:
        raise ValueError("❌ Expected full dataset tuple from load_kepler_dataset, got something else.")

    df, X_train, X_test, y_train, y_test, scaler, source_type, source_path = data

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # =========================================================
    # 🧮 Compute Metrics
    # =========================================================
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))

    # =========================================================
    # 💾 Save Model
    # =========================================================
    dataset_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:8]
    model_name = f"model_{dataset_hash}.joblib"
    model_path = ARTIFACT_DIR / model_name
    joblib.dump(model, model_path)

    # =========================================================
    # 🗂️ Update Registry
    # =========================================================
    entry = {
        "hash": dataset_hash,
        "path": str(model_path),
        "created_at": datetime.now().isoformat(),
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        "dataset_source": source_type,
        "dataset_path": source_path,
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

    print(f"✅ Model saved: {model_path}")
    print(f"✅ Registry updated: {REGISTRY_FILE}")

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

