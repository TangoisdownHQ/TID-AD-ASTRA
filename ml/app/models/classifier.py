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
    compute_habitability_profile,
    narrative_summary,
)
from app.models.utils import load_latest_model, get_feature_names
from app.models.features import (
    build_feature_vector,
    model_gain_weights,
    summarize_provenance,
)


FALLBACK_KOI_FEATURES = [
    "koi_period",
    "koi_period_err1",
    "koi_period_err2",
    "koi_time0bk",
    "koi_time0bk_err1",
    "koi_time0bk_err2",
    "koi_impact",
    "koi_impact_err1",
    "koi_impact_err2",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

FEATURE_DESCRIPTIONS = {
    "koi_period": "Orbital period in days. This tells how long the planet takes to orbit its star.",
    "koi_period_err1": "Upper uncertainty on orbital period. Larger uncertainty means the measured orbit is less precise.",
    "koi_period_err2": "Lower uncertainty on orbital period. This is another bound on how uncertain the orbital period is.",
    "koi_time0bk": "Reference transit time. This marks when a measured transit occurred.",
    "koi_time0bk_err1": "Upper uncertainty on the transit-timing reference point.",
    "koi_time0bk_err2": "Lower uncertainty on the transit-timing reference point.",
    "koi_impact": "Transit impact parameter. It indicates how centrally the planet crosses the face of its star.",
    "koi_impact_err1": "Upper uncertainty on the transit impact parameter.",
    "koi_impact_err2": "Lower uncertainty on the transit impact parameter.",
    "koi_duration": "Transit duration in hours. Longer or shorter crossings help characterize the orbit.",
    "koi_depth": "Transit depth in parts per million. Deeper transits usually indicate a larger planet relative to the star.",
    "koi_prad": "Planet radius in Earth radii.",
    "koi_teq": "Estimated equilibrium temperature in Kelvin, based on stellar heating rather than a directly measured surface temperature.",
    "koi_insol": "Starlight received relative to Earth. A value of 1 means the planet gets about as much energy as Earth does.",
    "koi_model_snr": "Signal-to-noise ratio of the transit detection. Higher values usually mean a cleaner, more reliable signal.",
    "koi_steff": "Host star effective temperature in Kelvin.",
    "koi_slogg": "Host star surface gravity on a logarithmic scale.",
    "koi_srad": "Host star radius in solar radii.",
    "koi_kepmag": "Kepler-band brightness of the host star. Lower values mean a brighter star.",
}

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
        update_awareness_state(feature_names=_derive_feature_names_from_dataframe(df))

    # Record the exact feature space the model was fitted on. The medians are what
    # inference falls back to for inputs a planet record cannot supply, so they
    # have to come from the same frame the model saw.
    if hasattr(X_train, "columns"):
        update_awareness_state(
            feature_names=[str(c) for c in X_train.columns],
            feature_medians={
                str(column): float(value)
                for column, value in X_train.median(numeric_only=True).items()
                if pd.notna(value)
            },
        )

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
    if _feature_names_are_generic(feature_names):
        feature_names = FALLBACK_KOI_FEATURES
    labels = []
    for i in range(count):
        if i < len(feature_names):
            value = feature_names[i]
            labels.append(str(value))
        else:
            labels.append(f"feature_{i}")
    return labels


def _feature_names_are_generic(feature_names) -> bool:
    if not feature_names:
        return True
    normalized = [str(item).strip().lower() for item in feature_names]
    return all(name.isdigit() or name.startswith("feature_") for name in normalized)


def _derive_feature_names_from_dataframe(df: pd.DataFrame) -> list[str]:
    working = df.copy()
    working.columns = [c.strip() for c in working.columns]
    working = working.replace([np.inf, -np.inf], np.nan)
    working = working.dropna(axis=1, how="all")

    label_col = None
    if "koi_disposition" in working.columns:
        label_col = "koi_disposition"
    elif "koi_pdisposition" in working.columns:
        label_col = "koi_pdisposition"

    feature_names = []
    for col in working.columns:
        if col == label_col:
            continue
        coerced = pd.to_numeric(working[col], errors="coerce")
        if coerced.notna().sum() > 0:
            feature_names.append(col)

    feature_names = [col for col in feature_names if col not in {"kepid", "ra", "dec"}]
    X = working[feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]
    return list(X.columns)


def describe_feature(feature_name: str) -> dict:
    label = str(feature_name)
    description = FEATURE_DESCRIPTIONS.get(label)
    if description:
        return {"label": label, "description": description}
    return {
        "label": label,
        "description": "This is one of the numeric inputs used by the classifier. Its importance shows relative influence on this prediction, not a direct percentage.",
    }


def describe_predicted_class(predicted_label) -> str:
    if predicted_label in (1, "1"):
        return (
            "Class 1 means the transit signal looks like a real planet detection rather "
            "than a false positive. It says nothing about habitability. Note that the "
            "catalog already excludes vetted false positives, so almost everything you "
            "can browse here scores class 1 — a high score confirms the detection is "
            "sound, it does not distinguish one planet from another."
        )
    if predicted_label in (0, "0"):
        return (
            "Class 0 means the signal resembles a false positive — an eclipsing binary or "
            "instrumental artefact rather than a planet. This is unusual for a catalog "
            "entry and worth treating as a data-quality flag."
        )
    return "The class is the model's category label. It is not itself a habitability grade."


PREDICTION_CAVEAT = (
    "This classifier answers 'is this a genuine detection?', not 'is this habitable?'. "
    "Use the habitability breakdown for the second question."
)


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


def metadata_only_explanation(planet_name=None, error: str | None = None):
    try:
        planet_info = get_planet_info(planet_name) if planet_name else {}
    except Exception as exc:
        planet_info = {}
        error = error or str(exc)

    profile = compute_habitability_profile(planet_info)
    habitability_index = profile["score"]
    summary = narrative_summary(planet_info) if planet_info else (
        f"There is limited publicly available data for {planet_name or 'this planet'} right now."
    )

    response = {
        "model": "metadata-only",
        "dataset_source": "planet_knowledge",
        "predicted_label": profile["status"],
        "confidence": None,
        "top_features": {},
        "habitability_index": habitability_index,
        "habitability_coverage": profile["coverage"],
        "habitability_status": profile["status"],
        "habitability_factors": profile["factors"],
        "habitability_explanation": profile["explanation"],
        "model_inputs": {
            "inputs_total": 0,
            "inputs_from_planet": 0,
            "planet_features": [],
            "coverage": 0.0,
            "basis": "no trained model was available, so this is catalog metadata only",
        },
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

    expected_n = getattr(model, "n_features_in_", len(features or []))
    labels = _feature_labels(expected_n)

    # =========================================================
    # 🌍 Planet knowledge layer (needed before building inputs)
    # =========================================================
    try:
        planet_info = get_planet_info(planet_name) if planet_name else {}
    except Exception as e:
        log_event(f"⚠️ Planet info lookup failed: {e}")
        planet_info = {}

    # Build the model input from the planet itself. Callers may still pass an
    # explicit vector, but an empty or padded one would make every planet look
    # identical to the model, so the catalog record takes precedence.
    if features:
        features = (list(features) + [0.0] * expected_n)[:expected_n]
        provenance = [
            {"feature": label, "value": value, "origin": "caller", "catalog_field": None}
            for label, value in zip(labels, features)
        ]
    else:
        features, provenance = build_feature_vector(planet_info, labels)

    input_summary = summarize_provenance(provenance, model_gain_weights(model, labels))

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
    # 🌍 Habitability + narrative
    # =========================================================
    profile = compute_habitability_profile(planet_info)
    habitability_index = profile["score"]
    narrative = narrative_summary(planet_info)

    habitability_text = (
        f"Habitability index: {habitability_index:.2f} "
        f"({int(profile['coverage'] * 100)}% of inputs measured). "
        if habitability_index is not None
        else "Habitability index: not enough measured data to score. "
    )

    summary = (
        f"{planet_name or 'This planet'} is predicted as class {int(pred)} "
        f"with confidence {confidence:.3f}. "
        f"Top features: {', '.join(list(top_features.keys())[:3]) or 'N/A'}. "
        f"{habitability_text}"
        f"Prediction basis: {input_summary['basis']}. "
        f"{narrative}"
    )

    return {
        "model": meta.get("hash"),
        "dataset_source": meta.get("dataset_source"),
        "predicted_label": int(pred),
        "predicted_class_explanation": describe_predicted_class(int(pred)),
        "prediction_caveat": PREDICTION_CAVEAT,
        "confidence": confidence,
        "top_features": top_features,
        "top_feature_details": [describe_feature(name) for name in top_features.keys()],
        "habitability_index": habitability_index,
        "habitability_coverage": profile["coverage"],
        "habitability_status": profile["status"],
        "habitability_factors": profile["factors"],
        "habitability_explanation": profile["explanation"],
        "model_inputs": input_summary,
        "model_input_details": provenance,
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
