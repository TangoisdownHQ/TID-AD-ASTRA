# ml/app/system/auto_retrain.py
import json
from pathlib import Path
from datetime import datetime
from app.data.prep import load_kepler_dataset
from app.models.classifier import train_model
from app.system.selfaware import compute_dataset_hash, log_dataset_state, update_awareness_state
from app.system.model_drift import detect_model_drift

AWARENESS_FILE = Path(__file__).resolve().parent / "awareness_state.json"


def auto_retrain():
    """
    Automatically retrain the model if the dataset has changed.
    Detects drift by comparing dataset hashes and retrains when necessary.
    Returns a result dict for the scheduler and awareness log.
    """
    print("🧠 Checking for dataset drift or updates...")

    # =========================================================
    # 🧩 Load previous awareness state
    # =========================================================
    prev_state = {}
    if AWARENESS_FILE.exists():
        try:
            with open(AWARENESS_FILE, "r") as f:
                prev_state = json.load(f)
            if not isinstance(prev_state, dict):
                prev_state = {}
        except json.JSONDecodeError:
            prev_state = {}
    else:
        print("⚠️ No previous awareness state found — first run.")

    prev_hash = prev_state.get("last_dataset_hash", None)
    prev_source = prev_state.get("dataset_source", "unknown")

    # =========================================================
    # 🚀 Load latest dataset
    # =========================================================
    try:
        df, X_train, X_test, y_train, y_test, scaler, source_type, source_path = load_kepler_dataset()
        current_hash = compute_dataset_hash(df)
    except Exception as e:
        print(f"❌ Failed to load dataset for auto-retrain: {e}")
        update_awareness_state(
            last_auto_retrain_status="failed",
            last_auto_retrain_error=str(e),
            last_auto_retrain_time=datetime.now().isoformat(timespec="seconds"),
        )
        return {"status": "failed", "error": str(e)}

    # =========================================================
    # 🧠 Detect drift or new data
    # =========================================================
    drift_detected = False
    drift_reason = None

    try:
        drift_detected, drift_reason = detect_model_drift(prev_hash, current_hash)
    except Exception:
        # Fallback to simple hash comparison if drift module fails
        if prev_hash and current_hash and prev_hash != current_hash:
            drift_detected = True
            drift_reason = "hash_mismatch"
        else:
            drift_detected = False
            drift_reason = "no_change"

    # =========================================================
    # 🔍 Log dataset state
    # =========================================================
    try:
        log_dataset_state(df, dataset_path=str(source_path), dataset_source=source_type)
    except Exception as e:
        print(f"⚠️ Awareness update skipped: {e}")

    # =========================================================
    # 🔄 Retrain if drifted
    # =========================================================
    if drift_detected:
        print(f"🔄 Dataset drift detected — retraining model ({drift_reason})...")
        metrics = train_model()

        update_awareness_state(
            last_auto_retrain_status="success",
            last_auto_retrain_reason=drift_reason,
            last_auto_retrain_time=datetime.now().isoformat(timespec="seconds"),
            last_auto_retrain_metrics=metrics,
            dataset_hash=current_hash,
            dataset_source=source_type,
            dataset_path=str(source_path)
        )

        print(f"✅ Auto-retrain complete — new model accuracy: {metrics['accuracy']:.3f}")
        return {"status": "retrained", "reason": drift_reason, "metrics": metrics, "dataset_source": source_type}

    else:
        print(f"✅ No dataset change detected — retraining skipped.")
        update_awareness_state(
            last_auto_retrain_status="skipped",
            last_auto_retrain_reason=drift_reason,
            last_auto_retrain_time=datetime.now().isoformat(timespec="seconds"),
            dataset_hash=current_hash,
            dataset_source=source_type,
            dataset_path=str(source_path)
        )
        return {"status": "no_change", "reason": drift_reason, "dataset_source": source_type}

