# app/system/selfaware.py
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Awareness state file
AWARENESS_FILE = Path(__file__).resolve().parent / "awareness_state.json"


# =========================================================
# 🧠 UTILITIES
# =========================================================
def _json_safe(obj):
    """Convert NumPy or unsupported types to JSON-safe equivalents."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [_json_safe(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    else:
        return obj


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute a unique hash for the given dataset.
    This helps detect when the data has changed.
    """
    try:
        hash_val = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
        return hash_val
    except Exception as e:
        print(f"⚠️ Failed to compute dataset hash: {e}")
        return "unknown"


# =========================================================
# 🧩 AWARENESS CORE
# =========================================================
def update_awareness_state(**kwargs):
    """
    Update or append fields to awareness_state.json dynamically.
    Automatically timestamps updates and ensures JSON safety.
    """
    try:
        if AWARENESS_FILE.exists():
            with open(AWARENESS_FILE, "r") as f:
                state = json.load(f)
                if not isinstance(state, dict):
                    state = {}
        else:
            state = {}
    except json.JSONDecodeError:
        state = {}

    # Sanitize new data
    safe_kwargs = _json_safe(kwargs)
    state.update(safe_kwargs)
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")

    # Enrich awareness state dynamically
    if "label_mapping" in state:
        state["trained_label_classes"] = list(state["label_mapping"].keys())

    # Compute lightweight integrity hashes if model/dataset info present
    if "last_model_path" in state and Path(state["last_model_path"]).exists():
        try:
            model_hash = hashlib.md5(Path(state["last_model_path"]).read_bytes()).hexdigest()
            state["last_model_file_hash"] = model_hash
        except Exception:
            pass

    if "last_trained_dataset" in state and Path(state["last_trained_dataset"]).exists():
        try:
            df = pd.read_csv(state["last_trained_dataset"])
            state["last_dataset_hash"] = compute_dataset_hash(df)
        except Exception:
            pass

    # Write back safely
    with open(AWARENESS_FILE, "w") as f:
        json.dump(state, f, indent=4)

    print(f"🧠 Awareness log updated → {AWARENESS_FILE}")
    return state


# =========================================================
# 🪐 COMPATIBILITY WRAPPER
# =========================================================
def log_dataset_state(df, dataset_path: str, dataset_source: str):
    """
    Compatibility wrapper for dataset logging.
    Converts positional arguments to keyword-based update call.
    """
    dataset_shape = list(df.shape) if hasattr(df, "shape") else None
    return update_awareness_state(
        dataset_path=dataset_path,
        dataset_source=dataset_source,
        dataset_shape=dataset_shape
    )

