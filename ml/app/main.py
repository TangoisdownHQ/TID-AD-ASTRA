from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from threading import Thread
from datetime import datetime
import shutil
import json
import joblib

# =========================================================
# 🧩 INTERNAL IMPORTS
# =========================================================
from app.routes import models, system, planets   
from app.models.classifier import predict, train_model
from app.schemas import PredictRequest, TrainResponse
from app.system.auto_retrain import auto_retrain
from app.system.scheduler import start_scheduler_background
from app.system.fetch_agent import run_all_fetchers
from app.system.watcher import watch_for_new_data
from app.system.selfaware import AWARENESS_FILE
from app.system.update_datasets import main as refresh_datasets


# =========================================================
# 📂 DIRECTORIES
# =========================================================
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 🚀 FASTAPI APP SETUP
# =========================================================
app = FastAPI(title="TID-AD-ASTRA API", version="0.5.1")


# =========================================================
# 🧠 SYSTEM HEALTH
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "TID-AD-ASTRA Core API",
        "uptime": "nominal"
    } 

# =========================================================
# 🚀 MANUAL TRAINING
# =========================================================
@app.post("/train", response_model=TrainResponse)
def train():
    metrics = train_model()
    return metrics


# =========================================================
# 🔮 PREDICTION
# =========================================================
@app.post("/predict")
def make_prediction(req: PredictRequest):
    return predict(req.features)


# =========================================================
# 📦 ROUTE INCLUSION
# =========================================================
from app.routes import models as models_routes
from app.routes import system, planets

app.include_router(models.router, prefix="/models", tags=["models"])   # ✅ enables /models/explain
app.include_router(system.router, prefix="/system", tags=["system"])
app.include_router(planets.router, prefix="/planets", tags=["planets"])


# =========================================================
# 🤖 AUTO RETRAIN CHECK
# =========================================================
@app.post("/auto-train")
def auto_train():
    """Check dataset awareness and retrain if dataset changed."""
    result = auto_retrain()
    return result


# =========================================================
# 🧩 DATASET UPLOAD
# =========================================================
@app.post("/upload-dataset")
def upload_dataset(file: UploadFile = File(...)):
    """
    Upload new dataset to the monitored uploads folder.
    The watcher will automatically detect and retrain.
    """
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "uploaded", "path": str(file_path)}


# =========================================================
# 🧠 AWARENESS DASHBOARD
# =========================================================
@app.get("/awareness")
def awareness_dashboard():
    """Return current awareness state (model status, last retrain, dataset info)."""
    if not AWARENESS_FILE.exists():
        return {"status": "no-awareness-data", "details": "No awareness_state.json found"}

    try:
        with open(AWARENESS_FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {"error": "Corrupted awareness file"}

    return data


# =========================================================
# 🧩 MODEL SCHEMA INSPECTOR
# =========================================================
@app.get("/model/schema")
def model_schema():
    """Return expected model input schema and awareness metadata."""
    from app.models.classifier import REGISTRY_FILE

    if not REGISTRY_FILE.exists():
        return {"error": "❌ No registry.json found — train a model first."}

    try:
        with open(REGISTRY_FILE, "r") as f:
            registry = json.load(f)

        if not registry:
            return {"error": "❌ Registry is empty — train a model first."}

        latest_entry = registry[-1]
        model_path = Path(latest_entry["path"])
        model_hash = latest_entry.get("hash")
        dataset_path = latest_entry.get("dataset_path")

        if not model_path.exists():
            return {"error": f"❌ Model file not found: {model_path}"}

        # Awareness feature info
        feature_names = []
        if AWARENESS_FILE.exists():
            try:
                with open(AWARENESS_FILE, "r") as f:
                    awareness = json.load(f)
                    feature_names = awareness.get("feature_names", [])
            except json.JSONDecodeError:
                feature_names = []

        # Load model for inspection
        model = joblib.load(model_path)
        try:
            feature_count = getattr(model.get_booster(), "num_features", lambda: None)()
        except Exception:
            feature_count = len(feature_names) if feature_names else "unknown"

        return {
            "model_hash": model_hash,
            "trained_on": dataset_path,
            "feature_count": feature_count or len(feature_names),
            "feature_names": feature_names or "Unknown — train again to store schema",
            "artifact_path": str(model_path),
            "registry_path": str(REGISTRY_FILE)
        }

    except Exception as e:
        return {"error": f"Failed to load model schema: {str(e)}"}


# =========================================================
# ⚙️ STARTUP AUTOMATION
# =========================================================
@app.on_event("startup")
def start_autonomous_systems():
    """
    Start autonomous background processes:
    - Scheduler
    - Fetch agent
    - Dataset watcher
    - Local dataset refresh (non-blocking)
    """
    print("🧩 [Startup] Initializing autonomous systems...")

    # 1️⃣ Scheduler
    start_scheduler_background()

    # 2️⃣ Fetch Agent (delayed start)
    def init_fetch():
        time.sleep(5)
        try:
            run_all_fetchers()
        except Exception as e:
            print(f"⚠️ Fetch agent failed: {e}")

    Thread(target=init_fetch, daemon=True).start()

    # 3️⃣ Dataset Watcher
    Thread(target=watch_for_new_data, kwargs={"interval": 3600}, daemon=True).start()

    # 4️⃣ Local Dataset Refresh
    def init_local_refresh():
        print("🌍 Checking local dataset freshness...")
        try:
            refresh_datasets()

            if AWARENESS_FILE.exists():
                with open(AWARENESS_FILE, "r") as f:
                    state = json.load(f)
                    last_refresh = state.get("last_dataset_refresh", "unknown")
                    sources = state.get("last_refresh_sources", {})
                    print("\n🪶 [Local Dataset Summary]")
                    print(f"   Last Refresh: {last_refresh}")
                    for name, info in sources.items():
                        rows = info.get("rows", "—")
                        path = info.get("path", "—")
                        print(f"   - {name}: {rows} rows ({path})")
                    print("🧠 Awareness state updated.\n")
        except Exception as e:
            print(f"⚠️ Local dataset refresh skipped: {e}")

    Thread(target=init_local_refresh, daemon=True).start()

    print("✅ Autonomous agents initialized.")

