import time
import json
from pathlib import Path
from datetime import datetime
from app.models.classifier import train_model
from app.system.selfaware import update_awareness_state

WATCH_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
STATE_FILE = Path(__file__).resolve().parent / "watcher_state.json"


def get_known_files():
    """Load the record of previously seen files."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return set(json.load(f).get("known_files", []))
        except Exception:
            return set()
    return set()


def save_known_files(files):
    """Persist the set of known files."""
    with open(STATE_FILE, "w") as f:
        json.dump({"known_files": list(files)}, f, indent=4)


def watch_for_new_data(interval: int = 3600):
    """
    Continuously watch the uploads directory for new files.
    Retrain the model when new datasets are detected.
    """
    print(f"👁️ Watching {WATCH_DIR} for new data every {interval/60:.0f} minutes...")

    known_files = get_known_files()
    WATCH_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        current_files = {f.name for f in WATCH_DIR.glob("*") if f.is_file()}

        # Detect new files
        new_files = current_files - known_files
        if new_files:
            for new_file in new_files:
                print(f"🆕 New dataset detected: {new_file}")
                try:
                    # Retrain model
                    metrics = train_model()
                    update_awareness_state(
                        last_retrain=datetime.utcnow().isoformat(),
                        retrain_trigger="file_watcher",
                        metrics=metrics,
                        new_file=new_file
                    )
                    print(f"✅ Retrained successfully on {new_file}")
                except Exception as e:
                    print(f"❌ Failed to retrain on {new_file}: {e}")
            known_files |= new_files
            save_known_files(known_files)
        else:
            print(f"⏳ No new data detected at {datetime.utcnow().isoformat()}")

        time.sleep(interval)

