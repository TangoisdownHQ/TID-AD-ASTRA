import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from app.system.selfaware import update_awareness_state

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
OPEN_EXOPLANET_URL = "https://raw.githubusercontent.com/OpenExoplanetCatalogue/oec_tables/master/comma_separated/open_exoplanet_catalogue.txt"
ASTROML_URL = "https://raw.githubusercontent.com/astroML/astroML-data/main/datasets/exoplanets.csv"


def fetch_nasa_exoplanets(limit=5000) -> Path:
    """Fetch NASA Exoplanet Archive (TAP) data."""
    query = f"select top {limit} * from exoplanets"
    params = {"query": query, "format": "csv"}
    path = DATA_DIR / "nasa_exoplanets.csv"

    print("🌐 Fetching exoplanets from NASA Exoplanet Archive...")
    try:
        r = requests.get(NASA_TAP_URL, params=params, timeout=20)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"✅ NASA dataset saved to {path}")
        update_awareness_state(last_fetch_source="NASA TAP", last_fetch_time=datetime.now().isoformat())
        return path
    except Exception as e:
        print(f"❌ Failed to fetch NASA TAP exoplanets: {e}")
        return None


def fetch_open_exoplanet_catalogue() -> Path:
    """Fetch Open Exoplanet Catalogue (community source)."""
    path = DATA_DIR / "open_exoplanet_catalogue.csv"
    print("🌐 Fetching data from Open Exoplanet Catalogue...")

    try:
        r = requests.get(OPEN_EXOPLANET_URL, timeout=20)
        r.raise_for_status()
        # The repo now provides CSV-like text; save directly
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"✅ Open Exoplanet Catalogue saved to {path}")
        update_awareness_state(last_fetch_source="Open Exoplanet Catalogue", last_fetch_time=datetime.now().isoformat())
        return path
    except Exception as e:
        print(f"❌ Failed to fetch Open Exoplanet Catalogue: {e}")
        return None


def fetch_astroml_exoplanets() -> Path:
    """Fetch AstroML dataset (CSV format)."""
    path = DATA_DIR / "astroml_exoplanets.csv"
    print("🌐 Fetching AstroML Exoplanet dataset...")

    try:
        r = requests.get(ASTROML_URL, timeout=20)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"✅ AstroML dataset saved to {path}")
        update_awareness_state(last_fetch_source="AstroML", last_fetch_time=datetime.now().isoformat())
        return path
    except Exception as e:
        print(f"❌ Failed to fetch AstroML dataset: {e}")
        return None


def run_all_fetchers():
    """Run all fetchers sequentially."""
    print("🧠 Running dataset fetch cycle...")

    nasa_path = fetch_nasa_exoplanets()
    oec_path = fetch_open_exoplanet_catalogue()
    astro_path = fetch_astroml_exoplanets()

    fetched = [p for p in [nasa_path, oec_path, astro_path] if p is not None]
    if fetched:
        update_awareness_state(
            last_successful_fetch=datetime.now().isoformat(),
            fetched_files=[str(f) for f in fetched],
        )
    else:
        print("⚠️ All fetch attempts failed — fallback dataset will remain active.")

    print("🚀 Fetch cycle complete.")

