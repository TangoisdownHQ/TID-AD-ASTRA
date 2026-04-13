import requests
import json
import pandas as pd
import os
from pathlib import Path
from io import StringIO
from datetime import datetime

# =========================================================
# 🗂️ Paths
# =========================================================
REPO_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
README_FILE = REPO_DIR / "README.md"
AWARENESS_FILE = Path(__file__).resolve().parent / "awareness_state.json"

# =========================================================
# 🌐 Sources
# =========================================================
NASA_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+5000+*+from+pscomppars&format=csv"
OEC_URL = "https://raw.githubusercontent.com/OpenExoplanetCatalogue/oec_tables/master/comma_separated/open_exoplanet_catalogue.csv"
ASTROML_URL = "https://raw.githubusercontent.com/astroML/astroML-data/main/datasets/exoplanets.csv"
DATA_REFRESH_INTERVAL_HOURS = int(os.getenv("DATA_REFRESH_INTERVAL_HOURS", "6"))


def fetch_csv(url: str, name: str):
    """
    Fetch CSV from a remote source and save it locally.
    Returns (rows, path)
    """
    print(f"🌐 Fetching {name} dataset...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        out = DATA_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"✅ Saved {name}: {len(df)} rows → {out.name}")
        return len(df), str(out)
    except Exception as e:
        print(f"❌ Failed to fetch {name}: {e}")
        return 0, None


def update_readme(nasa_rows, oec_rows, astroml_rows):
    """
    Update README External Data Sources table dynamically.
    """
    print("🪶 Updating README.md external sources table...")
    if not README_FILE.exists():
        print("⚠️ README.md not found — skipping.")
        return

    new_table = f"""| Source | Endpoint | Rows | Status |
|--------|-----------|------|--------|
| **NASA Exoplanet Archive** | `{NASA_URL}` | {nasa_rows if nasa_rows else "—"} | {'✅ Updated' if nasa_rows else '⚠️ Failed'} |
| **Open Exoplanet Catalogue** | `{OEC_URL}` | {oec_rows if oec_rows else "—"} | {'✅ Updated' if oec_rows else '⚠️ Failed'} |
| **AstroML Exoplanet Dataset** | `{ASTROML_URL}` | {astroml_rows if astroml_rows else "—"} | {'✅ Updated' if astroml_rows else '⚠️ Failed'} |
"""

    text = README_FILE.read_text()
    start = text.find("## 🌐 External Data Sources")
    if start == -1:
        print("⚠️ Could not find External Data Sources section — skipping README update.")
        return

    # Find the next "##" section or end of file to replace content in place
    end = text.find("## ", start + 10)
    if end == -1:
        end = len(text)

    updated = text[:start] + "## 🌐 External Data Sources\n\n" + new_table + "\n" + text[end:]
    README_FILE.write_text(updated)
    print("✅ README.md data sources table updated.")


def update_awareness_state(**kwargs):
    """
    Update awareness_state.json with dataset refresh info.
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

    state.update(kwargs)
    state["last_dataset_refresh"] = datetime.now().isoformat(timespec="seconds")

    with open(AWARENESS_FILE, "w") as f:
        json.dump(state, f, indent=4)

    print(f"🧠 Awareness updated → {AWARENESS_FILE}")


def get_refresh_state():
    try:
        if AWARENESS_FILE.exists():
            with open(AWARENESS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def dataset_refresh_needed(max_age_hours: int = DATA_REFRESH_INTERVAL_HOURS) -> bool:
    state = get_refresh_state()
    last_refresh = state.get("last_dataset_refresh")
    if not last_refresh:
        return True

    try:
        refreshed_at = datetime.fromisoformat(last_refresh)
    except Exception:
        return True

    age_seconds = (datetime.now() - refreshed_at).total_seconds()
    return age_seconds >= max_age_hours * 3600


def main():
    print("🚀 Starting dataset refresh cycle...")
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    nasa_rows, nasa_path = fetch_csv(NASA_URL, "nasa_exoplanets")
    oec_rows, oec_path = fetch_csv(OEC_URL, "open_exoplanet_catalogue")
    astroml_rows, astroml_path = fetch_csv(ASTROML_URL, "astroml_exoplanets")

    # Update README.md
    update_readme(nasa_rows, oec_rows, astroml_rows)

    # Log awareness
    update_awareness_state(
        last_refresh_sources={
            "nasa_exoplanets": {"rows": nasa_rows, "path": nasa_path},
            "open_exoplanet_catalogue": {"rows": oec_rows, "path": oec_path},
            "astroml_exoplanets": {"rows": astroml_rows, "path": astroml_path},
        },
        refresh_source="github_action",
    )

    print("✅ Dataset refresh complete — all changes logged and committed if on CI.")


def refresh_if_stale(max_age_hours: int = DATA_REFRESH_INTERVAL_HOURS):
    if dataset_refresh_needed(max_age_hours=max_age_hours):
        print(f"🌍 Dataset refresh is stale or missing. Refreshing now (threshold: {max_age_hours}h)...")
        main()
        return True

    print(f"✅ Dataset refresh is current within the last {max_age_hours} hours.")
    return False


if __name__ == "__main__":
    main()
