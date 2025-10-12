import requests
import json
import pandas as pd
from pathlib import Path
from io import StringIO
from datetime import datetime

# =========================================================
# 🗂️ Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
README_FILE = BASE_DIR / "README.md"
AWARENESS_FILE = BASE_DIR / "app" / "system" / "awareness_state.json"

# =========================================================
# 🌐 Sources
# =========================================================
NASA_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+5000+*+from+pscomppars&format=csv"
OEC_URL = "https://raw.githubusercontent.com/OpenExoplanetCatalogue/oec_tables/master/comma_separated/open_exoplanet_catalogue.csv"


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


def update_readme(nasa_rows, oec_rows):
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
| **AstroML Exoplanet Dataset** | `https://github.com/astroML/astroML-data` | — | ⚠️ May 404 |
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


def main():
    print("🚀 Starting dataset refresh cycle...")
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    nasa_rows, nasa_path = fetch_csv(NASA_URL, "nasa_exoplanets")
    oec_rows, oec_path = fetch_csv(OEC_URL, "open_exoplanet_catalogue")

    # Update README.md
    update_readme(nasa_rows, oec_rows)

    # Log awareness
    update_awareness_state(
        last_refresh_sources={
            "nasa_exoplanets": {"rows": nasa_rows, "path": nasa_path},
            "open_exoplanet_catalogue": {"rows": oec_rows, "path": oec_path},
        },
        refresh_source="github_action",
    )

    print("✅ Dataset refresh complete — all changes logged and committed if on CI.")


if __name__ == "__main__":
    main()

