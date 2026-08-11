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
# Only the columns the app actually reads. `select *` pulled 683 columns / 56 MB
# per refresh; this returns the full table in about 1 MB.
NASA_COLUMNS = (
    "pl_name,hostname,pl_bmasse,pl_rade,pl_eqt,pl_orbper,pl_orbsmax,pl_orbeccen,"
    "sy_dist,disc_year,discoverymethod,disc_facility,st_teff,st_rad,st_mass,"
    "st_spectype,ra,dec"
)
NASA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    f"query=select+{NASA_COLUMNS}+from+pscomppars&format=csv"
)
# The Open Exoplanet Catalogue table is published with a .txt extension; the
# payload is CSV. The .csv path has always 404'd.
OEC_URL = "https://raw.githubusercontent.com/OpenExoplanetCatalogue/oec_tables/master/comma_separated/open_exoplanet_catalogue.txt"

# Kepler Objects of Interest — the live version of the bundled koi_fallback.csv,
# and the table the classifier's labels come from.
KOI_COLUMNS = (
    "kepoi_name,kepler_name,koi_disposition,koi_pdisposition,koi_score,"
    "koi_period,koi_prad,koi_teq,koi_insol,koi_duration,koi_depth,koi_impact,"
    "koi_model_snr,koi_steff,koi_srad,koi_smass,koi_slogg,koi_kepmag,ra,dec"
)
KOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    f"query=select+{KOI_COLUMNS}+from+cumulative&format=csv"
)

# TESS Objects of Interest — the active discovery pipeline.
TOI_COLUMNS = (
    "toi,toipfx,tid,tfopwg_disp,pl_orbper,pl_rade,pl_eqt,pl_insol,"
    "pl_trandurh,pl_trandep,st_dist,st_teff,st_rad,st_logg,st_tmag,"
    "ra,dec,toi_created"
)
TOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    f"query=select+{TOI_COLUMNS}+from+toi&format=csv"
)

DATA_REFRESH_INTERVAL_HOURS = int(os.getenv("DATA_REFRESH_INTERVAL_HOURS", "6"))


def _shape_toi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Give TOI rows the same shape as the other catalogs.

    The table identifies planets by a bare number (`toi` = 1234.01), so build the
    display name and host name the way the mission refers to them, and lift the
    discovery year out of the creation timestamp.
    """
    if "toi" in df.columns:
        df["pl_name"] = "TOI-" + df["toi"].astype(str).str.strip()
    if "toipfx" in df.columns:
        df["hostname"] = "TOI-" + df["toipfx"].astype(str).str.strip().str.replace(
            r"\.0$", "", regex=True
        )
    if "toi_created" in df.columns:
        df["disc_year"] = pd.to_datetime(df["toi_created"], errors="coerce").dt.year
    # Every TOI is a transit detection.
    df["discoverymethod"] = "Transit"
    return df


def fetch_csv(url: str, name: str, transform=None, timeout: int = 60):
    """
    Fetch CSV from a remote source and save it locally.
    Returns (rows, path)
    """
    print(f"🌐 Fetching {name} dataset...")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if transform is not None:
            df = transform(df)
        out = DATA_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"✅ Saved {name}: {len(df)} rows → {out.name}")
        return len(df), str(out)
    except Exception as e:
        print(f"❌ Failed to fetch {name}: {e}")
        return 0, None


def update_readme(rows_by_source: dict):
    """
    Update README External Data Sources table dynamically.
    """
    print("🪶 Updating README.md external sources table...")
    if not README_FILE.exists():
        print("⚠️ README.md not found — skipping.")
        return

    def row(label, endpoint, key):
        count = rows_by_source.get(key, 0)
        status = "✅ Updated" if count else "⚠️ Failed"
        return f"| **{label}** | `{endpoint}` | {count if count else '—'} | {status} |"

    new_table = "\n".join(
        [
            "| Source | Endpoint | Rows | Status |",
            "|--------|-----------|------|--------|",
            row("NASA Exoplanet Archive (confirmed)", NASA_URL, "nasa_exoplanets"),
            row("NASA Kepler KOI (cumulative)", KOI_URL, "koi_cumulative"),
            row("NASA TESS Objects of Interest", TOI_URL, "tess_toi"),
            row("Open Exoplanet Catalogue", OEC_URL, "open_exoplanet_catalogue"),
            "| **NASA Kepler KOI (offline fallback)** | `ml/app/data/koi_fallback.csv` | 2935 | 📦 Bundled |",
            "",
        ]
    )

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

    fetched = {
        "nasa_exoplanets": fetch_csv(NASA_URL, "nasa_exoplanets"),
        "koi_cumulative": fetch_csv(KOI_URL, "koi_cumulative", timeout=90),
        "tess_toi": fetch_csv(TOI_URL, "tess_toi", transform=_shape_toi, timeout=90),
        "open_exoplanet_catalogue": fetch_csv(OEC_URL, "open_exoplanet_catalogue"),
    }

    rows_by_source = {name: rows for name, (rows, _) in fetched.items()}

    # Update README.md
    update_readme(rows_by_source)

    # Log awareness
    update_awareness_state(
        last_refresh_sources={
            name: {"rows": rows, "path": path} for name, (rows, path) in fetched.items()
        },
        refresh_source="github_action",
    )

    # New files on disk mean the cached catalog is stale.
    try:
        from app.system.planet_knowledge import invalidate_catalog_cache

        invalidate_catalog_cache()
    except Exception as e:
        print(f"⚠️ Could not invalidate catalog cache: {e}")

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
