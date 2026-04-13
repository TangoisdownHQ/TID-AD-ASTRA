import math
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException, Query

from app.system.planet_knowledge import (
    get_planet_catalog_records,
    get_planet_info,
    load_planet_data,
    search_planets,
)

router = APIRouter()

# =========================================================
# 🌌 Dataset paths
# =========================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PLANET_DATA_PATHS = [
    DATA_DIR / "nasa_exoplanets.csv",
    DATA_DIR / "open_exoplanet_catalogue.csv",
    DATA_DIR / "koi_fallback.csv",
    DATA_DIR / "astroml_exoplanets.csv",
]

NASA_DATA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+pl_name,pl_eqt,pl_rade,pl_orbsmax,pl_orbeccen,sy_dist,disc_year,discoverymethod+from+ps&format=csv"
)


# =========================================================
# 🧩 Helper — ensure dataset availability
# =========================================================
def ensure_dataset():
    """Ensure at least one dataset exists, otherwise download the NASA dataset."""
    found = False
    for path in PLANET_DATA_PATHS:
        if path.exists() and path.stat().st_size > 10000:
            found = True
            break

    if not found:
        print("🛰  No local planet datasets found — downloading NASA exoplanet archive...")
        try:
            resp = requests.get(NASA_DATA_URL, timeout=90)
            resp.raise_for_status()
            with open(PLANET_DATA_PATHS[0], "wb") as f:
                f.write(resp.content)
            print(f"✅ Downloaded NASA dataset to {PLANET_DATA_PATHS[0]}")
        except Exception as e:
            print(f"⚠️ Failed to download NASA dataset: {e}")


# =========================================================
# 🪐 /planets/info
# =========================================================
@router.get("/info")
async def planet_info(name: str | None = Query(None, description="Exact planet name (optional)")):
    """
    🌌  /planets/info
    - If `?name=PlanetName` is given → return detailed planet info.
    - If no name is given → return a list of available planets.
    """
    ensure_dataset()

    if name:
        info = get_planet_info(name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Planet '{name}' not found in database.")
        return info

    planets = []
    df = load_planet_data()
    if not df.empty:
        name_col = next(
            (c for c in ["pl_name", "name", "kepoi_name", "planet_name"] if c in df.columns),
            None,
        )
        if name_col:
            unique_names = (
                df[name_col]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            planets = [{"name": name} for name in unique_names[:250]]

    if not planets:
        raise HTTPException(status_code=500, detail="No planet data available.")
    return planets


# =========================================================
# 🔍 /planets/search
# =========================================================
@router.get("/search")
async def planet_search(query: str = Query(..., description="Partial search term for planet name")):
    """🔍 Search planets by partial name and return top matches."""
    results = search_planets(query)
    if not results:
        raise HTTPException(status_code=404, detail="No matching planets found")
    return results

# =========================================================
# 🌍 /planets/all — dynamic full dataset (hardened JSON-safe)
# =========================================================
@router.get("/all")
async def planet_all(limit: int = Query(100, description="Number of planets to return (default 100)")):
    """
    🧭 Return all planets with computed habitability and classification.
    Fully sanitizes floats (NaN, inf) for safe JSON serialization.
    """
    ensure_dataset()

    combined = get_planet_catalog_records()
    if not combined:
        raise HTTPException(status_code=500, detail="Failed to load planet data from local sources.")

    # Deep sanitize entire response
    def sanitize_for_json(obj):
        """Recursively remove NaN/inf values from dicts/lists for JSON safety."""
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(x) for x in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        else:
            return obj

    safe_combined = sanitize_for_json(combined)

    combined_sorted = sorted(
        safe_combined, key=lambda x: (x["habitability_score"] or 0), reverse=True
    )

    return combined_sorted[:limit]
