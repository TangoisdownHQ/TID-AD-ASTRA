from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import pandas as pd
import requests

from app.system.planet_knowledge import (
    get_planet_info,
    search_planets,
    compute_habitability,
)

router = APIRouter()

# =========================================================
# 🌌 Dataset paths
# =========================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PLANET_DATA_PATHS = [
    DATA_DIR / "nasa_exoplanets.csv",
    DATA_DIR / "open_exoplanet_catalogue.csv",
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
    for path in PLANET_DATA_PATHS:
        if path.exists():
            try:
                df = pd.read_csv(path, on_bad_lines="skip", engine="python")
                name_col = next(
                    (c for c in df.columns if "name" in c.lower() or "planet" in c.lower()),
                    None,
                )
                if name_col:
                    for pname in df[name_col].dropna().unique().tolist()[:100]:
                        planets.append({"name": str(pname)})
                    break
            except Exception as e:
                print(f"⚠️ Failed to load planet data from {path}: {e}")

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


import math

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

    essential_cols = [
        "pl_name", "pl_eqt", "pl_rade", "pl_orbsmax",
        "pl_orbeccen", "sy_dist", "disc_year", "discoverymethod",
    ]

    df = None
    last_error = None

    for path in PLANET_DATA_PATHS:
        if not path.exists():
            continue
        try:
            print(f"🧩 Loading dataset: {path}")
            df = pd.read_csv(
                path,
                usecols=lambda c: any(ec in c for ec in essential_cols),
                engine="python",
                on_bad_lines="skip"
            )
            print(f"✅ Loaded {len(df)} entries from {path.name}")
            break
        except Exception as e:
            last_error = e
            print(f"⚠️ Failed to read {path}: {e}")
            continue

    if df is None or df.empty:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load planet data. Last error: {last_error}",
        )

    df.columns = [c.lower().strip() for c in df.columns]

    # 🔧 Replace NaN/inf with None globally
    df = df.replace([float("inf"), float("-inf")], None)
    df = df.where(pd.notna(df), None)

    combined = []
    for _, row in df.head(limit).iterrows():
        name = row.get("pl_name") or "Unknown"
        score = compute_habitability(
            temp=row.get("pl_eqt"),
            radius=row.get("pl_rade"),
            semimajoraxis=row.get("pl_orbsmax"),
            ecc=row.get("pl_orbeccen"),
        )

        status = (
            "habitable"
            if score >= 0.7
            else "marginal"
            if score >= 0.3
            else "inhospitable"
        )

        combined.append(
            {
                "planet_name": str(name),
                "habitability_score": score,
                "status": status,
                "temperature": row.get("pl_eqt"),
                "radius": row.get("pl_rade"),
                "distance_ly": row.get("sy_dist"),
                "discovery_year": row.get("disc_year"),
                "discovery_method": row.get("discoverymethod"),
            }
        )

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

