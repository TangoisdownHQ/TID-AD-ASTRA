"""
Planet Knowledge Layer
Provides metadata, habitability scoring, and descriptive context
for exoplanets known to the model.
"""

from math import exp
import pandas as pd
from pathlib import Path

# =========================================================
# 🌌 DATA PATHS (multi-source aware)
# =========================================================
DATA_PATHS = [
    Path(__file__).resolve().parents[2] / "data" / "open_exoplanet_catalogue.csv",
    Path(__file__).resolve().parents[2] / "data" / "nasa_2025-10-06.csv",
    Path(__file__).resolve().parents[2] / "data" / "datasets" / "koi_fallback.csv",
]


# =========================================================
# 🪐 LOAD PLANET DATA
# =========================================================
def load_planet_data():
    """
    Load the first available planet dataset.
    Returns a pandas DataFrame with normalized lowercase columns.
    """
    for path in DATA_PATHS:
        if path.exists():
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip().lower() for c in df.columns]
                return df
            except Exception as e:
                print(f"⚠️ Failed to load planet metadata from {path}: {e}")
    print("⚠️ No valid planet dataset found.")
    return pd.DataFrame()


# =========================================================
# 🔍 SEARCH PLANETS
# =========================================================
def search_planets(query: str, limit: int = 10):
    """
    Search planets by partial name (case-insensitive).
    Returns a list of basic metadata for matching planets.
    """
    df = load_planet_data()
    if df.empty:
        return []

    name_col = None
    for c in df.columns:
        if "name" in c or "planet" in c:
            name_col = c
            break
    if not name_col:
        return []

    query_lower = query.lower()
    matches = df[df[name_col].astype(str).str.lower().str.contains(query_lower, na=False)].head(limit)

    results = []
    for _, row in matches.iterrows():
        results.append({
            "planet_name": row.get(name_col),
            "system_name": row.get("system", row.get("hostname", "Unknown System")),
            "distance_from_earth_ly": row.get("distance", row.get("sy_dist", None)),
            "host_star": row.get("star_name", row.get("hostname", "Unknown Star")),
            "spectral_type": row.get("star_type", row.get("st_spectype", None)),
            "discovery_year": row.get("discovered", row.get("disc_year", None)),
        })

    return results


# =========================================================
# 🧠 GET PLANET INFO (DETAILED)
# =========================================================
def get_planet_info(planet_name: str):
    """
    Retrieve detailed information for a specific planet by name.
    Enriches with physical and stellar data if available.
    """
    df = load_planet_data()
    if df.empty:
        return None

    # Identify name column
    name_col = None
    for c in df.columns:
        if "name" in c or "planet" in c:
            name_col = c
            break
    if not name_col:
        return None

    match = df[df[name_col].astype(str).str.lower() == planet_name.lower()]
    if match.empty:
        return None

    row = match.iloc[0].to_dict()

    def col_val(*names):
        for n in names:
            if n in df.columns:
                return row.get(n)
        return None

    return {
        "planet_name": row.get(name_col),
        "system_name": col_val("system", "hostname"),
        "mass_earth": col_val("mass", "pl_bmasse", "pl_massj", "msini"),
        "radius_earth": col_val("radius", "pl_rade", "pl_radj"),
        "equilibrium_temperature_k": col_val("temp", "pl_eqt", "teq", "planet_temp"),
        "orbital_period_days": col_val("period", "pl_orbper"),
        "semi_major_axis_au": col_val("semimajoraxis", "pl_orbsmax"),
        "eccentricity": col_val("ecc", "pl_orbeccen"),
        "distance_from_earth_ly": col_val("distance", "sy_dist"),
        "discovery_year": col_val("discovered", "disc_year"),
        "host_star": {
            "name": col_val("star_name", "hostname"),
            "temperature": col_val("star_teff", "st_teff"),
            "spectral_type": col_val("star_type", "st_spectype"),
            "mass_solar": col_val("star_mass", "st_mass"),
            "radius_solar": col_val("star_radius", "st_rad"),
        },
        "moons_detected": col_val("moons"),
    }


# =========================================================
# 🌍 HABITABILITY SCORING
# =========================================================
def compute_habitability(temp, radius, semimajoraxis, ecc):
    """
    Simple Earth Similarity Index — returns a habitability score between 0 and 1.
    """
    try:
        temp_term = exp(-abs(temp - 288) / 50)
        radius_term = exp(-abs(radius - 1))
        orbit_term = exp(-abs(semimajoraxis - 1))
        ecc_term = exp(-ecc * 2)
        score = round(temp_term * radius_term * orbit_term * ecc_term, 3)
        return max(min(score, 1.0), 0.0)
    except Exception:
        return 0.0

