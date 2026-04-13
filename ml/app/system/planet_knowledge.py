"""
Planet Knowledge Layer
Provides metadata, habitability scoring, and descriptive context
for exoplanets known to the model.
"""

from pathlib import Path

import pandas as pd

# =========================================================
# 🌌 DATA PATHS (single source of truth: ml/app/data)
# =========================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DATA_PATHS = [
    DATA_DIR / "nasa_exoplanets.csv",
    DATA_DIR / "open_exoplanet_catalogue.csv",
    DATA_DIR / "koi_fallback.csv",
    DATA_DIR / "astroml_exoplanets.csv",
]

SOURCE_LABELS = {
    "nasa_exoplanets.csv": "NASA Exoplanet Archive",
    "open_exoplanet_catalogue.csv": "Open Exoplanet Catalogue",
    "koi_fallback.csv": "NASA Kepler KOI fallback",
    "astroml_exoplanets.csv": "AstroML Exoplanet Dataset",
}

COLUMN_ALIASES = {
    "planet_name": ["pl_name", "name", "kepoi_name", "planet_name"],
    "host_star": ["hostname", "host_star", "star_name"],
    "distance_pc": ["sy_dist", "system_distance", "st_dist"],
    "radius": ["pl_rade", "radius", "koi_prad"],
    "temperature": ["pl_eqt", "temperature", "koi_teq"],
    "period": ["pl_orbper", "period", "koi_period"],
    "semi_major_axis": ["pl_orbsmax", "semimajoraxis"],
    "eccentricity": ["pl_orbeccen", "eccentricity"],
    "discovery_year": ["disc_year", "discoveryyear", "koi_year"],
    "discovery_method": ["discoverymethod", "disc_method"],
    "star_temp": ["st_teff", "hoststar_temperature", "koi_steff"],
    "star_radius": ["st_rad", "hoststar_radius", "koi_srad"],
    "spectral_type": ["st_spectype", "hoststar_spectraltype"],
}

# =========================================================
# 🪐 LOAD PLANET DATA (multi-source)
# =========================================================
def load_planet_data():
    dfs = []

    for path in DATA_PATHS:
        if path.exists():
            try:
                df = pd.read_csv(path, comment="#", on_bad_lines="skip")
                df.columns = [c.strip().lower() for c in df.columns]
                df["_source_file"] = path.name
                df["_source"] = SOURCE_LABELS.get(path.name, path.name)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Failed to load planet metadata from {path}: {e}")

    if not dfs:
        print("⚠️ No valid planet dataset found.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True, sort=False)


# =========================================================
# 🔧 Helpers
# =========================================================
def _norm_name(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("-", "").replace("_", "")


def _find_name_column(df: pd.DataFrame):
    preferred = COLUMN_ALIASES["planet_name"]
    for c in preferred:
        if c in df.columns:
            return c
    return next((c for c in df.columns if "name" in c or "planet" in c), None)


def _value_from_row(row: pd.Series, *logical_names):
    for logical_name in logical_names:
        for column in COLUMN_ALIASES.get(logical_name, []):
            value = row.get(column)
            if pd.notna(value):
                return value
    return None


def _to_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _json_safe_scalar(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def parsecs_to_lightyears(pc):
    try:
        return round(float(pc) * 3.26156, 1)
    except Exception:
        return None


def infer_planet_type(radius):
    if not radius:
        return "unknown"
    try:
        r = float(radius)
        if r < 1.6:
            return "rocky (Earth-like size)"
        elif r < 4:
            return "super-Earth / mini-Neptune"
        else:
            return "gas giant"
    except Exception:
        return "unknown"


def infer_water_likelihood(temp):
    if not temp:
        return "unknown"
    if 273 <= temp <= 373:
        return "liquid water could exist (right temperature range)"
    elif temp < 273:
        return "likely frozen (too cold for liquid water)"
    else:
        return "liquid water unlikely (too hot)"


def get_planet_catalog_records(limit: int | None = None):
    df = load_planet_data()
    if df.empty:
        return []

    name_col = _find_name_column(df)
    if not name_col:
        return []

    records = []
    for _, row in df.iterrows():
        planet_name = row.get(name_col)
        if not planet_name or pd.isna(planet_name):
            continue

        radius = _to_float(_value_from_row(row, "radius"))
        temperature = _to_float(_value_from_row(row, "temperature"))
        distance_pc = _to_float(_value_from_row(row, "distance_pc"))
        period = _to_float(_value_from_row(row, "period"))
        semi_major_axis = _to_float(_value_from_row(row, "semi_major_axis"))
        eccentricity = _to_float(_value_from_row(row, "eccentricity"))
        star_temp = _to_float(_value_from_row(row, "star_temp"))
        score = compute_habitability_index(
            {
                "radius": radius,
                "temperature": temperature,
                "period": period,
                "star_temp": star_temp,
            }
        )
        status = (
            "habitable"
            if score >= 0.7
            else "marginal"
            if score >= 0.3
            else "inhospitable"
        )
        records.append(
            {
                "planet_name": str(planet_name),
                "planet_name_normalized": _norm_name(planet_name),
                "habitability_score": score,
                "status": status,
                "radius": radius,
                "temperature": temperature,
                "period": period,
                "semi_major_axis": semi_major_axis,
                "eccentricity": eccentricity,
                "distance_pc": distance_pc,
                "distance_ly": parsecs_to_lightyears(distance_pc),
                "host_star": _json_safe_scalar(_value_from_row(row, "host_star")),
                "discovery_year": _json_safe_scalar(_value_from_row(row, "discovery_year")),
                "discovery_method": _json_safe_scalar(_value_from_row(row, "discovery_method")),
                "spectral_type": _json_safe_scalar(_value_from_row(row, "spectral_type")),
                "planet_type": infer_planet_type(radius),
                "water_likelihood": infer_water_likelihood(temperature),
                "source": _json_safe_scalar(row.get("_source")),
            }
        )

    deduped = []
    seen = set()
    for item in sorted(records, key=lambda x: x["habitability_score"], reverse=True):
        key = item["planet_name_normalized"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if limit is not None and len(deduped) >= limit:
            break

    return deduped


# =========================================================
# 🔍 SEARCH PLANETS
# =========================================================
def search_planets(query: str, limit: int = 10):
    df = load_planet_data()
    if df.empty:
        return []

    name_col = _find_name_column(df)
    if not name_col:
        return []

    q = query.lower()
    matches = df[df[name_col].astype(str).str.lower().str.contains(q, na=False)].head(limit)

    results = []
    for _, row in matches.iterrows():
        distance_pc = _to_float(_value_from_row(row, "distance_pc"))
        results.append({
            "planet_name": row.get(name_col),
            "system_name": _value_from_row(row, "host_star"),
            "distance_from_earth_pc": distance_pc,
            "distance_from_earth_ly": parsecs_to_lightyears(distance_pc),
            "host_star": _value_from_row(row, "host_star"),
            "spectral_type": _value_from_row(row, "spectral_type"),
            "discovery_year": _value_from_row(row, "discovery_year"),
            "discovery_method": _value_from_row(row, "discovery_method"),
            "source": row.get("_source"),
        })

    return results


# =========================================================
# 🧠 GET PLANET INFO
# =========================================================
def get_planet_info(planet_name: str):
    df = load_planet_data()
    if df.empty or not planet_name:
        return {}

    name_col = _find_name_column(df)
    if not name_col:
        return {}

    norm_query = _norm_name(planet_name)
    mask = df[name_col].astype(str).apply(_norm_name) == norm_query
    match = df[mask]

    if match.empty:
        mask2 = df[name_col].astype(str).apply(_norm_name).str.contains(norm_query, na=False)
        match = df[mask2]

    if match.empty:
        return {}

    row = match.iloc[0]

    radius = _to_float(_value_from_row(row, "radius"))
    temp = _to_float(_value_from_row(row, "temperature"))
    distance_pc = _to_float(_value_from_row(row, "distance_pc"))
    period = _to_float(_value_from_row(row, "period"))
    star_temp = _to_float(_value_from_row(row, "star_temp"))
    star_radius = _to_float(_value_from_row(row, "star_radius"))
    eccentricity = _to_float(_value_from_row(row, "eccentricity"))
    semi_major_axis = _to_float(_value_from_row(row, "semi_major_axis"))

    return {
        "planet_name": row.get(name_col),
        "radius": radius,
        "temperature": temp,
        "period": period,
        "semi_major_axis": semi_major_axis,
        "eccentricity": eccentricity,
        "distance_pc": distance_pc,
        "distance_ly": parsecs_to_lightyears(distance_pc),
        "star_temp": star_temp,
        "star_radius": star_radius,
        "host_star": _value_from_row(row, "host_star"),
        "discovery_year": _json_safe_scalar(_value_from_row(row, "discovery_year")),
        "discovery_method": _json_safe_scalar(_value_from_row(row, "discovery_method")),
        "spectral_type": _json_safe_scalar(_value_from_row(row, "spectral_type")),
        "planet_type": infer_planet_type(radius),
        "water_likelihood": infer_water_likelihood(temp),
        "moons": "unknown (exomoons are extremely hard to detect with current technology)",
        "source": _json_safe_scalar(row.get("_source")),
    }


# =========================================================
# 🌍 HABITABILITY SCORING
# =========================================================
def compute_habitability_index(info: dict) -> float:
    if not info:
        return 0.0

    score = 0.0
    weights = {
        "radius": 0.3,
        "temperature": 0.4,
        "period": 0.2,
        "star_temp": 0.1,
    }

    try:
        if info.get("radius"):
            score += weights["radius"] * max(0, 1 - abs(info["radius"] - 1))

        if info.get("temperature"):
            score += weights["temperature"] * max(0, 1 - abs(info["temperature"] - 288) / 300)

        if info.get("period"):
            score += weights["period"] * max(0, 1 - abs(info["period"] - 365) / 500)

        if info.get("star_temp"):
            score += weights["star_temp"] * max(0, 1 - abs(info["star_temp"] - 5778) / 4000)

        return round(min(max(score, 0.0), 1.0), 3)
    except Exception:
        return 0.0


# =========================================================
# 📝 NARRATIVE SUMMARY (Human-Friendly)
# =========================================================
def narrative_summary(info: dict) -> str:
    if not info:
        return (
            "There is limited publicly available data for this planet right now. "
            "As astronomers gather more observations, we’ll be able to estimate its habitability more accurately."
        )

    parts = []

    host_star = info.get("host_star") or "its host star"
    parts.append(f"{info.get('planet_name', 'This planet')} orbits the star {host_star}.")

    if info.get("planet_type"):
        parts.append(f"It is classified as a {info['planet_type']} planet.")

    if info.get("radius"):
        parts.append(f"It has a radius about {info['radius']:.2f} times that of Earth.")

    if info.get("temperature"):
        parts.append(f"The estimated surface temperature is around {info['temperature']:.0f} K.")

    if info.get("water_likelihood"):
        parts.append(f"Water outlook: {info['water_likelihood']}.")

    if info.get("distance_ly"):
        parts.append(
            f"It is roughly {info['distance_ly']} light-years away. "
            "(Astronomers often use parsecs — 1 parsec ≈ 3.26 light-years.)"
        )

    if info.get("moons"):
        parts.append(f"Known moons: {info['moons']}.")

    parts.append(
        "Habitability estimates are theoretical and based on limited data. "
        "They indicate potential conditions, not proof of life."
    )

    return " ".join(parts)


# =========================================================
# 🔁 Backward Compatibility Wrapper
# =========================================================
def compute_habitability(temp=None, radius=None, semimajoraxis=None, ecc=None):
    info = {
        "temperature": temp,
        "radius": radius,
        "semi_major_axis": semimajoraxis,
        "eccentricity": ecc,
    }
    return compute_habitability_index(info)
