"""
Planet Knowledge Layer
Provides metadata, habitability scoring, and descriptive context
for exoplanets known to the model.
"""

from pathlib import Path
from threading import Lock

import pandas as pd

# =========================================================
# 🌌 DATA PATHS (single source of truth: ml/app/data)
# =========================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DATA_PATHS = [
    DATA_DIR / "nasa_exoplanets.csv",
    DATA_DIR / "open_exoplanet_catalogue.csv",
    DATA_DIR / "koi_cumulative.csv",
    DATA_DIR / "tess_toi.csv",
    DATA_DIR / "koi_fallback.csv",
]

SOURCE_LABELS = {
    "nasa_exoplanets.csv": "NASA Exoplanet Archive",
    "open_exoplanet_catalogue.csv": "Open Exoplanet Catalogue",
    "koi_cumulative.csv": "NASA Kepler KOI (cumulative)",
    "tess_toi.csv": "NASA TESS Objects of Interest",
    "koi_fallback.csv": "NASA Kepler KOI fallback",
}

SOURCE_FAMILIES = {
    "nasa_exoplanets.csv": "exoplanet_catalog",
    "open_exoplanet_catalogue.csv": "exoplanet_catalog",
    "koi_cumulative.csv": "mission_candidates",
    "tess_toi.csv": "mission_candidates",
    "koi_fallback.csv": "exoplanet_catalog",
}

# Lower number wins when the same planet appears in more than one source.
SOURCE_PRIORITY = {
    "nasa_exoplanets.csv": 0,
    "open_exoplanet_catalogue.csv": 1,
    "koi_cumulative.csv": 2,
    "tess_toi.csv": 3,
    "koi_fallback.csv": 4,
}

# Canonical per-row planet name, resolved per source file at load time so that
# sources using different name columns stay reachable after the frames are merged.
NAME_COLUMN = "_planet_name"

# Not every catalog row is a planet. KOI tables carry vetted false positives,
# and the OEC list column carries retractions and Solar System bodies.
DISPOSITION_ALIASES = [
    "koi_disposition",
    "koi_pdisposition",
    "tfopwg_disp",
    "disposition",
    "list",
]

# Rows with these dispositions are excluded from the exoplanet catalog entirely.
EXCLUDED_DISPOSITIONS = {"false positive", "retracted", "solar system"}

# Ranked best-first when the same planet appears with different dispositions.
DISPOSITION_RANK = {"confirmed": 0, "candidate": 1, "controversial": 2}

# Sources that only ever publish confirmed planets and carry no status column.
SOURCE_DEFAULT_DISPOSITION = {"nasa_exoplanets.csv": "confirmed"}

# TFOPWG uses two-letter codes rather than words.
TFOPWG_DISPOSITIONS = {
    "cp": "confirmed",       # confirmed planet
    "kp": "confirmed",       # known planet
    "pc": "candidate",       # planet candidate
    "apc": "candidate",      # ambiguous planet candidate
    "fp": "false positive",
    "fa": "false positive",  # false alarm
}

DISPOSITION_COLUMN = "_disposition"

COLUMN_ALIASES = {
    "planet_name": ["pl_name", "name", "kepler_name", "kepoi_name", "planet_name"],
    "host_star": ["hostname", "host_star", "star_name"],
    "distance_pc": ["sy_dist", "system_distance", "st_dist"],
    "mass_earth": ["pl_bmasse", "mass", "koi_mass"],
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
    # Transit-geometry fields — only mission candidate tables carry these, and
    # they are what let the classifier see an actual planet instead of a median.
    "transit_duration": ["koi_duration", "pl_trandurh"],
    "transit_depth": ["koi_depth", "pl_trandep"],
    "transit_impact": ["koi_impact"],
    "transit_snr": ["koi_model_snr"],
    "star_logg": ["koi_slogg", "st_logg"],
    "star_magnitude": ["koi_kepmag"],
    "insolation": ["koi_insol", "pl_insol"],
    # NASA's own vetting confidence for a KOI (0-1). Unlike our classifier this
    # varies meaningfully per object, so it is worth showing when present.
    "disposition_score": ["koi_score"],
}

# =========================================================
# 🗃️ CACHE (parsed frames are reused until a source file changes on disk)
# =========================================================
_CACHE_LOCK = Lock()
_CACHE = {"signature": None, "frame": None, "records": None, "name_index": None}


def _source_signature():
    """Fingerprint the on-disk sources so a dataset refresh invalidates the cache."""
    signature = []
    for path in DATA_PATHS:
        try:
            stat = path.stat()
            signature.append((path.name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            continue
    return tuple(signature)


def invalidate_catalog_cache():
    """Drop cached frames and records. Call after refreshing local datasets."""
    with _CACHE_LOCK:
        _CACHE["signature"] = None
        _CACHE["frame"] = None
        _CACHE["records"] = None
        _CACHE["name_index"] = None


# =========================================================
# 🏷️ CANONICAL NAME RESOLUTION (per source file)
# =========================================================
def _clean_name_value(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _resolve_name_series(df: pd.DataFrame) -> pd.Series:
    """
    Build one canonical name column for a single source frame.

    Aliases are applied in preference order and combined per row, so a KOI row
    with a `kepler_name` uses it while a candidate-only row falls back to
    `kepoi_name`. Sources that use `name` or `pl_name` resolve just as well.
    """
    resolved = pd.Series([None] * len(df), index=df.index, dtype="object")

    candidate_columns = [c for c in COLUMN_ALIASES["planet_name"] if c in df.columns]
    if not candidate_columns:
        fallback = next((c for c in df.columns if "name" in c or "planet" in c), None)
        candidate_columns = [fallback] if fallback else []

    for column in candidate_columns:
        cleaned = df[column].map(_clean_name_value)
        resolved = resolved.where(resolved.notna(), cleaned)

    return resolved


def _normalize_disposition(value):
    """Fold the various catalogs' status vocabularies into one small set."""
    text = _clean_name_value(value)
    if not text:
        return None
    text = text.lower()

    if text in TFOPWG_DISPOSITIONS:
        return TFOPWG_DISPOSITIONS[text]

    # Order matters: "Retracted planet candidate" is a retraction, not a candidate.
    if "false positive" in text:
        return "false positive"
    if "retracted" in text:
        return "retracted"
    if "solar system" in text:
        return "solar system"
    if "controversial" in text:
        return "controversial"
    if "candidate" in text or "objects of interest" in text:
        return "candidate"
    if "confirmed" in text:
        return "confirmed"
    return None


def _resolve_disposition_series(df: pd.DataFrame, source_file: str) -> pd.Series:
    resolved = pd.Series([None] * len(df), index=df.index, dtype="object")

    for column in DISPOSITION_ALIASES:
        if column in df.columns:
            cleaned = df[column].map(_normalize_disposition)
            resolved = resolved.where(resolved.notna(), cleaned)

    default = SOURCE_DEFAULT_DISPOSITION.get(source_file)
    if default:
        resolved = resolved.fillna(default)

    return resolved


# =========================================================
# 🪐 LOAD PLANET DATA (multi-source)
# =========================================================
def load_planet_data():
    """
    Return the merged multi-source catalog frame.

    The result is cached and shared — treat it as read-only.
    """
    signature = _source_signature()
    with _CACHE_LOCK:
        if _CACHE["signature"] == signature and _CACHE["frame"] is not None:
            return _CACHE["frame"]

    dfs = []

    for path in DATA_PATHS:
        if path.exists():
            try:
                df = pd.read_csv(path, comment="#", on_bad_lines="skip")
                df.columns = [c.strip().lower() for c in df.columns]
                df[NAME_COLUMN] = _resolve_name_series(df)
                df = df[df[NAME_COLUMN].notna()].copy()
                if df.empty:
                    print(f"⚠️ No usable planet names found in {path.name} — skipping.")
                    continue

                df[DISPOSITION_COLUMN] = _resolve_disposition_series(df, path.name)
                rejected = df[DISPOSITION_COLUMN].isin(EXCLUDED_DISPOSITIONS)
                if rejected.any():
                    print(f"🚫 {path.name}: dropped {int(rejected.sum())} non-planet rows "
                          f"({', '.join(sorted(df.loc[rejected, DISPOSITION_COLUMN].unique()))})")
                    df = df[~rejected].copy()
                df["_source_file"] = path.name
                df["_source"] = SOURCE_LABELS.get(path.name, path.name)
                df["_source_family"] = SOURCE_FAMILIES.get(path.name, "exoplanet_catalog")
                df["_source_priority"] = SOURCE_PRIORITY.get(path.name, 99)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Failed to load planet metadata from {path}: {e}")

    if not dfs:
        print("⚠️ No valid planet dataset found.")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    merged["_planet_name_normalized"] = merged[NAME_COLUMN].map(_norm_name)
    name_index = _build_name_index(merged)

    with _CACHE_LOCK:
        _CACHE["signature"] = signature
        _CACHE["frame"] = merged
        _CACHE["records"] = None
        _CACHE["name_index"] = name_index

    return merged


def _build_name_index(df: pd.DataFrame) -> dict[str, list]:
    """
    Map every normalised name alias to the rows that carry it.

    A planet is reachable by any designation a source lists — `Kepler-227 b` and
    `K00752.01` both resolve, even though only one is the display name.
    """
    index: dict[str, set] = {}
    alias_columns = [c for c in COLUMN_ALIASES["planet_name"] if c in df.columns]
    alias_columns.append(NAME_COLUMN)

    for column in alias_columns:
        cleaned = df[column].map(_clean_name_value)
        for label, normalized in cleaned.dropna().map(_norm_name).items():
            if normalized:
                index.setdefault(normalized, set()).add(label)

    return {key: sorted(rows) for key, rows in index.items()}


def get_name_index() -> dict[str, list]:
    load_planet_data()
    with _CACHE_LOCK:
        return _CACHE["name_index"] or {}


# =========================================================
# 🔧 Helpers
# =========================================================
def _norm_name(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("-", "").replace("_", "")


def _find_name_column(df: pd.DataFrame):
    """Kept for backward compatibility — the canonical column is always present now."""
    if NAME_COLUMN in df.columns:
        return NAME_COLUMN
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


# Fields counted when deciding which source's row best represents a planet.
COMPLETENESS_FIELDS = (
    "radius",
    "temperature",
    "period",
    "semi_major_axis",
    "eccentricity",
    "distance_pc",
    "host_star",
    "discovery_year",
    "discovery_method",
    "spectral_type",
)


def _record_completeness(record: dict) -> int:
    return sum(1 for field in COMPLETENESS_FIELDS if record.get(field) is not None)


def _merge_rank(record: dict):
    """Prefer a confirmed planet, then the most complete record, then source priority."""
    disposition_rank = DISPOSITION_RANK.get(record.get("disposition"), 3)
    return (
        -disposition_rank,
        _record_completeness(record),
        -record.get("_source_priority", 99),
    )


def get_planet_catalog_records(limit: int | None = None):
    signature = _source_signature()
    with _CACHE_LOCK:
        if _CACHE["signature"] == signature and _CACHE["records"] is not None:
            cached = _CACHE["records"]
            return cached[:limit] if limit is not None else cached

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
        profile = compute_habitability_profile(
            {
                "radius": radius,
                "temperature": temperature,
                "period": period,
                "star_temp": star_temp,
            }
        )
        records.append(
            {
                "planet_name": str(planet_name),
                "planet_name_normalized": _norm_name(planet_name),
                "habitability_score": profile["score"],
                "habitability_coverage": profile["coverage"],
                "status": profile["status"],
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
                "disposition": _json_safe_scalar(row.get(DISPOSITION_COLUMN)),
                "source": _json_safe_scalar(row.get("_source")),
                "source_family": _json_safe_scalar(row.get("_source_family")),
                "_source_priority": int(row.get("_source_priority", 99)),
            }
        )

    # De-duplicate across sources: keep the most complete row for each planet and
    # record which other catalogs also list it.
    best: dict[str, dict] = {}
    contributors: dict[str, list[str]] = {}
    for item in records:
        key = item["planet_name_normalized"]
        source = item.get("source")
        if source and source not in contributors.setdefault(key, []):
            contributors[key].append(source)
        current = best.get(key)
        if current is None or _merge_rank(item) > _merge_rank(current):
            best[key] = item

    deduped = []
    for key, item in best.items():
        item.pop("_source_priority", None)
        item["sources"] = contributors.get(key, [])
        deduped.append(item)

    # Rank classified planets above ones we cannot classify. A planet with only
    # its orbital period measured can score 0.99 on that single factor, and it
    # must not outrank a planet whose temperature and radius are both known.
    deduped.sort(
        key=lambda x: (
            x["status"] != "unknown",
            x["habitability_score"] or 0.0,
            x["habitability_coverage"],
        ),
        reverse=True,
    )

    with _CACHE_LOCK:
        if _CACHE["signature"] == signature:
            _CACHE["records"] = deduped

    return deduped[:limit] if limit is not None else deduped


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

    q = _norm_name(query)
    if not q:
        return []

    hits = df["_planet_name_normalized"].str.contains(q, na=False, regex=False)

    # Also honour alternate designations (a KOI id, a Kepler name) that are not
    # the row's display name, including partial ones.
    alias_rows: set = set()
    for alias, rows in get_name_index().items():
        if q in alias:
            alias_rows.update(rows)
    if alias_rows:
        hits = hits | df.index.isin(sorted(alias_rows))

    matches = df[hits]

    # One row per planet, preferring the highest-priority source.
    matches = matches.sort_values("_source_priority", kind="stable")
    matches = matches.drop_duplicates(subset="_planet_name_normalized", keep="first").head(limit)

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
            "source_family": row.get("_source_family"),
        })

    return results


def list_planet_names(limit: int | None = None):
    """Return de-duplicated planet names across every loaded source."""
    df = load_planet_data()
    if df.empty:
        return []

    names = (
        df.sort_values("_source_priority", kind="stable")
        .drop_duplicates(subset="_planet_name_normalized", keep="first")[NAME_COLUMN]
        .astype(str)
        .sort_values()
        .tolist()
    )
    return names[:limit] if limit is not None else names


# =========================================================
# 🧠 GET PLANET INFO
# =========================================================
NUMERIC_FIELDS = {
    "mass_earth",
    "radius",
    "temperature",
    "period",
    "semi_major_axis",
    "eccentricity",
    "distance_pc",
    "star_temp",
    "star_radius",
    "transit_duration",
    "transit_depth",
    "transit_impact",
    "transit_snr",
    "star_logg",
    "star_magnitude",
    "insolation",
    "disposition_score",
}

INFO_FIELDS = (
    "mass_earth",
    "radius",
    "temperature",
    "period",
    "semi_major_axis",
    "eccentricity",
    "distance_pc",
    "star_temp",
    "star_radius",
    "host_star",
    "discovery_year",
    "discovery_method",
    "spectral_type",
    "transit_duration",
    "transit_depth",
    "transit_impact",
    "transit_snr",
    "star_logg",
    "star_magnitude",
    "insolation",
    "disposition_score",
)


def _field_from_row(row: pd.Series, field: str):
    value = _value_from_row(row, field)
    if field in NUMERIC_FIELDS:
        return _to_float(value)
    return _json_safe_scalar(value)


def get_planet_info(planet_name: str):
    """
    Look up one planet across every loaded source.

    The highest-priority source provides the base record; any field it is missing
    is filled from the remaining sources, and `field_sources` records where each
    value actually came from.
    """
    df = load_planet_data()
    if df.empty or not planet_name:
        return {}

    name_col = _find_name_column(df)
    if not name_col:
        return {}

    norm_query = _norm_name(planet_name)

    # Exact hit on any known designation for the planet.
    rows = get_name_index().get(norm_query)
    match = df.loc[rows] if rows else df.iloc[0:0]

    if match.empty:
        match = df[df["_planet_name_normalized"].str.contains(norm_query, na=False, regex=False)]
        if not match.empty:
            # Keep every row for the single closest planet, not a mix of partial matches.
            best_key = match.iloc[0]["_planet_name_normalized"]
            match = match[match["_planet_name_normalized"] == best_key]

    if match.empty:
        return {}

    match = match.sort_values("_source_priority", kind="stable")

    # The query may have arrived via an alias that only one source lists (a KOI
    # id, say). Re-resolve on the winning row's display name so the merged record
    # still draws on every source that knows this planet.
    canonical_key = match.iloc[0]["_planet_name_normalized"]
    match = df[df["_planet_name_normalized"] == canonical_key].sort_values(
        "_source_priority", kind="stable"
    )

    rows = [row for _, row in match.iterrows()]
    primary = rows[0]

    info: dict = {}
    field_sources: dict = {}
    for field in INFO_FIELDS:
        for row in rows:
            value = _field_from_row(row, field)
            if value is not None:
                info[field] = value
                field_sources[field] = _json_safe_scalar(row.get("_source"))
                break
        else:
            info[field] = None

    contributing_sources = []
    for row in rows:
        source = _json_safe_scalar(row.get("_source"))
        if source and source not in contributing_sources:
            contributing_sources.append(source)

    profile = compute_habitability_profile(info)
    info.update(
        {
            "planet_name": primary.get(name_col),
            "disposition": _json_safe_scalar(primary.get(DISPOSITION_COLUMN)),
            "distance_ly": parsecs_to_lightyears(info.get("distance_pc")),
            "planet_type": infer_planet_type(info.get("radius")),
            "water_likelihood": infer_water_likelihood(info.get("temperature")),
            "habitability_score": profile["score"],
            "habitability_coverage": profile["coverage"],
            "habitability_factors": profile["factors"],
            "habitability_missing": profile["missing"],
            "habitability_explanation": profile["explanation"],
            "status": profile["status"],
            "moons": "unknown (exomoons are extremely hard to detect with current technology)",
            "source": _json_safe_scalar(primary.get("_source")),
            "source_family": _json_safe_scalar(primary.get("_source_family")),
            "sources": contributing_sources,
            "field_sources": field_sources,
        }
    )
    return info


# =========================================================
# 🌍 HABITABILITY SCORING
# =========================================================
HABITABILITY_FACTORS = {
    "radius": {
        "weight": 0.30,
        "label": "Planet radius",
        "unit": "Earth radii",
        "reference": "1 Earth radius",
        "score": lambda v: max(0.0, 1 - abs(v - 1)),
    },
    "temperature": {
        "weight": 0.40,
        "label": "Equilibrium temperature",
        "unit": "K",
        "reference": "288 K (Earth average)",
        "score": lambda v: max(0.0, 1 - abs(v - 288) / 300),
    },
    "period": {
        "weight": 0.20,
        "label": "Orbital period",
        "unit": "days",
        "reference": "365 days",
        "score": lambda v: max(0.0, 1 - abs(v - 365) / 500),
    },
    "star_temp": {
        "weight": 0.10,
        "label": "Host star temperature",
        "unit": "K",
        "reference": "5778 K (the Sun)",
        "score": lambda v: max(0.0, 1 - abs(v - 5778) / 4000),
    },
}

# Below this share of the total weight, the inputs are too sparse to call.
MIN_HABITABILITY_COVERAGE = 0.5

TOTAL_HABITABILITY_WEIGHT = sum(f["weight"] for f in HABITABILITY_FACTORS.values())


def compute_habitability_profile(info: dict) -> dict:
    """
    Score habitability over the factors that are actually measured.

    The score is a weighted average across *available* factors rather than a sum
    over all of them, so a planet with no temperature reading is reported as
    unknown instead of being scored as though it were measured and hostile.
    """
    empty = {
        "score": None,
        "coverage": 0.0,
        "status": "unknown",
        "factors": [],
        "missing": list(HABITABILITY_FACTORS),
        "explanation": "No habitability inputs are available for this record.",
    }
    if not info:
        return empty

    factors = []
    missing = []
    weighted_total = 0.0
    available_weight = 0.0

    for key, spec in HABITABILITY_FACTORS.items():
        value = info.get(key)
        if value is None:
            missing.append(key)
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            missing.append(key)
            continue

        factor_score = min(max(spec["score"](value), 0.0), 1.0)
        weighted_total += spec["weight"] * factor_score
        available_weight += spec["weight"]
        factors.append(
            {
                "factor": key,
                "label": spec["label"],
                "value": round(value, 4),
                "unit": spec["unit"],
                "reference": spec["reference"],
                "weight": spec["weight"],
                "score": round(factor_score, 3),
            }
        )

    if available_weight <= 0:
        return empty

    score = round(weighted_total / available_weight, 3)
    coverage = round(available_weight / TOTAL_HABITABILITY_WEIGHT, 3)

    if coverage < MIN_HABITABILITY_COVERAGE:
        status = "unknown"
        explanation = (
            f"Only {int(coverage * 100)}% of the habitability inputs are measured for this "
            f"planet (missing: {', '.join(missing)}), which is too little to classify it."
        )
    else:
        if score >= 0.7:
            status = "habitable"
        elif score >= 0.3:
            status = "marginal"
        else:
            status = "inhospitable"
        explanation = (
            f"Scored {score:.2f} across {len(factors)} measured factor(s) covering "
            f"{int(coverage * 100)}% of the model's weighting."
        )
        if missing:
            explanation += f" Not measured: {', '.join(missing)}."

    return {
        "score": score,
        "coverage": coverage,
        "status": status,
        "factors": sorted(factors, key=lambda f: f["weight"], reverse=True),
        "missing": missing,
        "explanation": explanation,
    }


def compute_habitability_index(info: dict) -> float | None:
    """Backward-compatible scalar accessor. Returns None when nothing is measured."""
    return compute_habitability_profile(info)["score"]


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

    planet_name = info.get("planet_name", "This planet")
    host_star = info.get("host_star")
    if host_star:
        parts.append(f"{planet_name} orbits the star {host_star}.")
    else:
        parts.append(f"{planet_name} orbits a host star that this catalog does not name.")

    disposition = info.get("disposition")
    if disposition == "candidate":
        parts.append("It is an unconfirmed candidate, still awaiting validation.")
    elif disposition == "controversial":
        parts.append("Its status is disputed — some analyses question the detection.")

    planet_type = info.get("planet_type")
    if planet_type and planet_type != "unknown":
        parts.append(f"It is classified as a {planet_type} planet.")

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
