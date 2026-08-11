"""
Model input construction.

The classifier is fitted on the Kepler KOI feature space. Catalog records use
different column names and carry only some of those quantities, so this module
projects a planet record into the model's feature order and reports, per feature,
whether the value came from the planet's own measurements or from a training-set
median.

That provenance is the point: a prediction built mostly from medians is not
really about the planet, and the UI should be able to say so.
"""

import json
from pathlib import Path

import pandas as pd

from app.system.selfaware import AWARENESS_FILE, update_awareness_state

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAINING_FALLBACK = DATA_DIR / "koi_fallback.csv"

# Model feature -> logical field in a planet record (see planet_knowledge
# COLUMN_ALIASES). Features absent from this map cannot be derived from the
# catalog and always fall back to the training median.
MODEL_FEATURE_SOURCES = {
    "koi_period": "period",
    "koi_prad": "radius",
    "koi_teq": "temperature",
    "koi_steff": "star_temp",
    "koi_srad": "star_radius",
    "koi_duration": "transit_duration",
    "koi_depth": "transit_depth",
    "koi_impact": "transit_impact",
    "koi_model_snr": "transit_snr",
    "koi_slogg": "star_logg",
    "koi_kepmag": "star_magnitude",
}

_MEDIAN_CACHE: dict | None = None


def _awareness() -> dict:
    if not AWARENESS_FILE.exists():
        return {}
    try:
        with open(AWARENESS_FILE, "r") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _medians_from_training_file(feature_names: list[str]) -> dict:
    """
    Derive medians directly from the bundled training data.

    Used when the model was trained before medians were recorded, so explanations
    work without forcing a retrain.
    """
    if not TRAINING_FALLBACK.exists():
        return {}
    try:
        df = pd.read_csv(TRAINING_FALLBACK, comment="#", on_bad_lines="skip")
    except Exception:
        return {}

    medians = {}
    for name in feature_names:
        if name in df.columns:
            value = pd.to_numeric(df[name], errors="coerce").median()
            if pd.notna(value):
                medians[name] = float(value)
    return medians


def get_feature_medians(feature_names: list[str]) -> dict:
    """Training-set median per model feature, cached for the process lifetime."""
    global _MEDIAN_CACHE
    if _MEDIAN_CACHE is not None:
        return _MEDIAN_CACHE

    medians = _awareness().get("feature_medians") or {}
    medians = {k: float(v) for k, v in medians.items() if isinstance(v, (int, float))}

    missing = [name for name in feature_names if name not in medians]
    if missing:
        derived = _medians_from_training_file(feature_names)
        for name in missing:
            if name in derived:
                medians[name] = derived[name]
        if derived:
            # Persist so the next process does not repeat the work.
            try:
                update_awareness_state(feature_medians=medians)
            except Exception:
                pass

    _MEDIAN_CACHE = medians
    return medians


def reset_feature_median_cache():
    global _MEDIAN_CACHE
    _MEDIAN_CACHE = None


def build_feature_vector(planet_info: dict, feature_names: list[str]):
    """
    Project a planet record into the model's feature order.

    Returns (values, provenance) where provenance is one entry per feature
    describing where its value came from.
    """
    planet_info = planet_info or {}
    medians = get_feature_medians(feature_names)

    values = []
    provenance = []

    for name in feature_names:
        logical = MODEL_FEATURE_SOURCES.get(name)
        raw = planet_info.get(logical) if logical else None

        if raw is not None:
            try:
                value = float(raw)
                origin = "planet"
            except (TypeError, ValueError):
                value = None
                origin = None
        else:
            value = None
            origin = None

        if value is None:
            value = float(medians.get(name, 0.0))
            origin = "median" if name in medians else "zero"

        values.append(value)
        provenance.append(
            {
                "feature": name,
                "value": value,
                "origin": origin,
                "catalog_field": logical,
            }
        )

    return values, provenance


def model_gain_weights(model, feature_names: list[str]) -> dict:
    """
    Share of the model's total split gain attributable to each feature.

    Counting inputs is not enough: supplying 5 of 18 features matters a lot if
    they are the ones the model splits on, and not at all if they aren't.
    """
    try:
        raw = model.get_booster().get_score(importance_type="gain")
    except Exception:
        return {}

    total = sum(raw.values())
    if not total:
        return {}

    weights = {}
    for key, value in raw.items():
        if key.startswith("f") and key[1:].isdigit():
            index = int(key[1:])
            if index < len(feature_names):
                weights[feature_names[index]] = value / total
        elif key in feature_names:
            weights[key] = value / total
    return weights


def summarize_provenance(provenance: list[dict], gain_weights: dict | None = None) -> dict:
    """Condense per-feature provenance into something a report can state plainly."""
    from_planet = [p["feature"] for p in provenance if p["origin"] == "planet"]
    imputed = [p["feature"] for p in provenance if p["origin"] != "planet"]
    total = len(provenance)
    count = len(from_planet)

    gain_weights = gain_weights or {}
    influence = sum(gain_weights.get(name, 0.0) for name in from_planet)
    influence = round(influence, 3)

    if total == 0:
        basis = "no model inputs were available"
    elif count == 0:
        basis = (
            "none of the model's inputs could be filled from this planet's record, "
            "so the prediction reflects the training-set average rather than this planet"
        )
    else:
        basis = (
            f"{count} of {total} model inputs came from this planet's measurements "
            f"({', '.join(from_planet)})"
        )
        if gain_weights:
            basis += (
                f", accounting for {int(influence * 100)}% of the model's decision weight; "
                f"the remaining inputs are training-set medians"
            )
        else:
            basis += "; the rest are training-set medians"

    if influence >= 0.6:
        quality = "planet-specific"
    elif influence >= 0.3:
        quality = "partly planet-specific"
    else:
        quality = "not planet-specific"

    return {
        "inputs_total": total,
        "inputs_from_planet": count,
        "planet_features": from_planet,
        "imputed_features": imputed,
        "coverage": round(count / total, 3) if total else 0.0,
        "influence_coverage": influence,
        "quality": quality,
        "basis": basis,
    }
