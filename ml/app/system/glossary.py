"""
Plain-language glossary.

Reports show terms like `koi_duration`, `disposition`, `parsec` and `imputed`.
Users reading a report need to be able to ask what those mean without leaving the
chat, so every term the UI can display should have an entry here.

Each entry may name a `planet_field`, which lets the chat layer follow the
definition with the active planet's own value for that quantity.
"""

import re

# term key -> definition, aliases, and the record field holding its value
TERMS: dict[str, dict] = {
    # ---------- transit measurements ----------
    "orbital_period": {
        "label": "Orbital period",
        "aliases": ["koi_period", "orbital period", "period", "pl_orbper", "year length"],
        "planet_field": "period",
        "unit": "days",
        "definition": (
            "How long the planet takes to complete one orbit around its star — its "
            "year, measured in Earth days. It is found by timing the gap between "
            "repeated transits. Short periods mean the planet orbits close in, which "
            "usually means hot."
        ),
    },
    "transit_duration": {
        "label": "Transit duration",
        "aliases": ["koi_duration", "transit duration", "duration", "pl_trandurh"],
        "planet_field": "transit_duration",
        "unit": "hours",
        "definition": (
            "How many hours the planet takes to cross the face of its star as seen "
            "from Earth. Together with the orbital period it constrains the size of "
            "the orbit and the angle we view it from. A duration that does not fit "
            "the period is a classic sign the signal is not really a planet."
        ),
    },
    "transit_depth": {
        "label": "Transit depth",
        "aliases": ["koi_depth", "transit depth", "depth", "pl_trandep"],
        "planet_field": "transit_depth",
        "unit": "parts per million",
        "definition": (
            "How much the star dims when the planet passes in front of it, in parts "
            "per million. A deeper dip means a larger planet relative to its star — "
            "this is the measurement planet radius is derived from. Very deep dips "
            "often turn out to be two stars eclipsing each other rather than a planet."
        ),
    },
    "impact_parameter": {
        "label": "Impact parameter",
        "aliases": ["koi_impact", "impact parameter", "impact"],
        "planet_field": "transit_impact",
        "unit": "0 = dead centre, 1 = grazing",
        "definition": (
            "How centrally the planet crosses the star's disc. 0 means it passes "
            "straight across the middle; near 1 means it only clips the edge. "
            "Grazing crossings are harder to interpret and are a common false-positive "
            "signature."
        ),
    },
    "signal_to_noise": {
        "label": "Signal-to-noise ratio",
        "aliases": ["koi_model_snr", "signal to noise", "signal-to-noise", "snr"],
        "planet_field": "transit_snr",
        "unit": "ratio",
        "definition": (
            "How strong the transit signal is compared with the background noise in "
            "the light curve. Higher means a cleaner, more trustworthy detection. Low "
            "values are where marginal candidates and spurious detections live."
        ),
    },
    # ---------- planet properties ----------
    "planet_radius": {
        "label": "Planet radius",
        "aliases": ["koi_prad", "planet radius", "radius", "pl_rade", "size"],
        "planet_field": "radius",
        "unit": "Earth radii",
        "definition": (
            "The planet's size relative to Earth, where 1.0 is exactly Earth-sized. "
            "Below about 1.6 a planet is usually rocky; beyond about 4 it is a gas "
            "giant with no surface to stand on. This is the single biggest factor in "
            "whether a world could be Earth-like."
        ),
    },
    "equilibrium_temperature": {
        "label": "Equilibrium temperature",
        "aliases": [
            "koi_teq",
            "equilibrium temperature",
            "temperature",
            "pl_eqt",
            "teq",
            "how hot",
        ],
        "planet_field": "temperature",
        "unit": "Kelvin",
        "definition": (
            "An estimate of the planet's temperature based purely on how much "
            "starlight it receives, in Kelvin (0 K is absolute zero, water freezes at "
            "273 K and boils at 373 K). It ignores any atmosphere, so a real "
            "greenhouse effect would make the surface hotter. It is a climate clue, "
            "not a measured surface reading."
        ),
    },
    "insolation": {
        "label": "Insolation",
        "aliases": ["koi_insol", "insolation", "starlight received", "pl_insol"],
        "planet_field": "insolation",
        "unit": "× Earth",
        "definition": (
            "How much starlight the planet receives compared with Earth. A value of 1 "
            "means it gets the same energy we do; 100 means a hundred times as much. "
            "This is what sets the habitable zone."
        ),
    },
    "semi_major_axis": {
        "label": "Semi-major axis",
        "aliases": ["semi-major axis", "semi major axis", "orbital distance", "pl_orbsmax"],
        "planet_field": "semi_major_axis",
        "unit": "AU",
        "definition": (
            "The average distance between the planet and its star, in astronomical "
            "units, where 1 AU is the Earth-Sun distance. It is the size of the orbit "
            "rather than the distance from us."
        ),
    },
    "eccentricity": {
        "label": "Eccentricity",
        "aliases": ["eccentricity", "pl_orbeccen", "how circular"],
        "planet_field": "eccentricity",
        "unit": "0 = circular",
        "definition": (
            "How stretched the orbit is. 0 is a perfect circle; values approaching 1 "
            "are long ellipses that swing the planet through extreme temperature "
            "changes each orbit, which is hard on any stable climate."
        ),
    },
    # ---------- star properties ----------
    "star_temperature": {
        "label": "Host star temperature",
        "aliases": ["koi_steff", "star temperature", "stellar temperature", "st_teff", "steff"],
        "planet_field": "star_temp",
        "unit": "Kelvin",
        "definition": (
            "The surface temperature of the star the planet orbits. The Sun is about "
            "5778 K. Cooler red dwarfs run near 3000 K and are dim but extremely "
            "long-lived; hot blue stars burn out fast, leaving little time for life."
        ),
    },
    "star_radius": {
        "label": "Host star radius",
        "aliases": ["koi_srad", "star radius", "stellar radius", "st_rad", "srad"],
        "planet_field": "star_radius",
        "unit": "solar radii",
        "definition": (
            "The size of the host star compared with the Sun, where 1.0 is Sun-sized. "
            "It matters because transit depth only gives the planet's size *relative* "
            "to the star — the star's size converts that into a real planet radius."
        ),
    },
    "surface_gravity": {
        "label": "Stellar surface gravity",
        "aliases": ["koi_slogg", "surface gravity", "logg", "st_logg", "slogg"],
        "planet_field": "star_logg",
        "unit": "log g (cgs)",
        "definition": (
            "The star's surface gravity on a logarithmic scale. It helps separate "
            "compact main-sequence stars from bloated giants — useful because a giant "
            "star can mimic a planet transit."
        ),
    },
    "magnitude": {
        "label": "Stellar magnitude",
        "aliases": ["koi_kepmag", "kepmag", "magnitude", "brightness", "st_tmag"],
        "planet_field": "star_magnitude",
        "unit": "magnitude",
        "definition": (
            "How bright the star appears, on the astronomical magnitude scale where "
            "*lower numbers mean brighter*. Brighter stars give cleaner measurements "
            "and are easier to follow up with other telescopes."
        ),
    },
    "spectral_type": {
        "label": "Spectral type",
        "aliases": ["spectral type", "star class", "star family", "st_spectype"],
        "planet_field": "spectral_type",
        "unit": None,
        "definition": (
            "The star's classification letter — O, B, A, F, G, K, M — running from "
            "hottest to coolest. The Sun is a G star. M dwarfs are the most common "
            "hosts for small planets simply because they are the most common stars."
        ),
    },
    # ---------- distance ----------
    "parsec": {
        "label": "Parsec",
        "aliases": ["parsec", "parsecs", "pc", "distance_pc"],
        "planet_field": "distance_pc",
        "unit": "parsecs",
        "definition": (
            "A distance unit astronomers use: 1 parsec is about 3.26 light-years, or "
            "roughly 31 trillion kilometres. It comes from parallax — the tiny shift "
            "in a star's apparent position as Earth moves around the Sun."
        ),
    },
    "light_year": {
        "label": "Light-year",
        "aliases": ["light-year", "light year", "light years", "lightyear", "ly"],
        "planet_field": "distance_ly",
        "unit": "light-years",
        "definition": (
            "The distance light travels in one year, about 9.46 trillion kilometres. "
            "It is also a look-back time: a planet 100 light-years away is seen as it "
            "was 100 years ago."
        ),
    },
    # ---------- detection methods ----------
    "transit_method": {
        "label": "Transit method",
        "aliases": ["transit method", "transit", "transiting"],
        "planet_field": None,
        "unit": None,
        "definition": (
            "Finding planets by watching a star dim slightly each time a planet "
            "crosses in front of it. It only works when the orbit happens to be edge-on "
            "from our viewpoint, but it reveals the planet's size — and it is how "
            "Kepler and TESS found most known planets."
        ),
    },
    "radial_velocity": {
        "label": "Radial velocity",
        "aliases": ["radial velocity", "rv method", "doppler", "wobble"],
        "planet_field": None,
        "unit": None,
        "definition": (
            "Finding planets by detecting the star's small wobble as the planet's "
            "gravity tugs it back and forth, measured as a Doppler shift in the star's "
            "light. It reveals the planet's mass, where transits reveal size."
        ),
    },
    "microlensing": {
        "label": "Microlensing",
        "aliases": ["microlensing", "gravitational lensing"],
        "planet_field": None,
        "unit": None,
        "definition": (
            "Finding planets when a foreground star's gravity magnifies a background "
            "star, with a planet adding a brief extra spike. It reaches very distant "
            "planets, but the alignment never repeats, so these detections cannot be "
            "re-observed."
        ),
    },
    # ---------- this tool's own vocabulary ----------
    "habitability_index": {
        "label": "Habitability index",
        "aliases": ["habitability index", "habitability score", "habitability"],
        "planet_field": "habitability_score",
        "unit": "0 to 1",
        "definition": (
            "This project's own heuristic score from 0 to 1. It is a weighted average "
            "over the factors actually measured for a planet: equilibrium temperature "
            "(40%), radius (30%), orbital period (20%) and host-star temperature (10%). "
            "It is a rough Earth-similarity indicator, not a prediction of life, and it "
            "is only computed over measured factors — never over guesses."
        ),
    },
    "coverage": {
        "label": "Input coverage",
        "aliases": ["coverage", "inputs measured", "% of inputs", "data completeness"],
        "planet_field": "habitability_coverage",
        "unit": "percent",
        "definition": (
            "The share of the habitability weighting that came from real measurements "
            "for this planet. At 100% every factor was measured. Below 50% the planet "
            "is reported as 'not classified' rather than scored, because too little is "
            "known to make the call."
        ),
    },
    "disposition": {
        "label": "Disposition",
        "aliases": ["disposition", "catalog status", "confirmed", "candidate", "controversial"],
        "planet_field": "disposition",
        "unit": None,
        "definition": (
            "What the catalog says about whether the object is really a planet. "
            "'confirmed' means it has been validated by follow-up work. 'candidate' "
            "means it looks like a planet but has not been confirmed yet. "
            "'controversial' means published analyses disagree. Objects vetted as false "
            "positives are excluded from this catalog entirely."
        ),
    },
    "false_positive": {
        "label": "False positive",
        "aliases": ["false positive", "false alarm"],
        "planet_field": None,
        "unit": None,
        "definition": (
            "A signal that looked like a planet but turned out to be something else — "
            "most often two stars eclipsing each other, a background binary, or an "
            "instrument artefact. These are filtered out of this catalog."
        ),
    },
    "predicted_class": {
        "label": "Predicted class",
        "aliases": ["predicted class", "class", "classification"],
        "planet_field": None,
        "unit": None,
        "definition": (
            "The detection model's verdict. Class 1 means the signal looks like a "
            "genuine planet detection; class 0 means it resembles a false positive. "
            "This is about whether the *detection* is real — it says nothing about "
            "whether the planet is habitable."
        ),
    },
    "confidence": {
        "label": "Confidence",
        "aliases": ["confidence", "certainty"],
        "planet_field": None,
        "unit": "0 to 1",
        "definition": (
            "How strongly the detection model prefers its predicted class. It is the "
            "model's internal certainty about the detection being genuine — not a "
            "probability that the planet is habitable, and not a guarantee."
        ),
    },
    "nasa_vetting_score": {
        "label": "NASA vetting score",
        "aliases": ["koi_score", "vetting score", "nasa score", "disposition score"],
        "planet_field": "disposition_score",
        "unit": "0 to 1",
        "definition": (
            "NASA's own published confidence that a Kepler Object of Interest is a real "
            "planet, from 0 to 1. It is independent of this project's model, so it is "
            "a useful second opinion where available."
        ),
    },
    "imputed": {
        "label": "Imputed input",
        "aliases": ["imputed", "median", "measured vs imputed", "input basis"],
        "planet_field": None,
        "unit": None,
        "definition": (
            "A value that was not measured for this planet, so the typical value from "
            "the training data was substituted. Imputed inputs describe the dataset "
            "rather than the planet, which is why reports mark them — a prediction "
            "resting mostly on imputed values is not really about that planet."
        ),
    },
    "host_star": {
        "label": "Host star",
        "aliases": ["host star", "hostname", "parent star"],
        "planet_field": "host_star",
        "unit": None,
        "definition": (
            "The star the planet orbits. Planets are normally named after their host "
            "with a trailing letter: Kepler-442 b is the first planet found around the "
            "star Kepler-442."
        ),
    },
}


def _normalize(text: str) -> str:
    return re.sub(r"[_\-]+", " ", str(text or "").lower()).strip()


# Longest aliases first so "transit duration" wins over "duration".
_ALIAS_INDEX = sorted(
    ((alias, key) for key, spec in TERMS.items() for alias in spec["aliases"]),
    key=lambda pair: -len(pair[0]),
)

DEFINITION_PATTERNS = [
    r"\bwhat (?:is|are|does|do)\b",
    r"\bwhat'?s\b",
    r"\bmean(?:s|ing)?\b",
    r"\bexplain\b",
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\btell me what\b",
    r"\bhow do (?:i|you) read\b",
    r"\bstand(?:s)? for\b",
]


def is_definition_question(message: str) -> bool:
    lowered = _normalize(message)
    return any(re.search(pattern, lowered) for pattern in DEFINITION_PATTERNS)


def find_terms(message: str, limit: int = 3) -> list[str]:
    """Return glossary keys mentioned in the message, best match first."""
    lowered = _normalize(message)
    if not lowered:
        return []

    found: list[str] = []
    consumed = ""
    for alias, key in _ALIAS_INDEX:
        if key in found:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
            # Skip an alias already covered by a longer match ("duration" inside
            # "transit duration").
            if alias in consumed:
                continue
            found.append(key)
            consumed += f" {alias}"
            if len(found) >= limit:
                break
    return found


def describe(key: str) -> dict | None:
    spec = TERMS.get(key)
    if not spec:
        return None
    return {
        "term": key,
        "label": spec["label"],
        "unit": spec["unit"],
        "definition": spec["definition"],
        "planet_field": spec["planet_field"],
    }


def all_terms() -> list[str]:
    return sorted(spec["label"] for spec in TERMS.values())
