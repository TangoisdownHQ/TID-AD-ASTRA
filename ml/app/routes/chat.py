import re
from math import ceil
from uuid import uuid4

from fastapi import APIRouter

from app.models.classifier import explain_prediction
from app.schemas import ChatRequest
from app.system.enrichment import enrich_planet_info
from app.system.openai_chat import openai_available, render_openai_answer
from app.system import glossary
from app.system.planet_knowledge import (
    get_planet_catalog_records,
    get_planet_info,
    narrative_summary,
)

router = APIRouter(tags=["Chat"])

PLANET_LIMIT_HARD_CAP = 20
CHAT_SESSIONS: dict[str, dict] = {}
ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
EXPLANATION_KEYWORDS = {
    "feature",
    "features",
    "importance",
    "important",
    "class",
    "confidence",
    "index",
    "habitability index",
    "temperature",
    "kelvin",
    "parsec",
    "parsecs",
    "light-year",
    "light years",
    "meaning",
    "mean",
    "means",
    "star family",
    "star class",
    "spectral",
    "source family",
}


def _extract_limit(message: str, default: int) -> int:
    match = re.search(r"\b(?:top|show|list|give me)\s+(\d{1,2})\b", message)
    if match:
        return max(1, min(int(match.group(1)), PLANET_LIMIT_HARD_CAP))
    return max(1, min(default, PLANET_LIMIT_HARD_CAP))


def _extract_distance_limit(message: str):
    match = re.search(r"\b(?:within|under|below)\s+(\d+(?:\.\d+)?)\s*(ly|light[- ]?years?|pc|parsecs?)\b", message)
    if not match:
        return None, None
    value = float(match.group(1))
    unit = match.group(2)
    return value, unit


def _extract_year_bounds(message: str):
    year_after = None
    year_before = None

    after_match = re.search(r"\b(?:after|since|newer than)\s+(20\d{2}|19\d{2})\b", message)
    before_match = re.search(r"\b(?:before|older than|earlier than)\s+(20\d{2}|19\d{2})\b", message)

    if after_match:
        year_after = int(after_match.group(1))
    if before_match:
        year_before = int(before_match.group(1))

    return year_after, year_before


def _extract_radius_bounds(message: str):
    max_match = re.search(r"\b(?:radius under|radius below|smaller than|under)\s+(\d+(?:\.\d+)?)\s*(?:earth|radii|radius)?\b", message)
    min_match = re.search(r"\b(?:radius over|radius above|larger than|bigger than|over)\s+(\d+(?:\.\d+)?)\s*(?:earth|radii|radius)?\b", message)

    radius_min = float(min_match.group(1)) if min_match else None
    radius_max = float(max_match.group(1)) if max_match else None
    return radius_min, radius_max


def _extract_discovery_method(message: str):
    methods = {
        "transit": "transit",
        "radial velocity": "radial velocity",
        "rv": "radial velocity",
        "microlensing": "microlensing",
        "imaging": "imaging",
        "astrometry": "astrometry",
    }
    for needle, value in methods.items():
        if needle in message:
            return value
    return None


def _collect_mentioned_planets(message: str, records: list[dict]) -> list[str]:
    normalized_message = re.sub(r"[^a-z0-9]+", "", message.lower())
    mentions = []
    for record in records:
        normalized_name = record["planet_name_normalized"]
        if normalized_name and normalized_name in normalized_message:
            mentions.append((normalized_message.index(normalized_name), -len(normalized_name), record["planet_name"]))
    deduped = []
    seen = set()
    for _, _, name in sorted(mentions):
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _requested_status(message: str) -> str | None:
    """
    Work out which habitability status the user is asking for.

    Negative forms are checked first on purpose: "unhabitable" and
    "uninhabitable" both contain "habitable" as a substring, so a naive check
    returns the *most* habitable planets for a question asking for the opposite.
    """
    negatives = [
        "uninhabitable",
        "unhabitable",
        "not habitable",
        "non-habitable",
        "nonhabitable",
        "inhospitable",
        "hostile",
        "unlivable",
        "uninhabited",
        "least habitable",
        "worst",
    ]
    if any(token in message for token in negatives):
        return "inhospitable"

    if "marginal" in message or "borderline" in message:
        return "marginal"

    if any(token in message for token in ["unknown", "unclassified", "not classified", "no data"]):
        return "unknown"

    if re.search(r"\bhabitable\b", message) or "livable" in message or "life" in message:
        return "habitable"

    return None


def _filter_records(message: str, records: list[dict]):
    filtered = records[:]

    requested_status = _requested_status(message)
    if requested_status:
        filtered = [r for r in filtered if r.get("status") == requested_status]

    if "rocky" in message or "earth-like" in message:
        filtered = [r for r in filtered if "rocky" in (r.get("planet_type") or "")]
    elif "gas giant" in message:
        filtered = [r for r in filtered if "gas giant" in (r.get("planet_type") or "")]

    discovery_method = _extract_discovery_method(message)
    if discovery_method:
        filtered = [
            r for r in filtered
            if discovery_method in str(r.get("discovery_method") or "").lower()
        ]

    year_after, year_before = _extract_year_bounds(message)
    if year_after is not None:
        filtered = [r for r in filtered if r.get("discovery_year") is not None and float(r["discovery_year"]) >= year_after]
    if year_before is not None:
        filtered = [r for r in filtered if r.get("discovery_year") is not None and float(r["discovery_year"]) <= year_before]

    radius_min, radius_max = _extract_radius_bounds(message)
    if radius_min is not None:
        filtered = [r for r in filtered if r.get("radius") is not None and float(r["radius"]) >= radius_min]
    if radius_max is not None:
        filtered = [r for r in filtered if r.get("radius") is not None and float(r["radius"]) <= radius_max]

    distance_value, distance_unit = _extract_distance_limit(message)
    if distance_value is not None:
        distance_key = "distance_ly" if distance_unit.startswith("ly") or "light" in distance_unit else "distance_pc"
        filtered = [r for r in filtered if r.get(distance_key) is not None and float(r[distance_key]) <= distance_value]

    return filtered


def _sort_records(message: str, records: list[dict]) -> list[dict]:
    if "nearby" in message or "closest" in message or "nearest" in message:
        return sorted(records, key=lambda r: (r.get("distance_ly") is None, r.get("distance_ly") or 10**9))
    if "latest" in message or "newest" in message or "recent" in message:
        return sorted(records, key=lambda r: (r.get("discovery_year") is None, -(r.get("discovery_year") or 0)))
    # When the question is about hostile worlds, "best match" means lowest score.
    if _requested_status(message) == "inhospitable":
        return sorted(
            records,
            key=lambda r: (r.get("status") == "unknown", r.get("habitability_score") or 0),
        )

    # Otherwise match the catalog ordering: classified planets outrank
    # unclassifiable ones, then by score.
    return sorted(
        records,
        key=lambda r: (r.get("status") != "unknown", r.get("habitability_score") or 0),
        reverse=True,
    )


def _intent_for_message(message: str, planet_mentions: list[str]) -> str:
    if message in {"next", "next page", "more", "more results", "previous", "prev", "back"}:
        return "paginate"
    if re.search(r"\bpage\s+\d+\b", message):
        return "paginate"
    if len(planet_mentions) >= 2 and ("compare" in message or " vs " in message or " versus " in message):
        return "compare"
    if any(token in message for token in ["compare", " vs ", " versus "]):
        return "compare"
    if planet_mentions and any(token in message for token in ["tell me", "about", "info", "details", "what is", "who is"]):
        return "info"
    if re.search(r"\b(first|second|third|fourth|fifth|\d+)(?:\s+result|\s+planet)?\b", message) and any(
        token in message for token in ["tell me", "about", "details", "analyze", "inspect", "info"]
    ):
        return "reference"
    if any(token in message for token in ["find", "search", "show", "list", "top", "best", "nearby", "closest", "habitable"]):
        return "search"
    if planet_mentions:
        return "info"
    return "search"


def _is_explanation_question(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in EXPLANATION_KEYWORDS)


def _habitability_index_explanation(score, analysis: dict | None = None) -> str:
    analysis = analysis or {}
    if score is None:
        missing = analysis.get("habitability_missing") or []
        detail = f" Missing inputs: {', '.join(missing)}." if missing else ""
        return (
            "There is not enough measured data to score this planet's habitability. "
            "That is different from scoring badly — it means the measurements are absent."
            + detail
        )

    coverage = analysis.get("habitability_coverage")
    if score >= 0.7:
        band = "high"
    elif score >= 0.3:
        band = "moderate"
    else:
        band = "low"

    text = (
        "The habitability index is a heuristic score from 0 to 1, a weighted average over the "
        "factors actually measured for a planet: radius (30%), equilibrium temperature (40%), "
        "orbital period (20%), and host-star temperature (10%). "
        f"{score:.2f} falls in the {band} range, so it suggests potentially favorable conditions "
        "but not proof of life."
    )
    if coverage is not None:
        text += f" This score is based on {int(coverage * 100)}% of the weighting being measured."
    factors = analysis.get("habitability_factors") or []
    if factors:
        parts = [f"{f['label']} scored {f['score']:.2f}" for f in factors[:3]]
        text += " Breakdown: " + "; ".join(parts) + "."
    return text


def _temperature_explanation(info: dict) -> str:
    temp = info.get("temperature")
    if temp is None:
        return "Temperature data is missing for this planet."
    celsius = temp - 273.15
    return (
        f"The listed temperature is {temp:.1f} K, which is about {celsius:.1f} C. "
        "Here it is an equilibrium estimate based on incoming starlight, so it is a rough climate clue rather than a measured ground temperature."
    )


def _distance_explanation(info: dict) -> str:
    pc = info.get("distance_pc")
    ly = info.get("distance_ly")
    if pc is None and ly is None:
        return "Distance data is missing for this planet."
    if pc is not None and ly is not None:
        return f"The distance is {pc:.3f} parsecs, or about {ly:.1f} light-years. One parsec is about 3.26 light-years."
    if pc is not None:
        return f"The distance is {pc:.3f} parsecs. One parsec is about 3.26 light-years."
    return f"The distance is about {ly:.1f} light-years."


def _star_family_explanation(info: dict) -> str:
    spectral_type = info.get("spectral_type")
    star_temp = info.get("star_temp")
    if spectral_type:
        return f"The host star family or class is given by its spectral type: {spectral_type}."
    if star_temp is not None:
        return (
            f"No explicit spectral class is stored here, but the host star temperature is {star_temp:.0f} K. "
            "Astronomers use temperature together with spectra to group stars into families such as M, K, G, F, A, B, and O."
        )
    return "No star family or spectral class is available for this planet in the current dataset."


def _feature_explanation(analysis: dict) -> str:
    details = analysis.get("top_feature_details") or []
    raw = analysis.get("top_features") or {}
    if not raw:
        return "Feature importance shows which model inputs influenced this prediction most, but this result does not include feature-importance data."
    parts = [
        "Feature importance ranks which inputs pushed the model most strongly for this prediction. The numbers are relative influence scores, not percentages."
    ]
    for detail in details[:3]:
        parts.append(f"{detail.get('label')}: {detail.get('description')}")
    return " ".join(parts)


def _confidence_explanation(analysis: dict) -> str:
    confidence = analysis.get("confidence")
    if confidence is None:
        return "Confidence is unavailable for this result."

    text = (
        f"The confidence score is {confidence:.2f}. It is the classifier's certainty that this "
        "signal is a genuine planet detection rather than a false positive — not a statement "
        "about habitability."
    )
    inputs = analysis.get("model_inputs") or {}
    if inputs.get("quality") == "not planet-specific":
        text += (
            " Treat it with caution here: most of the model's decision weight came from "
            "training-set medians rather than this planet's own measurements, so the number "
            "is close to what the model returns for any catalog entry."
        )
    elif inputs.get("basis"):
        text += f" Basis: {inputs['basis']}."
    return text


def _class_explanation(analysis: dict) -> str:
    predicted_label = analysis.get("predicted_label")
    explanation = analysis.get("predicted_class_explanation")
    if predicted_label is None:
        return "The model did not return a class label."
    return f"The predicted class is {predicted_label}. {explanation or ''}".strip()


def _source_family_explanation(info: dict) -> str:
    source_family = info.get("source_family")
    source = info.get("source")
    if source_family:
        return f"Source family `{source_family}` means this record came from the broader data category behind the source, here `{source}`."
    if source:
        return f"The source is `{source}`, but there is no separate source-family label stored for this record."
    return "There is no source-family label stored for this record."


def _build_explanation_answer(message: str, session_id: str, session: dict):
    active_planet = session.get("active_planet")
    if not active_planet:
        return {
            "intent": "explain",
            "session_id": session_id,
            "answer": "Ask about a specific planet first, or analyze one from the current results, then I can explain what the fields mean.",
        }

    info = session.get("last_planet_info")
    analysis = session.get("last_analysis")
    if not info or (info.get("planet_name") or "").lower() != active_planet.lower():
        info = enrich_planet_info(get_planet_info(active_planet), include_external=True)
    if not analysis:
        analysis = _safe_explain(active_planet)

    lowered = message.lower()
    requested = []
    if "feature" in lowered or "importance" in lowered:
        requested.append(_feature_explanation(analysis))
    if "class" in lowered:
        requested.append(_class_explanation(analysis))
    if "confidence" in lowered:
        requested.append(_confidence_explanation(analysis))
    if "index" in lowered or "habitability index" in lowered:
        requested.append(_habitability_index_explanation(analysis.get("habitability_index"), analysis))
    if "temperature" in lowered or "kelvin" in lowered:
        requested.append(_temperature_explanation(info))
    if "parsec" in lowered or "parsecs" in lowered or "light-year" in lowered or "light years" in lowered:
        requested.append(_distance_explanation(info))
    if "star family" in lowered or "star class" in lowered or "spectral" in lowered:
        requested.append(_star_family_explanation(info))
    if "source family" in lowered:
        requested.append(_source_family_explanation(info))

    if not requested:
        # Fall back to the glossary before dumping every explanation at the user.
        terms = glossary.find_terms(lowered)
        for key in terms:
            spec = glossary.describe(key)
            if not spec:
                continue
            text = f"{spec['label']}: {spec['definition']}"
            planet_value = _format_term_value(spec, info)
            if planet_value:
                text += f" {planet_value}"
            requested.append(text)

    if not requested:
        requested = [
            _class_explanation(analysis),
            _confidence_explanation(analysis),
            _habitability_index_explanation(analysis.get("habitability_index"), analysis),
            _temperature_explanation(info),
            _distance_explanation(info),
            _feature_explanation(analysis),
        ]

    return {
        "intent": "explain",
        "session_id": session_id,
        "answer": f"For {active_planet}: " + " ".join(part for part in requested if part),
        "planet": info,
        "analysis": analysis,
    }


def _format_term_value(spec: dict, info: dict):
    """Render the active planet's value for a glossary term, when it has one."""
    field = spec.get("planet_field")
    if not field or not info:
        return None
    value = info.get(field)
    if value is None:
        return f"{info.get('planet_name', 'This planet')} has no measured value for this."

    if isinstance(value, float):
        rendered = f"{value:,.4g}"
    else:
        rendered = str(value)

    unit = spec.get("unit")
    if unit and unit not in {None, "0 to 1", "0 = circular", "0 = dead centre, 1 = grazing", "percent"}:
        rendered = f"{rendered} {unit}"

    return f"For {info.get('planet_name', 'this planet')} it is {rendered}."


def _build_glossary_answer(message: str, session_id: str, session: dict, terms: list[str]):
    info = session.get("last_planet_info") or {}
    active_planet = session.get("active_planet")

    parts = []
    described = []
    for key in terms:
        spec = glossary.describe(key)
        if not spec:
            continue
        planet_value = _format_term_value(spec, info) if active_planet else None
        spec["planet_value"] = planet_value
        described.append(spec)

        text = f"{spec['label']}: {spec['definition']}"
        if planet_value:
            text += f" {planet_value}"
        parts.append(text)

    if not parts:
        return {
            "intent": "glossary",
            "session_id": session_id,
            "answer": (
                "I do not have a definition for that yet. I can explain terms like "
                + ", ".join(glossary.all_terms()[:8])
                + ", and others shown in planet reports."
            ),
            "terms": [],
        }

    return {
        "intent": "glossary",
        "session_id": session_id,
        "answer": "\n\n".join(parts),
        "terms": described,
        "active_planet": active_planet,
    }


def _safe_explain(planet_name: str):
    try:
        return explain_prediction([], planet_name=planet_name)
    except Exception as exc:
        return {
            "planet_name": planet_name,
            "summary": f"Model explanation unavailable for {planet_name}.",
            "error": str(exc),
            "habitability_index": None,
            "confidence": None,
            "top_features": {},
            "model": "Unavailable",
        }


def _session_for(session_id: str | None):
    if session_id and session_id in CHAT_SESSIONS:
        return session_id, CHAT_SESSIONS[session_id]
    session_id = session_id or str(uuid4())
    CHAT_SESSIONS[session_id] = {}
    return session_id, CHAT_SESSIONS[session_id]


def _extract_reference_indexes(message: str) -> list[int]:
    indexes = []
    for word, value in ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", message):
            indexes.append(value)
    for match in re.findall(r"\b(\d{1,2})\b", message):
        indexes.append(int(match))
    deduped = []
    seen = set()
    for value in indexes:
        if value < 1 or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _build_page_payload(session_id: str, session: dict, page: int | None = None):
    results = session.get("results") or []
    limit = session.get("limit", 5)
    if not results:
        return {
            "intent": "paginate",
            "session_id": session_id,
            "answer": "There are no stored search results in this chat session yet.",
            "results": [],
        }

    total_pages = max(1, ceil(len(results) / limit))
    page = page or session.get("page", 1)
    page = min(max(page, 1), total_pages)
    session["page"] = page

    start = (page - 1) * limit
    end = start + limit
    page_results = results[start:end]
    answer = f"Showing page {page} of {total_pages}."
    if page_results:
        answer += " " + ", ".join(item["planet_name"] for item in page_results[:3]) + "."

    return {
        "intent": "search",
        "session_id": session_id,
        "answer": answer,
        "results": page_results,
        "page": page,
        "total_pages": total_pages,
        "total_matches": len(results),
    }


def _build_search_answer(message: str, records: list[dict], limit: int, session_id: str, session: dict):
    filtered = _sort_records(message, _filter_records(message, records))
    session["results"] = filtered
    session["limit"] = limit
    session["last_search_message"] = message
    session["page"] = 1

    if not filtered:
        return {
            "intent": "search",
            "session_id": session_id,
            "answer": "No planets matched that query. Try broadening the distance, year, or planet-type filters.",
            "results": [],
            "page": 1,
            "total_pages": 0,
            "total_matches": 0,
        }

    payload = _build_page_payload(session_id, session, page=1)
    requested_status = _requested_status(message)
    if "nearby" in message or "closest" in message or "nearest" in message:
        opener = "Closest matching planets"
    elif "latest" in message or "newest" in message or "recent" in message:
        opener = "Most recent matching discoveries"
    elif requested_status == "inhospitable":
        opener = "Least habitable planets on record"
    elif requested_status == "habitable":
        opener = "Most habitable planets on record"
    elif requested_status == "marginal":
        opener = "Marginally habitable planets"
    elif requested_status == "unknown":
        opener = "Planets with too little data to classify"
    else:
        opener = "Best matching planets"
    payload["answer"] = (
        f"{opener}: " + ", ".join(item["planet_name"] for item in payload["results"][:3]) + ". "
        f"Page {payload['page']} of {payload['total_pages']}."
    )
    return payload


def _build_info_answer(planet_name: str, session_id: str, session: dict):
    info = get_planet_info(planet_name)
    if not info:
        return {
            "intent": "info",
            "session_id": session_id,
            "answer": f"I could not find a planet record for {planet_name}.",
            "planet": None,
        }

    info = enrich_planet_info(info, include_external=True)
    explanation = _safe_explain(info["planet_name"])
    session["active_planet"] = info["planet_name"]
    session["last_planet_info"] = info
    session["last_analysis"] = explanation
    return {
        "intent": "info",
        "session_id": session_id,
        "answer": narrative_summary(info),
        "planet": info,
        "analysis": explanation,
    }


def _build_compare_answer(planet_a: str, planet_b: str, session_id: str):
    info_a = enrich_planet_info(get_planet_info(planet_a), include_external=False)
    info_b = enrich_planet_info(get_planet_info(planet_b), include_external=False)
    explanation_a = _safe_explain(planet_a)
    explanation_b = _safe_explain(planet_b)

    habitability_a = explanation_a.get("habitability_index") or 0.0
    habitability_b = explanation_b.get("habitability_index") or 0.0

    if habitability_a > habitability_b:
        summary = f"{planet_a} ranks higher on the current habitability estimate than {planet_b}."
    elif habitability_b > habitability_a:
        summary = f"{planet_b} ranks higher on the current habitability estimate than {planet_a}."
    else:
        summary = f"{planet_a} and {planet_b} are tied on the current habitability estimate."

    return {
        "intent": "compare",
        "session_id": session_id,
        "answer": summary,
        "comparison": {
            "planet_a": info_a or {"planet_name": planet_a},
            "planet_b": info_b or {"planet_name": planet_b},
            "analysis_a": explanation_a,
            "analysis_b": explanation_b,
            "habitability_delta": round(habitability_a - habitability_b, 3),
        },
    }


def _response_from_reference(message: str, session_id: str, session: dict):
    indexes = _extract_reference_indexes(message)
    results = session.get("results") or []
    if not indexes or not results:
        return {
            "intent": "reference",
            "session_id": session_id,
            "answer": "Reference a result number after running a search, for example: tell me about the second result.",
        }

    selected = []
    for index in indexes[:2]:
        if 1 <= index <= len(results):
            selected.append(results[index - 1]["planet_name"])

    if not selected:
        return {
            "intent": "reference",
            "session_id": session_id,
            "answer": "That result number is outside the current search results.",
        }

    if len(selected) >= 2 or "compare" in message:
        return _build_compare_answer(selected[0], selected[1], session_id)
    return _build_info_answer(selected[0], session_id, session)


def _apply_openai_if_requested(req: ChatRequest, payload: dict, session: dict):
    payload["openai_used"] = False
    if not req.use_openai:
        return payload
    if not openai_available():
        payload["openai_error"] = "OPENAI_API_KEY is not configured."
        return payload
    try:
        payload["answer_openai"] = render_openai_answer(req.message, payload, session)
        payload["openai_used"] = True
    except Exception as exc:
        payload["openai_error"] = str(exc)
    return payload


@router.post("/ask")
def ask_chat(req: ChatRequest):
    message = req.message.strip()
    session_id, session = _session_for(req.session_id)
    if req.active_planet:
        session["active_planet"] = req.active_planet

    if not message:
        payload = {
            "intent": "unknown",
            "session_id": session_id,
            "answer": "Ask about a planet, compare two planets, request filtered planet lists, or ask for the next page.",
        }
        return _apply_openai_if_requested(req, payload, session)

    records = get_planet_catalog_records()
    lowered = message.lower()
    planet_mentions = _collect_mentioned_planets(message, records)
    intent = _intent_for_message(lowered, planet_mentions)

    # "what does transit depth mean" is a definition question, not a planet search.
    glossary_terms = glossary.find_terms(lowered)
    definition_question = glossary.is_definition_question(lowered)
    # "which planets are ..." asks for a list even when it mentions a known term,
    # so a request that looks like a catalog query never becomes a definition lookup.
    wants_a_list = re.search(
        r"\b(planets?|worlds?|show|list|find|top|nearby|closest|nearest|within|discovered|orbiting)\b",
        lowered,
    )

    if intent == "search" and not planet_mentions and not wants_a_list:
        # A planet in context gets the richer planet-specific explanation for terms
        # that have one; everything else falls through to the glossary.
        if _is_explanation_question(lowered) and session.get("active_planet"):
            intent = "explain"
        elif definition_question or glossary_terms:
            intent = "glossary"

    limit = _extract_limit(lowered, req.limit)

    if intent == "glossary":
        payload = _build_glossary_answer(lowered, session_id, session, glossary_terms)
        return _apply_openai_if_requested(req, payload, session)

    if intent == "paginate":
        page = req.page
        if page is None:
            if lowered in {"next", "next page", "more", "more results"}:
                page = session.get("page", 1) + 1
            elif lowered in {"previous", "prev", "back"}:
                page = session.get("page", 1) - 1
            else:
                match = re.search(r"\bpage\s+(\d+)\b", lowered)
                page = int(match.group(1)) if match else session.get("page", 1)
        payload = _build_page_payload(session_id, session, page=page)
        return _apply_openai_if_requested(req, payload, session)

    if intent == "reference":
        payload = _response_from_reference(lowered, session_id, session)
        return _apply_openai_if_requested(req, payload, session)

    if intent == "explain":
        payload = _build_explanation_answer(lowered, session_id, session)
        return _apply_openai_if_requested(req, payload, session)

    if intent == "compare":
        if len(planet_mentions) >= 2:
            payload = _build_compare_answer(planet_mentions[0], planet_mentions[1], session_id)
        else:
            indexes = _extract_reference_indexes(lowered)
            results = session.get("results") or []
            selected = []
            for index in indexes[:2]:
                if 1 <= index <= len(results):
                    selected.append(results[index - 1]["planet_name"])
            if len(selected) >= 2:
                payload = _build_compare_answer(selected[0], selected[1], session_id)
            else:
                payload = {
                    "intent": "compare",
                    "session_id": session_id,
                    "answer": "Name two planets to compare, for example: compare Kepler-442 b vs TOI-700 e.",
                }
        return _apply_openai_if_requested(req, payload, session)

    if intent == "info":
        if planet_mentions:
            payload = _build_info_answer(planet_mentions[0], session_id, session)
        else:
            payload = {
                "intent": "info",
                "session_id": session_id,
                "answer": "Name a planet to inspect, for example: tell me about Kepler-442 b.",
            }
        return _apply_openai_if_requested(req, payload, session)

    payload = _build_search_answer(lowered, records, limit, session_id, session)
    return _apply_openai_if_requested(req, payload, session)
