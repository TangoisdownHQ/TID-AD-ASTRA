import json
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import requests

from app.system.planet_knowledge import get_planet_catalog_records


MAST_API_URL = "https://mast.stsci.edu/api/v0/invoke"
EXOMAST_API_URL = "https://exo.mast.stsci.edu/api/v0.1"

LIGHT_YEAR_KM = 9.4607e12
SECONDS_PER_YEAR = 31557600
TRAVEL_SPEEDS_KM_S = {
    "light_speed": 299792.458,
    "ten_percent_light_speed": 29979.2458,
    "parker_solar_probe": 192.0,
    "voyager_1": 17.0,
}


def classify_proximity(distance_ly):
    if distance_ly is None:
        return "unknown"
    if distance_ly < 50:
        return "nearby"
    if distance_ly < 200:
        return "regional neighborhood"
    if distance_ly < 1000:
        return "distant but comparatively reachable for future astronomy"
    return "deep-space distant"


def estimate_travel_times(distance_ly):
    if distance_ly is None:
        return {}
    distance_km = float(distance_ly) * LIGHT_YEAR_KM
    estimates = {}
    for label, speed in TRAVEL_SPEEDS_KM_S.items():
        years = distance_km / speed / SECONDS_PER_YEAR
        estimates[label] = round(years, 2)
    return estimates


def derive_habitability_signals(info: dict):
    signals = []
    radius = info.get("radius")
    temp = info.get("temperature")
    star_temp = info.get("star_temp")
    distance_ly = info.get("distance_ly")

    if radius is not None:
        if 0.8 <= radius <= 1.8:
            signals.append("planet radius is in an Earth-to-super-Earth range")
        elif radius > 4:
            signals.append("planet radius suggests a gas giant, reducing surface-habitability odds")

    if temp is not None:
        if 273 <= temp <= 373:
            signals.append("equilibrium temperature falls near a liquid-water-friendly band")
        elif temp < 220:
            signals.append("equilibrium temperature looks very cold")
        elif temp > 400:
            signals.append("equilibrium temperature looks very hot")

    if star_temp is not None:
        if 4800 <= star_temp <= 6500:
            signals.append("host star temperature is broadly Sun-like")
        elif star_temp < 3800:
            signals.append("host star is a cool star, useful for compact habitable-zone searches")

    if distance_ly is not None:
        if distance_ly < 50:
            signals.append("system is relatively close to Earth for follow-up observations")
        elif distance_ly > 1000:
            signals.append("system is very distant, making follow-up observations harder")

    return signals


def build_system_context(info: dict):
    host_star = info.get("host_star")
    if not host_star:
        return {"planet_count": 1, "neighbors": []}

    records = get_planet_catalog_records()
    neighbors = [
        {
            "planet_name": row.get("planet_name"),
            "distance_ly": row.get("distance_ly"),
            "habitability_score": row.get("habitability_score"),
            "planet_type": row.get("planet_type"),
        }
        for row in records
        if row.get("host_star") == host_star and row.get("planet_name") != info.get("planet_name")
    ]
    neighbors = sorted(neighbors, key=lambda item: item.get("habitability_score") or 0, reverse=True)
    return {
        "planet_count": len(neighbors) + 1,
        "neighbors": neighbors[:10],
    }


def _summarize_mission_candidates(payload: Any, mission_label: str):
    items = payload if isinstance(payload, list) else []
    candidates = []
    for item in items[:5]:
        candidates.append(
            {
                "tce_name": item.get("tce_name") or item.get("tceid") or item.get("tic_id"),
                "disposition": item.get("disposition") or item.get("label"),
                "period_days": item.get("orbital_period") or item.get("period"),
                "duration_hours": item.get("duration"),
                "depth_ppm": item.get("depth"),
            }
        )
    return {
        "mission": mission_label,
        "candidate_count": len(items),
        "candidates": candidates,
    }


def _mast_query(service: str, params: dict[str, Any], pagesize: int = 10):
    request_obj = {
        "service": service,
        "params": params,
        "format": "json",
        "pagesize": pagesize,
        "page": 1,
        "removenullcolumns": True,
    }
    response = requests.post(
        MAST_API_URL,
        data={"request": json.dumps(request_obj)},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=256)
def fetch_mast_identifiers(planet_name: str):
    url = f"{EXOMAST_API_URL}/exoplanets/identifiers/?name={quote(planet_name)}"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=256)
def fetch_mast_properties(canonical_name: str):
    url = f"{EXOMAST_API_URL}/exoplanets/{quote(canonical_name)}/properties"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=256)
def fetch_mast_tces(mission: str, star_id: str):
    url = f"{EXOMAST_API_URL}/dvdata/{mission}/{star_id}/tces/"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=256)
def fetch_gaia_dr3_by_position(ra: float, dec: float):
    payload = _mast_query(
        "Mast.Catalogs.GaiaDR3.Cone",
        {"ra": ra, "dec": dec, "radius": 0.002},
        pagesize=5,
    )
    data = payload.get("data") or []
    if not data:
        return {}
    top = data[0]
    return {
        "source_id": top.get("source_id"),
        "parallax_mas": top.get("parallax"),
        "g_mag": top.get("phot_g_mean_mag"),
        "pmra": top.get("pmra"),
        "pmdec": top.get("pmdec"),
        "ra": top.get("ra"),
        "dec": top.get("dec"),
        "reference": "MAST Gaia DR3",
    }


def _extract_property_excerpt(properties: dict[str, Any]):
    wanted = [
        "canonicalName",
        "starName",
        "discoveryMethod",
        "discoveryFacility",
        "planetMass",
        "planetRadius",
        "orbitalPeriod",
        "semiMajorAxis",
        "eccentricity",
        "inclination",
        "starDistance",
        "starTeff",
    ]
    excerpt = {}
    for key in wanted:
        if key in properties:
            excerpt[key] = properties[key]
    return excerpt


def build_external_enrichment(planet_name: str):
    enrichment = {
        "mast": {},
        "gaia": {},
        "missions": {},
        "status": "unavailable",
    }

    try:
        identifiers = fetch_mast_identifiers(planet_name)
        enrichment["mast"]["identifiers"] = identifiers

        canonical_name = identifiers.get("canonicalName")
        if canonical_name:
            try:
                properties = fetch_mast_properties(canonical_name)
                enrichment["mast"]["properties_excerpt"] = _extract_property_excerpt(properties)
            except Exception as exc:
                enrichment["mast"]["properties_error"] = str(exc)

        ra = identifiers.get("ra")
        dec = identifiers.get("dec")
        if ra is not None and dec is not None:
            try:
                enrichment["gaia"] = fetch_gaia_dr3_by_position(float(ra), float(dec))
            except Exception as exc:
                enrichment["gaia"]["error"] = str(exc)

        kepler_id = identifiers.get("keplerID")
        tess_id = identifiers.get("tessID")
        if kepler_id:
            try:
                enrichment["missions"]["kepler"] = _summarize_mission_candidates(
                    fetch_mast_tces("kepler", str(kepler_id)),
                    "Kepler",
                )
            except Exception as exc:
                enrichment["missions"]["kepler_error"] = str(exc)
        if tess_id:
            try:
                enrichment["missions"]["tess"] = _summarize_mission_candidates(
                    fetch_mast_tces("tess", str(tess_id)),
                    "TESS",
                )
            except Exception as exc:
                enrichment["missions"]["tess_error"] = str(exc)

        enrichment["status"] = "ok"
    except Exception as exc:
        enrichment["error"] = str(exc)

    return enrichment


def enrich_planet_info(info: dict, include_external: bool = False):
    if not info:
        return {}

    distance_ly = info.get("distance_ly")
    enriched = dict(info)
    enriched["proximity_category"] = classify_proximity(distance_ly)
    enriched["travel_estimates_years"] = estimate_travel_times(distance_ly)
    enriched["habitability_signals"] = derive_habitability_signals(info)
    enriched["system_context"] = build_system_context(info)

    if include_external and info.get("planet_name"):
        enriched["external_enrichment"] = build_external_enrichment(info["planet_name"])

    return enriched
