from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Query


router = APIRouter(tags=["Small Bodies"])

JPL_SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb.api"


def _extract_primary_fields(payload: dict[str, Any]):
    obj = payload.get("object") or {}
    orbit = payload.get("orbit") or {}
    phys = payload.get("phys_par") or []

    period = None
    aphelion = None
    perihelion = None
    inclination = None

    for element in orbit.get("elements", []) or []:
        name = element.get("name")
        value = element.get("value")
        if name == "per":
            period = value
        elif name == "ad":
            aphelion = value
        elif name == "q":
            perihelion = value
        elif name == "i":
            inclination = value

    absolute_magnitude = None
    rotation_period = None
    for parameter in phys:
        name = parameter.get("name")
        if name == "H":
            absolute_magnitude = parameter.get("value")
        elif name in {"rot_per", "rot_per_err"} and rotation_period is None:
            rotation_period = parameter.get("value")

    return {
        "object_name": obj.get("fullname") or obj.get("shortname") or obj.get("des"),
        "designation": obj.get("des"),
        "spk_id": obj.get("spkid"),
        "orbit_class": (obj.get("orbit_class") or {}).get("name"),
        "is_neo": obj.get("neo"),
        "is_pha": obj.get("pha"),
        "period_days": period,
        "perihelion_au": perihelion,
        "aphelion_au": aphelion,
        "inclination_deg": inclination,
        "absolute_magnitude": absolute_magnitude,
        "rotation_period_hours": rotation_period,
        "source": "JPL Small-Body Database API",
        "source_family": "solar_system_object",
    }


@router.get("/lookup")
def lookup_small_body(
    query: str = Query(..., description="Small-body designation, number, or name"),
    include_physical: bool = Query(True, description="Include physical parameters when available"),
):
    params = {"sstr": query}
    if include_physical:
        params["phys-par"] = 1
    try:
        response = requests.get(JPL_SBDB_API, params=params, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to query JPL SBDB: {exc}")

    payload = response.json()
    if "message" in payload and not payload.get("object"):
        raise HTTPException(status_code=404, detail=payload["message"])

    return {
        "query": query,
        "summary": _extract_primary_fields(payload),
        "raw": payload,
    }
