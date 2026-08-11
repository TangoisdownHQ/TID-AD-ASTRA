#!/usr/bin/env python3
"""
TID-AD-ASTRA CLI | Interactive Planetary Explainability Terminal Interface
Now dynamically fetches planets and shows habitability status.
"""
import sys
import time
import requests
import questionary
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

API_EXPLAIN = "http://127.0.0.1:8080/models/explain"
API_PLANETS_ALL = "http://127.0.0.1:8080/planets/all"
API_PLANET_INFO = "http://127.0.0.1:8080/planets/info"
API_CHAT = "http://127.0.0.1:8080/chat/ask"
API_DATASET_UPLOAD = "http://127.0.0.1:8080/datasets/upload"
API_DATASET_LIST = "http://127.0.0.1:8080/datasets/uploads"
API_DATASET_PREVIEW = "http://127.0.0.1:8080/datasets/preview"
API_SMALL_BODY_LOOKUP = "http://127.0.0.1:8080/small-bodies/lookup"
DEFAULT_FEATURES = [0.2,1.1,0.9,365,1,0.05,10,50,0,89,288,4.5,2015,12.3,1,1,0,5778,4.6]

# =========================================================
# 🧩 UTILITY FUNCTIONS
# =========================================================
def habitability_bar(score: float, coverage: float | None = None):
    if score is None:
        return "⚪ Not enough measured data to score"
    percent = int(score * 100)
    color = "red"
    if score >= 0.7:
        color = "green"
    elif score >= 0.3:
        color = "yellow"

    bar_length = 30
    filled = int(bar_length * score)
    bar = f"[{color}]" + "█" * filled + "[/]" + "·" * (bar_length - filled)
    suffix = ""
    if coverage is not None:
        suffix = f"  [dim]({int(coverage * 100)}% of inputs measured)[/dim]"
    return f"{bar} {percent}%{suffix}"


def display_value(value, fallback="—"):
    if value is None:
        return fallback
    if isinstance(value, str) and not value.strip():
        return fallback
    return str(value)


def fetch_planet_catalog():
    console.print("[cyan]📡 Fetching planet catalog from backend...[/cyan]")
    try:
        resp = requests.get(API_PLANETS_ALL, timeout=15)
        if resp.status_code != 200:
            console.print(f"[red]⚠️ Unexpected response from backend: {resp.status_code}[/red]")
            return []
        planets = resp.json()
        console.print(f"[green]✅ Loaded {len(planets)} planets from backend.[/green]")
        return planets
    except Exception as e:
        console.print(f"[red]❌ Failed to fetch planets: {e}[/red]")
        return []


def fetch_planet_info(planet_name: str):
    try:
        resp = requests.get(
            API_PLANET_INFO,
            params={"name": planet_name, "include_external": "true"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"planet_name": planet_name}
    except Exception:
        return {"planet_name": planet_name}


def send_request(planet: str, features: list):
    payload = {"planet_name": planet, "features": features}
    console.print(f"[cyan]🚀 Requesting analysis for [bold]{planet}[/bold]...[/cyan]")
    resp = requests.post(API_EXPLAIN, json=payload, timeout=30)
    if resp.status_code != 200:
        console.print(f"[red]❌ API Error:[/red] {resp.status_code} — {resp.text}")
        sys.exit(1)
    return resp.json()


def send_chat_request(
    message: str,
    limit: int = 5,
    session_id: str | None = None,
    use_openai: bool = False,
    page: int | None = None,
    active_planet: str | None = None,
):
    payload = {"message": message, "limit": limit, "session_id": session_id, "use_openai": use_openai}
    if page is not None:
        payload["page"] = page
    if active_planet:
        payload["active_planet"] = active_planet
    resp = requests.post(API_CHAT, json=payload, timeout=45)
    if resp.status_code != 200:
        console.print(f"[red]❌ Chat API Error:[/red] {resp.status_code} — {resp.text}")
        return {"intent": "error", "answer": "Chat request failed."}
    return resp.json()


def list_uploaded_datasets():
    resp = requests.get(API_DATASET_LIST, timeout=20)
    if resp.status_code != 200:
        console.print(f"[red]❌ Dataset API Error:[/red] {resp.status_code} — {resp.text}")
        return []
    return (resp.json() or {}).get("files", [])


def preview_uploaded_dataset(filename: str, limit: int = 15, sort_by: str | None = None, ascending: bool = False):
    params = {"filename": filename, "limit": limit, "ascending": ascending}
    if sort_by:
        params["sort_by"] = sort_by
    resp = requests.get(API_DATASET_PREVIEW, params=params, timeout=30)
    if resp.status_code != 200:
        console.print(f"[red]❌ Preview API Error:[/red] {resp.status_code} — {resp.text}")
        return None
    return resp.json()


def upload_dataset_file(path: str):
    try:
        with open(path, "rb") as handle:
            files = {"file": (path.split("/")[-1], handle, "text/csv")}
            resp = requests.post(API_DATASET_UPLOAD, files=files, timeout=60)
    except FileNotFoundError:
        console.print(f"[red]❌ File not found:[/red] {path}")
        return None
    except Exception as exc:
        console.print(f"[red]❌ Failed to open file:[/red] {exc}")
        return None

    if resp.status_code != 200:
        console.print(f"[red]❌ Upload failed:[/red] {resp.status_code} — {resp.text}")
        return None
    return resp.json()


def lookup_small_body(query: str):
    resp = requests.get(API_SMALL_BODY_LOOKUP, params={"query": query}, timeout=45)
    if resp.status_code != 200:
        console.print(f"[red]❌ Small-body lookup failed:[/red] {resp.status_code} — {resp.text}")
        return None
    return resp.json()


def display_report(data: dict, planet_info: dict):
    console.rule("[bold cyan]🌌  TID-AD-ASTRA | Exoplanet Habitability Report[/bold cyan]")

    status = (
        planet_info.get("status") or data.get("habitability_status") or "unknown"
    ).lower()
    status_icon = {"habitable": "🟢", "marginal": "🟠", "inhospitable": "🔴"}.get(status, "⚪")
    status_label = status.capitalize() if status != "unknown" else "Not classified"

    console.print(f"{status_icon}  [bold]Habitability status:[/bold] {status_label}")
    console.print(f"🪐 [bold]Planet:[/bold] {planet_info.get('planet_name') or 'Unknown'}")

    disposition = planet_info.get("disposition")
    if disposition:
        badge = {
            "confirmed": "[green]confirmed planet[/green]",
            "candidate": "[yellow]unconfirmed candidate[/yellow]",
            "controversial": "[yellow]disputed detection[/yellow]",
        }.get(disposition, disposition)
        console.print(f"🏷  [bold]Catalog status:[/bold] {badge}")

    hi = data.get("habitability_index")
    coverage = data.get("habitability_coverage", planet_info.get("habitability_coverage"))
    console.print(f"🌡  [bold]Habitability Index:[/bold] {habitability_bar(hi, coverage)}")
    if data.get("habitability_explanation"):
        console.print(f"    [dim]{data['habitability_explanation']}[/dim]")

    console.print(f"📅 [bold]Timestamp:[/bold] {datetime.now().isoformat()}")

    # ----- Habitability breakdown: why the score is what it is -----
    factors = data.get("habitability_factors") or planet_info.get("habitability_factors") or []
    if factors:
        console.rule("[magenta]Why This Habitability Score[/magenta]")
        factor_table = Table(show_header=True, header_style="bold magenta")
        factor_table.add_column("Factor")
        factor_table.add_column("Measured", justify="right")
        factor_table.add_column("Earth reference")
        factor_table.add_column("Weight", justify="right")
        factor_table.add_column("Factor score", justify="right")
        for factor in factors:
            factor_table.add_row(
                str(factor.get("label")),
                f"{factor.get('value')} {factor.get('unit', '')}".strip(),
                str(factor.get("reference")),
                f"{factor.get('weight'):.0%}",
                f"{factor.get('score'):.2f}",
            )
        console.print(factor_table)
        missing = data.get("habitability_missing") or planet_info.get("habitability_missing") or []
        if missing:
            console.print(f"[yellow]Not measured for this planet:[/yellow] {', '.join(missing)}")

    # ----- Detection model: reported with what it can and cannot tell you -----
    console.rule("[magenta]Detection Model[/magenta]")
    console.print(
        f"📊 [bold]Predicted class:[/bold] {display_value(data.get('predicted_label'), 'Unknown')}"
    )
    confidence = data.get("confidence")
    if confidence is None:
        console.print("📈 [bold]Confidence:[/bold] ⚪ Unknown (insufficient data)")
    else:
        console.print(f"📈 [bold]Confidence:[/bold] {confidence * 100:.1f}%")

    inputs = data.get("model_inputs") or {}
    if inputs:
        quality = inputs.get("quality", "unknown")
        quality_color = {
            "planet-specific": "green",
            "partly planet-specific": "yellow",
            "not planet-specific": "red",
        }.get(quality, "white")
        console.print(f"🧪 [bold]Input basis:[/bold] [{quality_color}]{quality}[/{quality_color}]")
        console.print(f"    [dim]{inputs.get('basis', '')}[/dim]")

    nasa_score = planet_info.get("disposition_score")
    if nasa_score is not None:
        console.print(
            f"🛰  [bold]NASA vetting score:[/bold] {float(nasa_score):.2f} "
            "[dim](NASA's own confidence this KOI is a planet)[/dim]"
        )

    if data.get("predicted_class_explanation"):
        console.print(f"    [dim]{data['predicted_class_explanation']}[/dim]")
    if data.get("prediction_caveat"):
        console.print(f"[yellow]⚠  {data['prediction_caveat']}[/yellow]")
    console.print(f"🔬 [bold]Model hash:[/bold] {display_value(data.get('model'), 'Unknown')}")

    console.rule("[magenta]Physical Characteristics[/magenta]")
    field_sources = planet_info.get("field_sources") or {}
    info_table = Table(show_header=True, header_style="bold magenta", box=None)
    info_table.add_column("Property")
    info_table.add_column("Value")
    info_table.add_column("Measured by", style="dim")

    def add_row(label, key):
        info_table.add_row(
            label,
            display_value(planet_info.get(key)),
            display_value(field_sources.get(key), "—"),
        )

    add_row("Mass (Earth)", "mass_earth")
    add_row("Radius (Earth)", "radius")
    add_row("Temperature (K)", "temperature")
    add_row("Distance (pc)", "distance_pc")
    info_table.add_row("Distance (ly)", display_value(planet_info.get("distance_ly")), "—")
    add_row("Discovery Year", "discovery_year")
    add_row("Discovery Method", "discovery_method")
    add_row("Host Star", "host_star")
    info_table.add_row(
        "Proximity", display_value(planet_info.get("proximity_category")), "—"
    )
    console.print(info_table)

    sources = planet_info.get("sources") or []
    if sources:
        console.print(f"[dim]Catalogs listing this planet: {', '.join(sources)}[/dim]")

    habitability_signals = planet_info.get("habitability_signals") or []
    if habitability_signals:
        console.rule("[magenta]Habitability Signals[/magenta]")
        for signal in habitability_signals:
            console.print(f"• {signal}")

    system_context = planet_info.get("system_context") or {}
    neighbors = system_context.get("neighbors") or []
    if system_context:
        console.rule("[magenta]System Context[/magenta]")
        console.print(
            f"Known planets around host star: {system_context.get('planet_count', '—')}"
        )
        if neighbors:
            neighbor_table = Table(show_header=True, header_style="bold magenta")
            neighbor_table.add_column("Neighbor")
            neighbor_table.add_column("Type")
            neighbor_table.add_column("Habitability", justify="right")
            neighbor_table.add_column("Distance (ly)", justify="right")
            for neighbor in neighbors[:5]:
                score = neighbor.get("habitability_score")
                neighbor_table.add_row(
                    str(neighbor.get("planet_name", "—")),
                    str(neighbor.get("planet_type", "—")),
                    f"{score:.3f}" if score is not None else "—",
                    str(neighbor.get("distance_ly", "—")),
                )
            console.print(neighbor_table)

    travel_estimates = planet_info.get("travel_estimates_years") or {}
    if travel_estimates:
        console.rule("[magenta]Travel Time From Earth[/magenta]")
        travel_table = Table(show_header=True, header_style="bold magenta")
        travel_table.add_column("Reference Speed")
        travel_table.add_column("Years", justify="right")
        labels = {
            "light_speed": "At light speed",
            "ten_percent_light_speed": "At 10% light speed",
            "parker_solar_probe": "At Parker Solar Probe speed",
            "voyager_1": "At Voyager 1 speed",
        }
        for key, label in labels.items():
            if key in travel_estimates:
                travel_table.add_row(label, str(travel_estimates[key]))
        console.print(travel_table)

    external = planet_info.get("external_enrichment") or {}
    if external:
        console.rule("[magenta]Gaia / Mission Enrichment[/magenta]")
        console.print(f"Enrichment status: {external.get('status', 'unknown')}")
        gaia = external.get("gaia") or {}
        if gaia:
            gaia_table = Table(show_header=False, box=None)
            gaia_table.add_row("Gaia Source ID", str(gaia.get("source_id", "—")))
            gaia_table.add_row("Parallax (mas)", str(gaia.get("parallax_mas", "—")))
            gaia_table.add_row("G Magnitude", str(gaia.get("g_mag", "—")))
            gaia_table.add_row("PMRA", str(gaia.get("pmra", "—")))
            gaia_table.add_row("PMDEC", str(gaia.get("pmdec", "—")))
            console.print(gaia_table)

        mast = external.get("mast") or {}
        properties = mast.get("properties_excerpt") or {}
        if properties:
            mast_table = Table(show_header=False, box=None)
            mast_table.add_row("Canonical Name", str(properties.get("canonicalName", "—")))
            mast_table.add_row("Discovery Facility", str(properties.get("discoveryFacility", "—")))
            mast_table.add_row("Orbital Period", str(properties.get("orbitalPeriod", "—")))
            mast_table.add_row("Semi-major Axis", str(properties.get("semiMajorAxis", "—")))
            mast_table.add_row("Star Distance", str(properties.get("starDistance", "—")))
            mast_table.add_row("Star Teff", str(properties.get("starTeff", "—")))
            console.print(mast_table)

        missions = external.get("missions") or {}
        for mission_key in ["kepler", "tess"]:
            mission = missions.get(mission_key) or {}
            if not mission:
                continue
            console.print(
                f"{mission.get('mission', mission_key.title())}: "
                f"{mission.get('candidate_count', 0)} tracked candidate events"
            )
            for candidate in mission.get("candidates") or []:
                console.print(
                    "  - "
                    f"{candidate.get('tce_name', 'unknown')} | "
                    f"{candidate.get('disposition', 'unknown')} | "
                    f"period {candidate.get('period_days', '—')}"
                )

    console.rule("[magenta]Feature Influence[/magenta]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature")
    table.add_column("Input")
    table.add_column("Meaning")
    table.add_column("Importance", justify="right")

    top_features = data.get("top_features") or {}
    feature_details = {
        item.get("label"): item.get("description")
        for item in (data.get("top_feature_details") or [])
        if item.get("label")
    }
    origins = {
        item.get("feature"): item.get("origin")
        for item in (data.get("model_input_details") or [])
    }
    origin_labels = {
        "planet": "[green]measured[/green]",
        "median": "[yellow]imputed[/yellow]",
        "zero": "[red]missing[/red]",
        "caller": "supplied",
    }

    if not top_features:
        table.add_row("N/A", "—", "No feature data available", "—")
    else:
        for k, v in top_features.items():
            origin = origin_labels.get(origins.get(k), "—")
            try:
                importance = f"{float(v):.5f}"
            except Exception:
                importance = "N/A"
            table.add_row(k, origin, feature_details.get(k, "—"), importance)

    console.print(table)
    if any(origins.get(k) in {"median", "zero"} for k in top_features):
        console.print(
            "[yellow]Note:[/yellow] [dim]features marked 'imputed' were not measured for this "
            "planet — the model used a training-set median, so their influence reflects the "
            "dataset, not this planet.[/dim]"
        )

    console.rule()
    console.print(f"[green]💡 Summary:[/green] {data.get('summary', 'No summary available.')}")
    if data.get("reason"):
        console.print(f"[yellow]🧩 Diagnostics:[/yellow] {data['reason']}")
    console.rule()


def render_search_results(results: list[dict]):
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("Planet")
    table.add_column("Catalog status")
    table.add_column("Habitability")
    table.add_column("Score", justify="right")
    table.add_column("Distance (ly)", justify="right")
    table.add_column("Method")

    status_colors = {
        "habitable": "green",
        "marginal": "yellow",
        "inhospitable": "red",
        "unknown": "dim",
    }
    disposition_colors = {
        "confirmed": "green",
        "candidate": "yellow",
        "controversial": "yellow",
    }

    for idx, row in enumerate(results, start=1):
        score = row.get("habitability_score")
        status = str(row.get("status", "unknown"))
        disposition = row.get("disposition") or "—"
        coverage = row.get("habitability_coverage")

        status_cell = f"[{status_colors.get(status, 'white')}]{status}[/]"
        if status == "unknown" and coverage is not None:
            status_cell += f" [dim]({int(coverage * 100)}% data)[/dim]"

        table.add_row(
            str(idx),
            str(row.get("planet_name", "Unknown")),
            f"[{disposition_colors.get(disposition, 'white')}]{disposition}[/]",
            status_cell,
            f"{score:.3f}" if score is not None else "—",
            str(row.get("distance_ly", "—")),
            str(row.get("discovery_method", "—")),
        )

    console.print(table)


def render_info_result(payload: dict):
    planet = payload.get("planet") or {}
    analysis = payload.get("analysis") or {}
    console.print(f"[green]💬 {payload.get('answer', 'No answer available.')}[/green]")
    if planet:
        display_report(analysis, planet)


def render_compare_result(payload: dict):
    comparison = payload.get("comparison") or {}
    planet_a = comparison.get("planet_a") or {}
    planet_b = comparison.get("planet_b") or {}
    analysis_a = comparison.get("analysis_a") or {}
    analysis_b = comparison.get("analysis_b") or {}

    console.rule("[bold cyan]Planet Comparison[/bold cyan]")
    console.print(f"[green]💬 {payload.get('answer', 'No answer available.')}[/green]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column(planet_a.get("planet_name", "Planet A"))
    table.add_column(planet_b.get("planet_name", "Planet B"))
    table.add_row("Habitability", str(analysis_a.get("habitability_index", "—")), str(analysis_b.get("habitability_index", "—")))
    table.add_row("Radius", str(planet_a.get("radius", "—")), str(planet_b.get("radius", "—")))
    table.add_row("Temperature (K)", str(planet_a.get("temperature", "—")), str(planet_b.get("temperature", "—")))
    table.add_row("Distance (ly)", str(planet_a.get("distance_ly", "—")), str(planet_b.get("distance_ly", "—")))
    table.add_row("Method", str(planet_a.get("discovery_method", "—")), str(planet_b.get("discovery_method", "—")))
    console.print(table)
    console.rule()


def maybe_analyze_from_results(results: list[dict]):
    if not results:
        return None
    if not questionary.confirm("Analyze one of these planets now?").ask():
        return None

    choices = [row["planet_name"] for row in results[:10] if row.get("planet_name")]
    if not choices:
        return None

    planet = questionary.select("Select a planet to analyze:", choices=choices).ask()
    if planet:
        analyze_planet(planet)
        return planet
    return None


def handle_chat_response(payload: dict):
    intent = payload.get("intent")
    answer = payload.get("answer_openai") or payload.get("answer") or "No answer available."
    openai_note = payload.get("openai_error")
    if intent == "search":
        console.print(f"[green]💬 {answer}[/green]")
        results = payload.get("results") or []
        if results:
            render_search_results(results)
            if payload.get("total_pages"):
                console.print(
                    f"[cyan]Page {payload.get('page', 1)} of {payload.get('total_pages')} "
                    f"({payload.get('total_matches', len(results))} matches).[/cyan]"
                )
            chosen_planet = maybe_analyze_from_results(results)
            if chosen_planet:
                return chosen_planet
        if openai_note:
            console.print(f"[yellow]OpenAI fallback note:[/yellow] {openai_note}")
        return None

    if intent == "glossary":
        console.rule("[bold cyan]Glossary[/bold cyan]")
        for term in payload.get("terms") or []:
            unit = f" [dim]({term['unit']})[/dim]" if term.get("unit") else ""
            console.print(f"[bold]{term['label']}[/bold]{unit}")
            console.print(f"  {term['definition']}")
            if term.get("planet_value"):
                console.print(f"  [cyan]{term['planet_value']}[/cyan]")
            console.print()
        if not payload.get("terms"):
            console.print(f"[yellow]{answer}[/yellow]")
        elif payload.get("active_planet"):
            console.print(f"[dim]Values shown are for {payload['active_planet']}.[/dim]")
        console.rule()
        if openai_note:
            console.print(f"[yellow]OpenAI fallback note:[/yellow] {openai_note}")
        return None

    if intent in {"info", "explain"}:
        if payload.get("answer_openai"):
            payload = dict(payload)
            payload["answer"] = payload["answer_openai"]
        render_info_result(payload)
        if openai_note:
            console.print(f"[yellow]OpenAI fallback note:[/yellow] {openai_note}")
        planet = payload.get("planet") or {}
        return planet.get("planet_name")

    if intent == "compare":
        if payload.get("answer_openai"):
            payload = dict(payload)
            payload["answer"] = payload["answer_openai"]
        render_compare_result(payload)
        if openai_note:
            console.print(f"[yellow]OpenAI fallback note:[/yellow] {openai_note}")
        return None

    console.print(f"[yellow]{answer}[/yellow]")

    if openai_note:
        console.print(f"[yellow]OpenAI fallback note:[/yellow] {openai_note}")
    return None


def analyze_planet(planet: str):
    planet_info = fetch_planet_info(planet)
    console.print(f"[blue]Analyzing {planet}...[/blue]")

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing planetary data...", total=20)
        for _ in range(20):
            time.sleep(0.05)
            progress.advance(task)

    data = send_request(planet, DEFAULT_FEATURES)
    display_report(data, planet_info)


def run_analysis_mode(planets: list[dict]):
    planet_choices = [p["planet_name"] for p in planets if p.get("planet_name")]
    planet_choices.append("Custom input")

    planet = questionary.select("Select a planet to analyze:", choices=planet_choices).ask()
    if planet == "Custom input":
        planet = questionary.text("Enter planet name:").ask()

    if planet:
        analyze_planet(planet)


def render_dataset_preview(payload: dict):
    if not payload:
        return
    columns = payload.get("columns") or []
    preview_rows = payload.get("preview") or []

    console.rule(f"[bold cyan]Dataset Preview | {payload.get('filename', 'Unknown')}[/bold cyan]")
    console.print(f"Rows: {payload.get('rows', '—')}")
    if not preview_rows:
        console.print("[yellow]No preview rows available.[/yellow]")
        return

    shown_columns = columns[:6]
    table = Table(show_header=True, header_style="bold magenta")
    for column in shown_columns:
        table.add_column(str(column))

    for row in preview_rows:
        table.add_row(*[str(row.get(column, "—")) for column in shown_columns])
    console.print(table)


def render_small_body(payload: dict):
    summary = (payload or {}).get("summary") or {}
    if not summary:
        console.print("[yellow]No small-body data available.[/yellow]")
        return

    console.rule("[bold cyan]Small-Body Lookup[/bold cyan]")
    table = Table(show_header=False, box=None)
    table.add_row("Object", str(summary.get("object_name", "—")))
    table.add_row("Designation", str(summary.get("designation", "—")))
    table.add_row("Orbit Class", str(summary.get("orbit_class", "—")))
    table.add_row("NEO", str(summary.get("is_neo", "—")))
    table.add_row("PHA", str(summary.get("is_pha", "—")))
    table.add_row("Period (days)", str(summary.get("period_days", "—")))
    table.add_row("Perihelion (au)", str(summary.get("perihelion_au", "—")))
    table.add_row("Aphelion (au)", str(summary.get("aphelion_au", "—")))
    table.add_row("Inclination (deg)", str(summary.get("inclination_deg", "—")))
    table.add_row("Absolute Magnitude", str(summary.get("absolute_magnitude", "—")))
    table.add_row("Rotation Period", str(summary.get("rotation_period_hours", "—")))
    table.add_row("Source", str(summary.get("source", "—")))
    console.print(table)
    console.rule()


def run_upload_mode():
    path = questionary.text("Enter the path to a CSV file to upload:").ask()
    if not path:
        return
    payload = upload_dataset_file(path)
    if not payload:
        return
    console.print(f"[green]✅ Uploaded:[/green] {payload.get('path')}")
    render_dataset_preview(payload.get("dataset") or {})


def run_uploaded_dataset_browser():
    files = list_uploaded_datasets()
    if not files:
        console.print("[yellow]No uploaded datasets found.[/yellow]")
        return

    filename = questionary.select(
        "Choose an uploaded dataset:",
        choices=[item["filename"] for item in files],
    ).ask()
    if not filename:
        return

    initial = preview_uploaded_dataset(filename)
    if not initial:
        return
    render_dataset_preview(initial)

    columns = initial.get("columns") or []
    if columns and questionary.confirm("Sort this dataset preview by a column?").ask():
        sort_by = questionary.select("Choose a sort column:", choices=columns[:50]).ask()
        ascending = questionary.confirm("Sort ascending?").ask()
        sorted_payload = preview_uploaded_dataset(filename, sort_by=sort_by, ascending=ascending)
        if sorted_payload:
            render_dataset_preview(sorted_payload)


def run_small_body_mode():
    query = questionary.text("Enter asteroid/comet/small-body name or designation:").ask()
    if not query:
        return
    payload = lookup_small_body(query)
    if payload:
        render_small_body(payload)


def run_chat_mode():
    console.rule("[bold cyan]Mission Chat[/bold cyan]")
    console.print("Ask things like:")
    console.print(" - show the most habitable nearby planets")
    console.print(" - tell me about Kepler-442 b")
    console.print(" - compare TOI-700 e vs Kepler-1649 b")
    console.print(" - show rocky planets within 100 ly discovered after 2020")
    console.print(" - which planets are uninhabitable")
    console.print(" - what is transit depth")
    console.print(" - what does koi_duration mean")
    console.print(" - next")
    console.print(" - tell me about the second result\n")

    use_openai = questionary.confirm(
        "Use OpenAI-enhanced answers when OPENAI_API_KEY is configured?"
    ).ask()
    session_id = None
    active_planet = None

    while True:
        prompt = questionary.text("Mission chat").ask()
        if prompt is None:
            return

        cleaned = prompt.strip()
        if cleaned.lower() in {"exit", "quit", "back"}:
            return

        payload = send_chat_request(
            cleaned,
            session_id=session_id,
            use_openai=bool(use_openai),
            active_planet=active_planet,
        )
        session_id = payload.get("session_id", session_id)
        active_planet = handle_chat_response(payload) or active_planet
        console.print()


# =========================================================
# 🚀 MAIN LOOP
# =========================================================
def main():
    console.print("[bold cyan]🧠  Welcome to the TID-AD-ASTRA Explainability Console[/bold cyan]")
    console.print("Use this tool to query planetary habitability models or chat with the catalog.\n")

    planets = fetch_planet_catalog()
    if not planets:
        console.print("[red]No planets loaded. Backend might be offline.[/red]")
        sys.exit(1)

    while True:
        mode = questionary.select(
            "Choose a mission mode:",
            choices=[
                "Analyze planet",
                "Chat with catalog",
                "Lookup small body",
                "Upload custom dataset",
                "Browse uploaded datasets",
                "Exit",
            ],
        ).ask()

        if mode == "Analyze planet":
            run_analysis_mode(planets)
        elif mode == "Chat with catalog":
            run_chat_mode()
        elif mode == "Lookup small body":
            run_small_body_mode()
        elif mode == "Upload custom dataset":
            run_upload_mode()
        elif mode == "Browse uploaded datasets":
            run_uploaded_dataset_browser()
        else:
            console.print("[cyan]👋 Mission control link closed.[/cyan]")
            return

        console.print()


if __name__ == "__main__":
    main()
