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

API_EXPLAIN = "http://127.0.0.1:8000/models/explain"
API_PLANETS_ALL = "http://127.0.0.1:8000/planets/all"
API_PLANET_INFO = "http://127.0.0.1:8000/planets/info"

# =========================================================
# 🧩 UTILITY FUNCTIONS
# =========================================================
def habitability_bar(score: float):
    """Render colorized bar gauge for habitability index."""
    percent = int(score * 100)
    color = "red"
    if score >= 0.7:
        color = "green"
    elif score >= 0.3:
        color = "yellow"

    bar_length = 30
    filled = int(bar_length * score)
    bar = f"[{color}]" + "█" * filled + "[/]" + "·" * (bar_length - filled)
    return f"{bar} {percent}%"

def fetch_planet_catalog():
    """Fetch a list of planets dynamically from the backend."""
    console.print("[cyan]📡 Fetching planet catalog from backend...[/cyan]")
    try:
        resp = requests.get(API_PLANETS_ALL)
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
    """Fetch detailed planet info."""
    try:
        resp = requests.get(API_PLANET_INFO, params={"name": planet_name})
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"planet_name": planet_name, "status": "unknown"}
    except Exception:
        return {"planet_name": planet_name, "status": "unknown"}

def send_request(planet: str, features: list):
    """Send model explanation request."""
    payload = {"planet_name": planet, "features": features}
    console.print(f"[cyan]🚀 Requesting analysis for [bold]{planet}[/bold]...[/cyan]")
    resp = requests.post(API_EXPLAIN, json=payload)
    if resp.status_code != 200:
        console.print(f"[red]❌ API Error:[/red] {resp.status_code} — {resp.text}")
        sys.exit(1)
    return resp.json()

def display_report(data: dict, planet_info: dict):
    """Render rich-text explanation report."""
    console.rule("[bold cyan]🌌  TID-AD-ASTRA | Exoplanet Habitability Report[/bold cyan]")

    # Status badge
    status = planet_info.get("status", "unknown").lower()
    status_icon = {"habitable": "🟢", "marginal": "🟠", "inhospitable": "🔴"}.get(status, "⚪")
    status_label = status.capitalize() if status != "unknown" else "Unknown"

    console.print(f"{status_icon}  [bold]Status:[/bold] {status_label}")
    console.print(f"🪐 [bold]Planet:[/bold] {planet_info.get('planet_name', 'Unknown')}")
    console.print(f"📈 [bold]Confidence:[/bold] {data['confidence']*100:.1f}%")
    console.print(f"🌡  [bold]Habitability Index:[/bold] {habitability_bar(data['habitability_index'])}")
    console.print(f"📊 [bold]Predicted Class:[/bold] {data['predicted_label']}")
    console.print(f"🔬 [bold]Model Hash:[/bold] {data['model']}")
    console.print(f"📅 [bold]Timestamp:[/bold] {datetime.now().isoformat()}")

    # Physical data table
    console.rule("[magenta]Physical Characteristics[/magenta]")
    info_table = Table(show_header=False, box=None)
    info_table.add_row("Mass (Earth)", str(planet_info.get("mass_earth", "—")))
    info_table.add_row("Radius (Earth)", str(planet_info.get("radius_earth", "—")))
    info_table.add_row("Temperature (K)", str(planet_info.get("temperature_k", "—")))
    info_table.add_row("Distance (ly)", str(planet_info.get("system_distance_ly", "—")))
    info_table.add_row("Discovery Year", str(planet_info.get("discovery_year", "—")))
    info_table.add_row("Discovery Method", str(planet_info.get("discovery_method", "—")))
    console.print(info_table)

    # Feature influence
    console.rule("[magenta]Feature Influence[/magenta]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature")
    table.add_column("Importance", justify="right")
    for k, v in data.get("top_features", {}).items():
        table.add_row(k, f"{v:.5f}")
    console.print(table)

    console.rule()
    console.print(f"[green]💡 Summary:[/green] {data['summary']}")
    console.rule()

# =========================================================
# 🚀 MAIN LOOP
# =========================================================
def main():
    console.print("[bold cyan]🧠  Welcome to the TID-AD-ASTRA Explainability Console[/bold cyan]")
    console.print("Use this tool to query planetary habitability models.\n")

    # Fetch planets dynamically
    planets = fetch_planet_catalog()
    if not planets:
        console.print("[red]No planets loaded. Backend might be offline.[/red]")
        sys.exit(1)

    planet_choices = [p["planet_name"] for p in planets if "planet_name" in p]
    planet_choices.append("Custom input")

    planet = questionary.select("Select a planet to analyze:", choices=planet_choices).ask()

    if planet == "Custom input":
        planet = questionary.text("Enter planet name:").ask()

    planet_info = fetch_planet_info(planet)
    console.print(f"[blue]Analyzing {planet}...[/blue]")
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing planetary data...", total=20)
        for _ in range(20):
            time.sleep(0.05)
            progress.advance(task)

    # Predefined feature input (replace with real in future)
    default_features = [0.2,1.1,0.9,365,1,0.05,10,50,0,89,288,4.5,2015,12.3,1,1,0,5778,4.6]

    data = send_request(planet, default_features)
    display_report(data, planet_info)

    again = questionary.confirm("Run another analysis?").ask()
    if again:
        console.print()
        main()
    else:
        console.print("[cyan]👋 Mission control link closed.[/cyan]")


if __name__ == "__main__":
    main()

