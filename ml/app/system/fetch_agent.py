"""
Fetch agent.

This module used to hold a second, independent set of downloaders that wrote the
same files as `app.system.update_datasets` with a different column schema — so
whichever ran last decided what the catalog looked like. It now delegates to the
single refresh implementation instead.
"""

from app.system.update_datasets import (
    DATA_REFRESH_INTERVAL_HOURS,
    NASA_URL,
    OEC_URL,
    main as refresh_datasets,
    refresh_if_stale,
)

__all__ = [
    "NASA_URL",
    "OEC_URL",
    "refresh_datasets",
    "run_all_fetchers",
]


def run_all_fetchers(force: bool = False):
    """
    Refresh local catalog files.

    By default this is a no-op when the local cache is still inside the refresh
    window, so it is safe to call on startup alongside the scheduler.
    """
    print("🧠 Running dataset fetch cycle...")

    if force:
        refresh_datasets()
        refreshed = True
    else:
        refreshed = refresh_if_stale(max_age_hours=DATA_REFRESH_INTERVAL_HOURS)

    print("🚀 Fetch cycle complete.")
    return refreshed
