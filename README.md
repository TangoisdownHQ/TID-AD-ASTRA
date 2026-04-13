# 🌌 **TID-AD-ASTRA**
### _Decoding the Universe, One Planet at a Time_

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![NASA Space Apps](https://img.shields.io/badge/NASA_Space_Apps-2025-red?logo=nasa&logoColor=white)
![TangoisdownHQ](https://img.shields.io/badge/TangoisdownHQ-Cyber_Intelligence-002b36?logo=linux&logoColor=white)
![TID-AD-ASTRA](https://img.shields.io/badge/TID--AD--ASTRA-To_the_Stars-003366?logo=nasa&logoColor=white)

CLI-first exoplanet explorer and explainability backend built for the 2025 NASA Space Apps Challenge.

## What It Does

TID-AD-ASTRA lets users browse planet records, inspect habitability-oriented metadata, and request model-backed explanations through:

- a FastAPI backend in `ml/app`
- a terminal UI in `app/cli/cli_explain.py`

The current build merges local and refreshable data from:

- NASA Exoplanet Archive
- Open Exoplanet Catalogue
- AstroML exoplanet dataset
- NASA KOI fallback data

In this project:

- `exoplanets` are planets around other stars
- `small bodies` are Solar System objects such as asteroids and comets
- `space objects` is the broad umbrella term for planets, moons, asteroids, comets, and similar bodies

## Repo Layout

- `ml/app/main.py`: FastAPI app
- `ml/app/routes/planets.py`: planet catalog and metadata endpoints
- `ml/app/routes/chat.py`: natural-language catalog chat
- `ml/app/routes/datasets.py`: custom dataset upload and preview endpoints
- `ml/app/routes/small_bodies.py`: JPL small-body lookup for asteroids and comets
- `ml/app/system/planet_knowledge.py`: source merging and habitability helpers
- `ml/app/system/update_datasets.py`: dataset refresh utility
- `app/cli/cli_explain.py`: interactive CLI interface
- `Makefile`: local run commands

## Quick Start

1. Create the virtualenv in `ml/.venv`.
2. Install dependencies:

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. From the repo root, start the backend:

```bash
make run-api
```

Or launch both backend and CLI together:

```bash
make run
```

On first run, if no local model artifact exists yet, the backend trains a starter model automatically before serving explainability features.

4. In a second terminal, launch the CLI:

```bash
make cli-explain
```

5. In the CLI, choose either:

- `Analyze planet` for the original select-and-report flow
- `Chat with catalog` for natural-language search, compare, and planet lookup
- `Lookup small body` for separate Solar System object queries
- `Upload custom dataset` to send a CSV to the backend
- `Browse uploaded datasets` to preview and sort uploaded CSV files

6. Smoke-test the catalog:

```bash
curl http://127.0.0.1:8080/planets/all?limit=5
curl "http://127.0.0.1:8080/planets/info?name=Kepler-442b"
curl "http://127.0.0.1:8080/planets/info?name=Kepler-442b&include_external=true"
curl "http://127.0.0.1:8080/planets/search?query=kepler"
curl "http://127.0.0.1:8080/small-bodies/lookup?query=Eros"
curl -X POST http://127.0.0.1:8080/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"compare TOI-700 e vs Kepler-1649 b","limit":5}'
curl -X POST http://127.0.0.1:8080/datasets/upload \
  -F "file=@/absolute/path/to/your_planets.csv"
```

## Data Refresh

Refresh the local CSV catalog cache with:

```bash
cd ml
.venv/bin/python -m app.system.update_datasets
```

Runtime behavior:

- on app startup, the backend checks whether the catalog refresh is older than 6 hours
- if the cache is stale or missing, it refreshes before serving users
- after startup, the scheduler rechecks every 6 hours and refreshes again when needed

You can change the threshold with:

```bash
export DATA_REFRESH_INTERVAL_HOURS=6
```

## First-Run Model Training

Fresh clones do not ship with committed model artifacts.

Runtime behavior:

- when the backend starts, it checks for a local trained model
- if none exists, it trains a starter model automatically
- after that, explainability and comparison features are available without extra setup

If you want to train manually instead:

```bash
cd ml
.venv/bin/python -m app.models.classifier --train ml/app/data/koi_fallback.csv
```

## When To Retrain

You do not need to retrain every time you start the app.

Retrain when:

- you pulled new code that changes model logic or feature handling
- you refreshed datasets and want the model to reflect the latest catalog
- you uploaded your own CSV files and want those files included in training
- explainability responses are falling back to `metadata-only` and you want model-backed feature importance again
- you see model-version drift warnings and want a fresh artifact built in your current environment

You usually do not need to retrain when:

- you only want to browse planets, chat with the catalog, or inspect metadata
- the current model is loading correctly and explanations are already working
- you only restarted the API with no code or dataset changes

Why retraining helps:

- it rebuilds the local model artifact in your own environment
- it updates the registry used by `/models/explain`
- it reduces stale-model issues after dependency or dataset changes
- it improves the chance that feature-importance output is available instead of a fallback response

Recommended retrain flow:

```bash
cd ml
source .venv/bin/activate
python -m app.models.classifier --train app/data/koi_fallback.csv
cd ..
make run
```

If you want a fresh backend after retraining, restart the app so the API loads the newest artifact.

## Custom Dataset Uploads

Users can upload their own CSV files and sort through them from the CLI.

CLI flow:

- choose `Upload custom dataset`
- provide the path to a local `.csv`
- preview the uploaded dataset
- choose `Browse uploaded datasets` to sort preview rows by a selected column

API endpoints:

- `POST /datasets/upload`
- `GET /datasets/uploads`
- `GET /datasets/preview?filename=...&sort_by=...&ascending=false`

Uploaded CSVs are stored in `ml/app/data/uploads/` and are also picked up by the training pipeline.

## 🌐 External Data Sources

| Source | Endpoint | Rows | Status |
|--------|-----------|------|--------|
| **NASA Exoplanet Archive** | `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+5000+*+from+pscomppars&format=csv` | 5000 | ✅ Updated |
| **Open Exoplanet Catalogue** | `https://raw.githubusercontent.com/OpenExoplanetCatalogue/oec_tables/master/comma_separated/open_exoplanet_catalogue.csv` | — | ⚠️ Failed |
| **AstroML Exoplanet Dataset** | `https://raw.githubusercontent.com/astroML/astroML-data/main/datasets/exoplanets.csv` | — | ⚠️ Failed |

## What Changed For Multi-Source Support

- The backend now merges NASA, OEC, KOI, and AstroML files through one metadata layer.
- `/planets/all` returns de-duplicated records across sources instead of reading only one dataset.
- `/planets/info` and `/planets/search` now understand both NASA-style and OEC-style column names.
- Dataset refresh now includes AstroML alongside NASA and OEC.
- `/chat/ask` adds deterministic natural-language search, info, compare, and ranking over the catalog.
- The CLI now includes a chat mode that can answer catalog questions and then jump into full planet analysis.
- Chat sessions now support follow-up context, result references like `tell me about the second result`, and pagination with `next`, `prev`, or `page 2`.
- Optional OpenAI-enhanced answers can be enabled from the CLI when `OPENAI_API_KEY` is configured.
- `/small-bodies/lookup` adds a separate JPL-backed mode for asteroids and comets.
- Exoplanets and Solar System objects are now treated as separate source families.
- `/planets/info?include_external=true` now adds local system-context enrichment plus optional live Gaia and MAST mission details.
- Rich planet reports now include host-system neighbors, proximity category, travel-time estimates from Earth, and habitability-oriented signals.

## Optional OpenAI Answer Layer

If you want the chat responses rewritten into a more conversational answer style on top of the deterministic catalog results, set:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-5.4-mini
```

The app still performs search, filtering, compare logic, and pagination locally first. OpenAI is only used to rewrite the structured result into a cleaner answer when enabled.

## Gaia And MAST Enrichment

Planet detail lookups now combine local catalog data with optional live enrichment when network access is available.

If the app is offline or the external APIs are unavailable, the planet detail view still works and falls back to the local merged catalog.

What users now see on detailed planet views:

- host-star and system-neighbor context
- distance-based proximity labels
- travel-time estimates from Earth at several reference speeds
- habitability-oriented signals based on size, temperature, star type, and distance
- Gaia DR3 crossmatch details when a match is found
- MAST exoplanet metadata and Kepler/TESS candidate-event summaries when available

Design notes:

- exoplanets and Solar System bodies stay separated as different source families
- Gaia and MAST are enrichment layers, not replacements for the merged exoplanet catalog
- uploaded user CSVs stay isolated from curated source files so provenance stays clear

Primary enrichment sources:

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- ESA Gaia Archive and Gaia user services: https://www.cosmos.esa.int/web/gaia-users/register
- MAST mission archives and APIs: https://archive.stsci.edu/ and https://mast.stsci.edu/api/v0/
- JPL Small-Body Database API: https://ssd-api.jpl.nasa.gov/doc/cad.html

## Troubleshooting

If you retrained but still see old responses:

- stop the old backend first with `make stop`
- restart with `make run` or `make run-api`
- make sure your request URLs are quoted in `zsh` when they include `?`

If a planet still does not show model-backed output:

- the metadata layer may still have useful information even when the trained model cannot explain that planet
- in that case the app should return a `metadata-only` explanation instead of a blank result

## GitHub Push Checklist

Before pushing, verify:

- generated model artifacts, caches, logs, and awareness state are not tracked
- the app can refresh datasets at startup instead of relying on committed snapshots
- `OPENAI_API_KEY` is optional and not committed

Standard push flow:

```bash
git status
git commit -m "Your commit message"
git push origin main
```
