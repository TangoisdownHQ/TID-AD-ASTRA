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

## Repo Layout

- `ml/app/main.py`: FastAPI app
- `ml/app/routes/planets.py`: planet catalog and metadata endpoints
- `ml/app/routes/chat.py`: natural-language catalog chat
- `ml/app/routes/datasets.py`: custom dataset upload and preview endpoints
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

4. In a second terminal, launch the CLI:

```bash
make cli-explain
```

5. In the CLI, choose either:

- `Analyze planet` for the original select-and-report flow
- `Chat with catalog` for natural-language search, compare, and planet lookup
- `Upload custom dataset` to send a CSV to the backend
- `Browse uploaded datasets` to preview and sort uploaded CSV files

6. Smoke-test the catalog:

```bash
curl http://127.0.0.1:8080/planets/all?limit=5
curl "http://127.0.0.1:8080/planets/info?name=Kepler-442b"
curl "http://127.0.0.1:8080/planets/search?query=kepler"
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

| Source | Purpose |
|--------|---------|
| NASA Exoplanet Archive | Primary exoplanet parameters and discovery metadata |
| Open Exoplanet Catalogue | Community-maintained supplemental planet records |
| AstroML dataset | Additional exoplanet tabular reference data |
| NASA KOI fallback | Local fallback when richer catalogs are unavailable |

## What Changed For Multi-Source Support

- The backend now merges NASA, OEC, KOI, and AstroML files through one metadata layer.
- `/planets/all` returns de-duplicated records across sources instead of reading only one dataset.
- `/planets/info` and `/planets/search` now understand both NASA-style and OEC-style column names.
- Dataset refresh now includes AstroML alongside NASA and OEC.
- `/chat/ask` adds deterministic natural-language search, info, compare, and ranking over the catalog.
- The CLI now includes a chat mode that can answer catalog questions and then jump into full planet analysis.
- Chat sessions now support follow-up context, result references like `tell me about the second result`, and pagination with `next`, `prev`, or `page 2`.
- Optional OpenAI-enhanced answers can be enabled from the CLI when `OPENAI_API_KEY` is configured.

## Optional OpenAI Answer Layer

If you want the chat responses rewritten into a more conversational answer style on top of the deterministic catalog results, set:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-5.4-mini
```

The app still performs search, filtering, compare logic, and pagination locally first. OpenAI is only used to rewrite the structured result into a cleaner answer when enabled.

## More Data Sources To Add Next

The best next sources are official mission archives with complementary schemas:

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- ESA Gaia Archive and Gaia user services: https://www.cosmos.esa.int/web/gaia-users/register
- JPL Small-Body Database API: https://ssd-api.jpl.nasa.gov/doc/cad.html
- MAST mission archives: https://archive.stsci.edu/

Recommended design:

- keep exoplanets and Solar System bodies as separate source families
- use Gaia and MAST as enrichment layers, not replacements for the planet catalog
- keep uploaded user CSVs isolated from curated source files so provenance stays clear

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
