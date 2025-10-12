 # 🌌 TID-AD-ASTRA  
### _Decoding the Universe, One Planet at a Time_  
**Built for the 2025 NASA Space Apps Challenge – Team TangoisdownHQ  

---

## 🛰️ Overview
**TID-AD-ASTRA** is an explainable AI system that analyzes and interprets planetary data from NASA’s Exoplanet Archive and the Open Exoplanet Catalogue.  
It predicts potential exoplanet habitability while explaining the **why** behind every decision — giving scientists, educators, and explorers transparent insight into how AI understands alien worlds.

The name means **“To the Stars”**, symbolizing our mission to make deep-space data more interpretable, accessible, and open to everyone.

---

## 🧠 Core Features

### 1. Explainable AI for Planetary Habitability
- Predicts exoplanet classification and habitability index.  
- Generates interpretable explanations for each prediction.  
- Detects missing or incomplete data and provides contextual diagnostics.

### 2. Data Lineage & Provenance
- Integrates both **NASA Exoplanet Archive** and **Open Exoplanet Catalogue**.  
- Tracks the dataset origin for each trained model artifact.  
- Ensures transparency in AI learning sources and evolution.

### 3. Model Management Console (FastAPI)
- Full model registry with:
  - Metadata (`registry.json`)
  - Lineage traceability
  - Explainability endpoints (`/models/explain`, `/models/lineage`)
- Auto-loads the latest trained model and exposes prediction APIs.

### 4. CLI Mission Console
- Terminal-based “Mission Control” interface for analyzing planets.  
- Displays:
  - Prediction summary  
  - Habitability index  
  - Missing-data diagnostics  
  - Dataset provenance  

---

## 🪐 Example Output

```bash
🧠  Summary:
   BD+20 2457 b is predicted as class 1 with confidence 0.77.
   Top influencing factors: feature_1, feature_0.
   Habitability index: 0.00 — unlikely to support Earth-like life.

🧩  Diagnostics:
   Missing or incomplete data for fields:
   discovery_year, radius_earth, host_star.temperature.
   Default estimates were used where possible.


🧬 Architecture
Layer	Description
Data	NASA Exoplanet Archive (nasa_exoplanets.csv) + Open Exoplanet Catalogue (open_exoplanet_catalogue.csv)
Model	Scikit-learn classifier with feature explainability
Backend	FastAPI service exposing /models, /planets, and /explain endpoints
Interface	CLI console (make run-console) and cURL API examples
Storage	Local model registry and artifacts (models/artifacts/registry.json)

🚀 Quick Start
1. Clone the Repository
git clone https://github.com/TangoisdownHQ/TID-AD-ASTRA.git
cd TID-AD-ASTRA/ml

2. Set Up Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3. Run the Backend
make run

4. Test the Explainability API
curl -X POST "http://127.0.0.1:8000/models/explain" \
  -H "Content-Type: application/json" \
  -d '{"planet_name":"Kepler-442b","features":[]}' | jq

5. Run the Mission Console
make run-console

🧩 API Endpoints
Method	Endpoint	Description
GET	/models/artifacts	List all trained models
GET	/models/latest	Fetch latest model metadata
POST	/models/load	Load latest model into memory
POST	/models/predict	Predict exoplanet class
POST	/models/explain	Explain a planet’s prediction
GET	/models/lineage	Retrieve model lineage and dataset provenance
🪙 Dataset Sources
Source	Description
NASA Exoplanet Archive	https://exoplanetarchive.ipac.caltech.edu/

Open Exoplanet Catalogue	https://github.com/OpenExoplanetCatalogue/open_exoplanet_catalogue/
🔭 Technology Stack

Language: Python 3.11

Framework: FastAPI

ML Library: Scikit-learn

Explainability: SHAP / Feature importance

Data Layer: CSV + JSON registries

Interface: CLI (Rich-based), REST API

🌍 Vision

TID-AD-ASTRA was designed as a foundation for autonomous, explainable AI agents that can:

Adaptively reason about planetary data across star systems.

Integrate with future NASA APIs and spaceborne sensors.

Support real-time decision support for interplanetary logistics.

🧑‍🚀 Team TangoisdownHQ
Role	Name / Handle	Focus
Founder & Engineer	@TangoisdownHQ	
Cybersecurity, AI Infrastructure, Explainability, System Design


🛰️ Contact

For collaboration, testing, or academic exchange:

📫 Email: tangoisdown1@proton.me

💻 GitHub: https://github.com/TangoisdownHQ

🌐 Live Demo (optional): https://tid-adastra.fly.dev

🏁 Submission Info
Region	NASA Space Apps Challenge 2025
Team Name:	TangoisdownHQ
Project:	TID-AD-ASTRA

⚡ License

This project is open-source under the MIT License.
NASA datasets and related content are used under the NASA Open Data policy.

“To the stars — and beyond the noise.”


---
