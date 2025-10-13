
# 🌌 **TID-AD-ASTRA**
### _Decoding the Universe, One Planet at a Time_

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![NASA Space Apps](https://img.shields.io/badge/NASA_Space_Apps-2025-red?logo=nasa&logoColor=white)

**Built for the 2025 NASA Space Apps Challenge — Team TangoisdownHQ**

---

## 🛰️ Overview

**TID-AD-ASTRA** is an **explainable AI system** that analyzes and interprets planetary data from **NASA’s Exoplanet Archive** and the **Open Exoplanet Catalogue**.  
It predicts potential **exoplanet habitability** while explaining the *why* behind every decision — giving scientists, educators, and explorers transparent insight into how AI understands alien worlds.

> The name means **“To the Stars”**, symbolizing our mission to make deep-space data more interpretable, accessible, and open to everyone.

---

## 🧩 Data Integration

Users can:
- Upload new datasets directly into:
  ```bash
  ml/app/data/uploads/

python fetch_data.py --source nasa
python fetch_data.py --source open


 ## Core Features
1. Explainable AI for Planetary Habitability

Predicts exoplanet classification and habitability index

Generates interpretable explanations for each prediction

Detects missing or incomplete data and provides contextual diagnostics

2. Data Lineage & Provenance

Integrates NASA Exoplanet Archive + Open Exoplanet Catalogue

Tracks dataset origin for each trained model artifact

Ensures transparency in AI learning and decision-making

3. Model Management Console (FastAPI)

Central registry with:

Metadata (registry.json)

Lineage tracking

Explainability endpoints (/models/explain, /models/lineage)

Auto-loads the latest trained model and exposes prediction APIs

4. CLI Mission Console

Terminal-based “Mission Control” for planetary analysis

Displays:

Prediction summary

Habitability index

Missing-data diagnostics

Dataset provenance

🪐 Example Output

## 🧠 Summary:
   BD+20 2457 b is predicted as class 1 with confidence 0.77.
   Top influencing factors: feature_1, feature_0.
   Habitability index: 0.00 — unlikely to support Earth-like life.

## 🧩 Diagnostics:
   Missing or incomplete data for fields:
   discovery_year, radius_earth, host_star.temperature.
   Default estimates were used where possible.

## 🧬 Architecture
Layer	Description
Data	NASA Exoplanet Archive + Open Exoplanet Catalogue
Model	Scikit-learn classifier with explainability
Backend	FastAPI service exposing /models, /planets, /explain
Interface	CLI console + REST API
Storage	Local model registry (models/artifacts/registry.json)

## Quick Start
## 1. Clone the Repository
git clone https://github.com/TangoisdownHQ/TID-AD-ASTRA.git
cd TID-AD-ASTRA/ml

## 2. Set Up Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## 3. Run the Backend
make run

## 4. Test the Explainability API
curl -X POST "http://127.0.0.1:8000/models/explain" \
  -H "Content-Type: application/json" \
  -d '{"planet_name":"Kepler-442b","features":[]}' | jq

5. Launch the Mission Console
make run-console


 ## Add Your Own Datasets
TID-AD-ASTRA supports open exploration — upload your own exoplanet datasets, telescope data, or simulated planetary environments.

1. Upload CSV Files

Place your dataset in:

ml/app/data/uploads/
└── your_exoplanet_data.csv


## CSV Schema (flexible, missing values handled automatically):


Column	Description
planet_name	Planet identifier
mass_earth	Mass (in Earth masses)
radius_earth	Radius (in Earth radii)
orbital_period_days	Orbital period
semi_major_axis_au	Semi-major axis
equilibrium_temperature_k	Temperature (K)
eccentricity	Orbital eccentricity
discovery_year	Discovery year
distance_from_earth_ly	Distance (light-years)
host_star_temperature	Star temperature (K)
host_star_spectral_type	Star type (e.g., G2V)


## 2. Fetch Data Automatically
python fetch_data.py --source nasa
python fetch_data.py --source open


## 3. Train a Model on Custom Data
python -m app.models.classifier --train ml/app/data/uploads/your_exoplanet_data.csv

 
 ## Dataset Sources
Source	Description
NASA Exoplanet Archive
	Official exoplanet dataset
Open Exoplanet Catalogue
	Community-curated planetary data


## Technology Stack
Category	Technology
Language	Python 3.11
Framework	FastAPI
ML Library	Scikit-learn
Explainability	SHAP / Feature Importance
Data Layer	CSV + JSON registries
Interface	CLI (Rich-based) + REST API


## 🌍 Vision

TID-AD-ASTRA is a foundation for autonomous, explainable AI agents that can:

Reason adaptively about planetary data across star systems

Integrate with future NASA APIs and sensors

Support real-time decision support for interplanetary logistics


## 🚀 Future Enhancements
🧠 AI & Modeling

Expand models with neural networks and ensemble systems

Add SHAP visual dashboards for interpretability

Enable real-time inference from streaming telemetry


## 🛰️ Data & Integration

Automate synchronization with NASA’s live API

Merge data from multiple observatories (Kepler, TESS, Gaia)

Visualize data provenance with interactive lineage graphs


## 🪐 Interface & Visualization

Build a web-based Mission Console with dynamic dashboards

Support drag-and-drop dataset uploads

Add 3D planetary system visualizations (using Three.js + NASA JPL data)


## ☁️ Deployment & Scale

Containerize via Docker and deploy on Fly.io or Kubernetes

Add secure API keys and user roles for collaboration

Implement automated retraining and model versioning pipelines


## 🧬 Long-Term Vision

Develop an autonomous science agent capable of reading research papers, updating models, and proposing new exploration targets

Use generative AI to simulate unseen planetary systems

Integrate with spacecraft telemetry for live adaptive AI analytics

“To the stars — and beyond the noise.” 🌠


##  Team TangoisdownHQ
Role	Name / Handle	Focus
Founder & Engineer	@TangoisdownHQ
Cybersecurity, AI Infrastructure, Explainability, System Design


## 🛰️ Contact
📫 Email: tangoisdown1@proton.me

💻 GitHub: TangoisdownHQ

🌐 Live Demo: https://tid-adastra.fly.dev


## 🏁 Submission Info
Field	Value
Region	NASA Space Apps Challenge 2025
Team Name	TangoisdownHQ
Project	TID-AD-ASTRA


## ⚡ License
This project is open-source under the MIT License.
NASA datasets and related content are used under the NASA Open Data Policy.

## "TID-AD-ASTRA — decoding the universe, one planet at a time & beyond the noise" 🌌
