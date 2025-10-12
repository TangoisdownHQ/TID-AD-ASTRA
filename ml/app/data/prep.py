import os
import requests
import pandas as pd
from pathlib import Path
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.system.selfaware import log_dataset_state

DATA_DIR = Path(__file__).resolve().parent / "uploads"
DATA_FALLBACK = Path(__file__).resolve().parent.parent / "data" / "open_exoplanet_catalogue.csv"


def load_kepler_dataset():
    """
    Load the most up-to-date dataset.
    Priority:
    1️⃣ User uploads (CSV/TSV in /uploads)
    2️⃣ NASA Exoplanet Archive (TAP query)
    3️⃣ Open Exoplanet Catalogue fallback
    """

    # =========================================================
    # 1️⃣ Check uploads folder first
    # =========================================================
    uploaded_files = sorted(DATA_DIR.glob("*.csv"))
    if uploaded_files:
        latest = uploaded_files[-1]
        print(f"📂 Using uploaded dataset: {latest.name}")
        df = pd.read_csv(latest)
        X, y, X_train, X_test, y_train, y_test, scaler = _prepare_dataset(df)
        log_dataset_state(df, dataset_path=str(latest), dataset_source="user_upload")
        return df, X_train, X_test, y_train, y_test, scaler, "user_upload", latest

    # =========================================================
    # 2️⃣ Attempt to fetch from NASA Exoplanet Archive
    # =========================================================
    nasa_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        "?query=select+top+5000+*+from+pscomppars&format=csv"
    )
    try:
        print("🌐 Fetching data from NASA Exoplanet Archive...")
        response = requests.get(nasa_url, timeout=20)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        print(f"✅ Loaded {len(df)} samples from NASA Exoplanet Archive.")
        X, y, X_train, X_test, y_train, y_test, scaler = _prepare_dataset(df)
        log_dataset_state(df, dataset_path="NASA_API", dataset_source="nasa_archive")
        return df, X_train, X_test, y_train, y_test, scaler, "nasa_archive", nasa_url
    except Exception as e:
        print(f"⚠️ NASA fetch failed ({e}) — falling back to local dataset.")

    # =========================================================
    # 3️⃣ Fallback to Open Exoplanet Catalogue
    # =========================================================
    if DATA_FALLBACK.exists():
        print(f"🪐 Using fallback dataset: {DATA_FALLBACK.name}")
        df = pd.read_csv(DATA_FALLBACK)
        X, y, X_train, X_test, y_train, y_test, scaler = _prepare_dataset(df)
        log_dataset_state(df, dataset_path=str(DATA_FALLBACK), dataset_source="fallback_catalogue")
        return df, X_train, X_test, y_train, y_test, scaler, "fallback_catalogue", DATA_FALLBACK
    else:
        raise FileNotFoundError("No valid dataset found in uploads or fallback paths.")


def _prepare_dataset(df: pd.DataFrame):
    """
    Prepare features and labels for ML training.
    Simple version — assumes 'pl_name' or 'name' as identifier, 'pl_orbper' or similar numeric features.
    """
    df = df.dropna(axis=1, how="all")
    df = df.select_dtypes(include=["number"]).dropna()
    if df.empty:
        raise ValueError("Dataset does not contain usable numeric features.")

    X = df.drop(df.columns[-1], axis=1)
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✅ Dataset ready — {len(X_train)} train, {len(X_test)} test samples.")
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler

