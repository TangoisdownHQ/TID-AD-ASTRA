# ml/app/data/prep.py
"""
TID-AD-ASTRA Dataset Loader

Priority order:
1️⃣ User uploads            → ml/app/data/uploads/*.csv
2️⃣ NASA Exoplanet Archive  → Live TAP query (CSV)
3️⃣ Local fallback datasets → koi_fallback.csv, nasa_exoplanets.csv
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.system.selfaware import log_dataset_state

# 🔒 Single source of truth for data
BASE_DATA_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DATA_DIR / "uploads"

FALLBACK_DATASETS = [
    BASE_DATA_DIR / "koi_cumulative.csv",
    BASE_DATA_DIR / "koi_fallback.csv",
]

# Training needs a labelled table. `pscomppars` has no disposition column, so the
# previous URL downloaded 56 MB, failed the label check, and fell through to the
# bundled CSV on every single run. `cumulative` is the table the labels come from.
NASA_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+cumulative&format=csv"
)

# Columns that encode the vetting outcome itself. Training on these leaks the
# label: koi_score IS the disposition confidence, and the fpflag_* columns are the
# individual false-positive tests that produce the disposition. Including them
# yields ~99% accuracy that means nothing.
LEAKAGE_COLUMNS = {
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
}

# Identifiers and positions carry no physical signal but correlate with survey
# bookkeeping.
NON_FEATURE_COLUMNS = {"kepid", "ra", "dec", "ra_err", "dec_err"}

# Train on the features the serving catalog can actually supply.
#
# The KOI table has ~100 numeric columns, most of them measurement uncertainties
# that no other catalog publishes. Training on all of them produces a model whose
# decision weight sits almost entirely in columns that must be imputed at
# inference time — the prediction then reflects the training-set median rather
# than the planet. Restricting training to this set keeps train and serve aligned:
# every column here maps to a catalog field via MODEL_FEATURE_SOURCES.
TRAINING_FEATURES = [
    "koi_period",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_model_snr",
    "koi_steff",
    "koi_srad",
    "koi_slogg",
    "koi_kepmag",
]

# Below this many matching columns, assume a non-KOI dataset (a user upload) and
# fall back to "every usable numeric column".
MIN_TRAINING_FEATURE_MATCH = 6

DATA_SOURCES = {
    "nasa_archive": "https://exoplanetarchive.ipac.caltech.edu/",
    "nasa_tap_api": "https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html",
    "open_exoplanet_catalogue": "https://github.com/OpenExoplanetCatalogue/open_exoplanet_catalogue",
    "kepler_koi_table": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative",
}


# =========================================================
# 🧼 CSV SANITIZER (SHAP-SAFE)
# =========================================================
def _safe_read_csv(path_or_buf):
    df = pd.read_csv(
        path_or_buf,
        sep=None,
        engine="python",
        comment="#",
        on_bad_lines="skip",
    )

    # 🔥 Strip bracket-wrapped floats "[7.9982966E-1]"
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\[\]]", "", regex=True)
                .str.strip()
            )

    return df


# =========================================================
# 📦 DATASET LOADER
# =========================================================
def load_kepler_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    uploads = sorted(DATA_DIR.glob("*.csv"))
    if uploads:
        latest = uploads[-1]
        print(f"📂 Using uploaded dataset: {latest.name}")
        df = _safe_read_csv(latest)
        return _prepare_and_log(df, str(latest), "user_upload")

    try:
        print("🌐 Fetching data from NASA Exoplanet Archive...")
        r = requests.get(NASA_TAP_URL, timeout=25)
        r.raise_for_status()
        df = _safe_read_csv(StringIO(r.text))
        print(f"✅ Loaded {len(df)} samples from NASA TAP.")
        return _prepare_and_log(df, NASA_TAP_URL, "nasa_archive")
    except Exception as e:
        print(f"⚠️ NASA fetch failed ({e}) — using fallback datasets.")

    for path in FALLBACK_DATASETS:
        if path.exists():
            print(f"🪐 Using fallback dataset: {path.name}")
            df = _safe_read_csv(path)
            return _prepare_and_log(df, str(path), "fallback")

    raise FileNotFoundError("❌ No valid dataset found.")


# =========================================================
# 🧠 PREP + LOGGING
# =========================================================
def _prepare_and_log(df: pd.DataFrame, dataset_path: str, dataset_source: str):

    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")

    label_col = None
    if "koi_disposition" in df.columns:
        label_col = "koi_disposition"
    elif "koi_pdisposition" in df.columns:
        label_col = "koi_pdisposition"

    if not label_col:
        raise ValueError("❌ No valid label column found (koi_disposition / koi_pdisposition).")

    y = df[label_col].astype(str).str.upper().apply(
        lambda x: 1 if x in ["CONFIRMED", "CANDIDATE"] else 0
    )

    # 🔢 SHAP-safe numeric feature extraction
    excluded = LEAKAGE_COLUMNS | NON_FEATURE_COLUMNS | {label_col, "koi_pdisposition"}

    available = [c for c in TRAINING_FEATURES if c in df.columns]
    if len(available) >= MIN_TRAINING_FEATURE_MATCH:
        candidates = available
        print(f"🎯 Training on {len(candidates)} catalog-alignable features.")
    else:
        candidates = [c for c in df.columns if c not in excluded]
        print(f"🎯 Unrecognised schema — training on {len(candidates)} numeric columns.")

    columns = {}
    for col in candidates:
        if col in excluded:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0:
            columns[col] = coerced

    X = pd.DataFrame(columns, index=df.index)

    dropped = sorted(LEAKAGE_COLUMNS & set(df.columns))
    if dropped:
        print(f"🛡  Excluded label-leaking columns from training: {', '.join(dropped)}")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    if X.empty:
        raise ValueError("❌ No numeric features available.")

    # 🧨 Drop constants (kills SHAP)
    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # The scaler is fitted for callers that want it, but training and inference
    # both use raw values. Gradient-boosted trees are scale-invariant, and the
    # previous code trained on scaled arrays while `explain_prediction` fed raw
    # numbers — every request landed far outside the training distribution, so
    # the model returned effectively the same answer for every planet.
    scaler = StandardScaler().fit(X_train)

    print(f"✅ Dataset ready — {len(X_train)} train / {len(X_test)} test")
    print(f"📊 Labels → confirmed/candidate={int(y.sum())} | false positives={int((y == 0).sum())}")
    print(f"🔗 Source: {dataset_source}")

    log_dataset_state(
        df,
        dataset_path=dataset_path,
        dataset_source=dataset_source,
        source_links=DATA_SOURCES
    )

    return df, X_train, X_test, y_train, y_test, scaler, dataset_source, dataset_path

