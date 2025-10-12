# ml/app/system/model_drift.py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from datetime import datetime


def detect_model_drift(prev_hash: str, current_hash: str, old_df: pd.DataFrame = None, new_df: pd.DataFrame = None):
    """
    Detects dataset or model drift between previous and current datasets.
    Returns (bool, reason)
    """

    # 🔹 Basic hash comparison first
    if prev_hash and current_hash and prev_hash != current_hash:
        if old_df is not None and new_df is not None:
            # Attempt statistical drift detection
            try:
                drift_score = _compare_distributions(old_df, new_df)
                if drift_score > 0.25:
                    return True, f"distribution_drift (score={drift_score:.3f})"
            except Exception:
                pass
        return True, "hash_mismatch"

    # 🔹 If hashes are equal, check for subtle drift
    if old_df is not None and new_df is not None:
        try:
            drift_score = _compare_distributions(old_df, new_df)
            if drift_score > 0.3:
                return True, f"statistical_drift (score={drift_score:.3f})"
            else:
                return False, f"stable (score={drift_score:.3f})"
        except Exception:
            return False, "stable_no_hash_change"

    # 🔹 Default: no drift
    return False, "no_change"


def _compare_distributions(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Compare two datasets statistically using the Kolmogorov–Smirnov test.
    Returns average drift score (0-1).
    """
    common_cols = [c for c in df1.columns if c in df2.columns]
    if not common_cols:
        return 0.0

    scores = []
    for col in common_cols:
        if np.issubdtype(df1[col].dtype, np.number) and np.issubdtype(df2[col].dtype, np.number):
            try:
                score = ks_2samp(df1[col].dropna(), df2[col].dropna()).statistic
                scores.append(score)
            except Exception:
                continue

    if scores:
        mean_score = float(np.mean(scores))
        print(f"📊 Drift score across {len(scores)} features: {mean_score:.3f}")
        return mean_score
    else:
        return 0.0

