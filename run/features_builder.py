#!/usr/bin/env python3
"""
Compute multi-timeframe features for the latest bar from lookbacks using
existing feature_engineering functions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from feature_engineering.build_multi_timeframe_features import compute_features_one


def compute_latest_features_from_lookbacks(lookbacks_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Given a dict {tf: lookback_df}, compute features for the latest bar across
    all timeframes and return a single-row DataFrame indexed by the last timestamp.
    """
    if not lookbacks_by_tf:
        raise ValueError("lookbacks_by_tf is empty")

    # Determine the latest timestamp across provided lookbacks
    # Use the 1H lookback last index if available, else the max of last indices
    last_ts = None
    if '1H' in lookbacks_by_tf and lookbacks_by_tf['1H'] is not None and not lookbacks_by_tf['1H'].empty:
        last_ts = pd.to_datetime(lookbacks_by_tf['1H'].index[-1])
    else:
        last_ts = max((pd.to_datetime(df.index[-1]) for df in lookbacks_by_tf.values() if df is not None and not df.empty))

    features: Dict[str, float] = {}
    for tf, lb in lookbacks_by_tf.items():
        if lb is None or lb.empty:
            continue
        feats_tf = compute_features_one(lb, tf=tf, skip_slow=False)
        # compute_features_one may return NaNs; keep as-is
        features.update(feats_tf)

    # Assemble into single-row frame
    row = {k: (np.nan if v is None else v) for k, v in features.items()}
    df = pd.DataFrame([row])
    df.insert(0, 'timestamp', last_ts)
    return df


def validate_features_for_model(booster, features_row: pd.DataFrame) -> None:
    """Ensure model-required features exist and are non-NaN in the features row.

    Aligns to booster.feature_name() and checks for missing/NaN values.
    """
    from model.lgbm_inference import align_features_for_booster

    if 'timestamp' not in features_row.columns:
        raise ValueError("features_row must include 'timestamp'")

    aligned = align_features_for_booster(features_row.drop(columns=['timestamp'], errors='ignore'), booster)
    model_cols = list(booster.feature_name())
    # Missing columns are filled by aligner; we still consider this invalid for strict mode
    missing = [c for c in model_cols if c not in features_row.columns]
    if missing:
        raise ValueError(f"Missing model features in built row: {missing[:20]}")
    na_mask = aligned[model_cols].isna().any(axis=1)
    if bool(na_mask.iloc[0]):
        na_cols = aligned.columns[aligned.iloc[0].isna()].tolist()
        raise ValueError(f"NaN in model features for inference row; columns: {na_cols[:20]}")




