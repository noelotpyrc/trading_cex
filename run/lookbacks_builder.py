#!/usr/bin/env python3
"""
Build in-memory lookbacks for the latest bar using existing feature_engineering utilities.
"""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

# Reuse existing utilities
from feature_engineering.utils import generate_timeframe_lookbacks


def build_latest_lookbacks(
    df_1h: pd.DataFrame,
    *,
    window_hours: int,
    timeframes: Iterable[str] = ("1H", "4H", "12H", "1D"),
) -> Dict[str, pd.DataFrame]:
    """
    Return per-timeframe lookbacks ending at the latest row of df_1h.
    - df_1h: must contain a 'timestamp' column (UTC-naive) and be sorted.
    - window_hours: base window size in 1H bars (e.g., 30d * 24 = 720 hours).
    """
    if 'timestamp' not in df_1h.columns:
        raise ValueError("df_1h must include a 'timestamp' column")
    if len(df_1h) == 0:
        raise ValueError("df_1h is empty")

    # Ensure an hourly index-like alignment: use positional index at the last row
    df_sorted = df_1h.sort_values('timestamp').reset_index(drop=True)
    current_idx = len(df_sorted) - 1
    lookbacks = generate_timeframe_lookbacks(
        data=df_sorted.set_index('timestamp'),
        current_idx=current_idx,
        window_size=int(window_hours),
        timeframes=timeframes,
    )
    return lookbacks


def trim_lookbacks_to_base_window(
    lookbacks_by_tf: Dict[str, pd.DataFrame],
    *,
    base_hours: int = 720,
) -> Dict[str, pd.DataFrame]:
    """
    Trim lookbacks so that feature computation uses the exact base window length
    used in training (e.g., 720 hours = 30 days):
      - 1H: last 720 rows
      - 4H: last 720/4 rows
      - 12H: last 720/12 rows
      - 1D: last 720/24 rows
    Extra rows (from buffer) are retained for completeness only, not used.
    """
    def _hours_per_bar(tf: str) -> int:
        tfu = str(tf).upper()
        if tfu in ("1H", "H", "60T"):
            return 1
        if tfu.endswith("H"):
            try:
                return int(tfu[:-1])
            except Exception:
                return 1
        if tfu.endswith("D"):
            try:
                return int(tfu[:-1]) * 24
            except Exception:
                return 24
        # Fallback: treat as hourly
        return 1

    trimmed: Dict[str, pd.DataFrame] = {}
    for tf, df in lookbacks_by_tf.items():
        if df is None or df.empty:
            trimmed[tf] = df
            continue
        hpb = max(1, _hours_per_bar(tf))
        expect = max(1, base_hours // hpb)
        # keep the last `expect` rows
        trimmed[tf] = df.tail(expect)
    return trimmed


def validate_lookbacks_exact(
    lookbacks_by_tf: Dict[str, pd.DataFrame],
    *,
    base_hours: int,
    end_ts: pd.Timestamp,
) -> None:
    """Validate lookbacks have exact expected lengths and alignment per timeframe.

    Requirements:
      - 1H/4H/12H/1D lengths: 720/180/60/30 for base_hours=720
      - Last index aligns to end_ts (1H) or floor(end_ts, TF) for higher TFs
      - OHLCV columns present and non-NaN
    """
    def _hours_per_bar(tf: str) -> int:
        tfu = str(tf).upper()
        if tfu in ("1H", "H", "60T"):
            return 1
        if tfu.endswith("H"):
            try:
                return int(tfu[:-1])
            except Exception:
                return 1
        if tfu.endswith("D"):
            try:
                return int(tfu[:-1]) * 24
            except Exception:
                return 24
        return 1

    required_cols = ['open','high','low','close','volume']
    for tf, df in lookbacks_by_tf.items():
        if df is None or df.empty:
            raise ValueError(f"Lookback for {tf} is empty at {end_ts}")
        hpb = max(1, _hours_per_bar(tf))
        expect = max(1, base_hours // hpb)
        if len(df) != expect:
            raise ValueError(f"Lookback length mismatch for {tf}: got {len(df)} expected {expect}")
        # alignment
        last_idx = pd.to_datetime(df.index[-1], errors='coerce')
        aligned = end_ts.floor('h') if tf in ("1H","H","60T") else end_ts.floor(tf)
        if pd.Timestamp(last_idx) != pd.Timestamp(aligned):
            raise ValueError(f"Lookback alignment mismatch for {tf}: last={last_idx} expected={aligned}")
        # columns present and not NA
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {tf} lookback")
        if df[required_cols].isna().any().any():
            bad = df[required_cols].isna().any(axis=1)
            bad_ts = [str(x) for x in df.index[bad][:10]]
            raise ValueError(f"NaNs in {tf} lookback OHLCV near end; sample rows: {bad_ts}")



