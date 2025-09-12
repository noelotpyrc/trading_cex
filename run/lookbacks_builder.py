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



