#!/usr/bin/env python3
"""
Inference data loading utilities for hourly OHLCV feeds.

Responsibilities:
- Read CSV (or DataFrame in future), normalize timestamps (UTC-naive), dedupe, sort
- Validate last row is aligned to the hour boundary
- Validate minimum history coverage (>= required hours)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class HistoryRequirement:
    required_hours: int
    buffer_hours: int = 0

    @property
    def total_required_hours(self) -> int:
        return int(self.required_hours + max(0, self.buffer_hours))


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if 'timestamp' in data.columns:
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index().rename(columns={'index': 'timestamp'})
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        # Fallback: try first column
        first_col = data.columns[0]
        if first_col != 'timestamp':
            maybe_ts = pd.to_datetime(data[first_col], errors='coerce', utc=True)
            data = data.rename(columns={first_col: 'timestamp'})
            data['timestamp'] = maybe_ts.dt.tz_convert('UTC').dt.tz_localize(None)

    if 'timestamp' not in data.columns:
        raise ValueError("Could not infer a 'timestamp' column from input data")

    data = data.dropna(subset=['timestamp'])
    data = data[~data['timestamp'].duplicated(keep='last')]
    data = data.sort_values('timestamp')
    return data


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv', '.txt'):
        df = pd.read_csv(path)
    elif ext in ('.parquet', '.pq'):
        df = pd.read_parquet(path)
    elif ext in ('.pkl', '.pickle'):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported input format: {ext}")
    df = _normalize_timestamp_column(df)
    # Standardize OHLCV casing if present
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def latest_complete_hour(df: pd.DataFrame) -> pd.Timestamp:
    if 'timestamp' not in df.columns:
        raise ValueError("Dataframe must include 'timestamp' column")
    last_ts = pd.Timestamp(df['timestamp'].iloc[-1])
    # Align to hour floor
    last_hour = last_ts.floor('H')
    if last_ts != last_hour:
        # Drop the partial last bar by returning its previous hour
        return last_hour
    return last_ts


def trim_to_latest_complete_hour(df: pd.DataFrame) -> pd.DataFrame:
    last_hour = latest_complete_hour(df)
    return df[df['timestamp'] <= last_hour].copy()


def ensure_min_history(df: pd.DataFrame, hours_required: int) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Ensure at least `hours_required` hours of history up to the latest complete hour.

    Returns the trimmed DataFrame up to the latest complete hour, and that timestamp.
    Raises ValueError if insufficient coverage.
    """
    trimmed = trim_to_latest_complete_hour(df)
    if trimmed.empty:
        raise ValueError("No complete-hour rows available in input data")
    last_ts = pd.Timestamp(trimmed['timestamp'].iloc[-1])
    earliest_ok = last_ts - pd.Timedelta(hours=hours_required)
    if trimmed['timestamp'].min() > earliest_ok:
        have_hours = (last_ts - trimmed['timestamp'].min()) / pd.Timedelta(hours=1)
        raise ValueError(
            f"Insufficient history: have ~{int(have_hours)}h, require >= {hours_required}h through {last_ts}"
        )
    return trimmed, last_ts



