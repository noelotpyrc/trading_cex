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
import duckdb  # type: ignore


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


def load_ohlcv_duckdb(
    db_path: str | os.PathLike,
    *,
    table: str = 'ohlcv_btcusdt_1h',
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load OHLCV from a DuckDB table into a normalized DataFrame.

    - Selects columns: timestamp, open, high, low, close, volume
    - Normalizes `timestamp` to UTC-naive and lowercases column names
    - Dedupe on timestamp and sort ascending
    - Optional time range via `start` (inclusive) and `end` (inclusive)
    - Optional `limit` (applied after ordering ascending)
    """
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET TimeZone='UTC';")
        clauses = []
        params: list[object] = []
        if start is not None:
            s = pd.to_datetime(start, utc=True).tz_convert('UTC').tz_localize(None)
            clauses.append("timestamp >= ?")
            params.append(pd.Timestamp(s).to_pydatetime())
        if end is not None:
            e = pd.to_datetime(end, utc=True).tz_convert('UTC').tz_localize(None)
            clauses.append("timestamp <= ?")
            params.append(pd.Timestamp(e).to_pydatetime())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        q = f"SELECT timestamp, open, high, low, close, volume FROM {table}{where} ORDER BY timestamp ASC"
        if limit is not None and int(limit) > 0:
            q = f"{q} LIMIT {int(limit)}"
        df = con.execute(q, params).fetch_df()
    finally:
        con.close()

    # Normalize like CSV path
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'timestamp' not in df.columns:
        raise ValueError("DuckDB query did not return 'timestamp' column")
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    # Keep only the expected columns
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[cols]
    # Coerce numeric
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Drop NaN timestamps and dedupe
    df = df.dropna(subset=['timestamp'])
    df = df[~df['timestamp'].duplicated(keep='last')]
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def latest_complete_hour(df: pd.DataFrame) -> pd.Timestamp:
    if 'timestamp' not in df.columns:
        raise ValueError("Dataframe must include 'timestamp' column")
    last_ts = pd.Timestamp(df['timestamp'].iloc[-1])
    # Align to hour floor
    last_hour = last_ts.floor('h')
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


def validate_hourly_continuity(
    df: pd.DataFrame,
    *,
    end_ts: pd.Timestamp,
    required_hours: int,
) -> None:
    """Fail-fast validation: ensure strictly continuous hourly data ending at end_ts.

    - No missing hours, no duplicates within the last `required_hours` hours
    - Timestamps normalized to hour boundary are expected
    """
    if 'timestamp' not in df.columns:
        raise ValueError("Dataframe must include 'timestamp' column for continuity validation")

    start_ts = pd.Timestamp(end_ts) - pd.Timedelta(hours=required_hours - 1)
    window = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].copy()
    if window.empty:
        raise ValueError("Continuity check window is empty; insufficient data")

    # Normalize to hour floor
    ts_series = pd.to_datetime(window['timestamp'], errors='coerce')
    ts_series = ts_series.dt.floor('h')
    # Duplicates
    if ts_series.duplicated(keep=False).any():
        dups = ts_series[ts_series.duplicated(keep=False)].astype(str).unique().tolist()
        raise ValueError(f"Duplicate hourly timestamps detected in inference window: {dups[:10]}")

    # Missing hours
    expected = pd.date_range(end=end_ts.floor('h'), periods=required_hours, freq='h')
    have = pd.Index(ts_series.unique())
    missing = expected.difference(have)
    if len(missing) > 0:
        raise ValueError(f"Missing {len(missing)} hourly bars in inference window; sample missing: {[str(x) for x in list(missing[:10])]}")



