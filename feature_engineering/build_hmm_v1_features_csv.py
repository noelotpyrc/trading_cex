#!/usr/bin/env python3
"""
One-off script to reuse the existing feature sets.

Build an HMM v1 feature CSV by selecting the minimal observation set from two
existing engineered tables and merging on timestamp.

Inputs (defaults match the current dataset layout):
  --features-csv  Path to merged_features_targets.csv (contains Parkinson 20)
  --lags-csv      Path to merged_lags_targets.csv (contains current-bar logret & log vol delta)
  --output        Output CSV path (will be created/overwritten)

Selected v1 features:
  - close_logret_current_1H          (from lags CSV)
  - log_volume_delta_current_1H      (from lags CSV)
  - close_parkinson_20_1H            (from features CSV)

The script normalizes timestamps, performs an inner join on 'timestamp',
keeps only the selected columns, sorts ascending, and writes CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a proper 'timestamp' column of dtype datetime64[ns].
    Drops rows with unparsable timestamps, de-duplicates on timestamp, and sorts.
    """
    data = df.copy()

    if 'timestamp' in data.columns:
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif 'Unnamed: 0' in data.columns:
        data = data.rename(columns={'Unnamed: 0': 'timestamp'})
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif 'index' in data.columns:
        data = data.rename(columns={'index': 'timestamp'})
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index().rename(columns={'index': 'timestamp'})
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        # Attempt to interpret the first column as timestamp
        first_col = data.columns[0]
        if first_col != 'timestamp':
            maybe_ts = pd.to_datetime(data[first_col], errors='coerce', utc=True)
            if maybe_ts.notna().any():
                data = data.rename(columns={first_col: 'timestamp'})
                data['timestamp'] = maybe_ts.dt.tz_convert('UTC').dt.tz_localize(None)

    if 'timestamp' not in data.columns:
        raise ValueError("Could not infer a 'timestamp' column from input data")

    data = data.dropna(subset=['timestamp'])
    data = data[~data['timestamp'].duplicated(keep='last')]
    data = data.sort_values('timestamp')
    return data


def _read_table_with_timestamp(path: Path) -> pd.DataFrame:
    lower = str(path).lower()
    if lower.endswith(('.parquet', '.pq')):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return _normalize_timestamp_column(df)


def _validate_presence(df: pd.DataFrame, cols: List[str], label: str) -> List[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns in {label}: {missing}")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HMM v1 feature CSV by merging two engineered tables on timestamp")
    ap.add_argument('--features-csv', type=Path, default=Path('/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv'))
    ap.add_argument('--lags-csv', type=Path, default=Path('/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_lags_targets.csv'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--how', choices=['inner', 'left', 'right', 'outer'], default='inner')
    ap.add_argument('--drop-na', action='store_true', help='Drop rows with NaN in any selected feature')
    args = ap.parse_args()

    # v1 feature names and which table they are expected to come from
    v1_from_lags = [
        'close_logret_current_1H',
        'log_volume_delta_current_1H',
    ]
    v1_from_features = [
        'close_parkinson_20_1H',
    ]

    # Read inputs
    df_feat = _read_table_with_timestamp(args.features_csv)
    df_lags = _read_table_with_timestamp(args.lags_csv)

    # Validate presence and inform user
    _validate_presence(df_lags, v1_from_lags, label='lags CSV')
    _validate_presence(df_feat, v1_from_features, label='features CSV')

    # Reduce to timestamp + selected columns per table
    left = df_lags[['timestamp'] + [c for c in v1_from_lags if c in df_lags.columns]].copy()
    right = df_feat[['timestamp'] + [c for c in v1_from_features if c in df_feat.columns]].copy()

    # Merge
    merged = pd.merge(left, right, on='timestamp', how=args.how, validate='one_to_one')

    # Keep exactly the v1 columns we expect (and timestamp)
    ordered_cols = ['timestamp'] + v1_from_lags + v1_from_features
    keep_cols = [c for c in ordered_cols if c in merged.columns]
    merged = merged[keep_cols].sort_values('timestamp')

    # Optional NA drop
    if args.drop_na:
        before = len(merged)
        merged = merged.dropna(subset=[c for c in merged.columns if c != 'timestamp'])
        after = len(merged)
        print(f"Dropped {before - after} rows due to NaNs in selected features")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"HMM v1 features written: {args.output} | rows={len(merged)} cols={len(merged.columns)}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == '__main__':
    main()

