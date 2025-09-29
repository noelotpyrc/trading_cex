#!/usr/bin/env python3
"""
Build an HMM v2 feature CSV by selecting the 1H/4H/12H/1D v2 observation set
from existing engineered tables and merging on timestamp.

Inputs (defaults match the current dataset layout):
  --features-csv  Path to merged_features_targets_normalized.csv
                  (contains: ret_zscore, rvol, VWAP ratios)
  --lags-csv      Path to merged_lags_targets.csv
                  (contains: intrabar current-bar features)
  --output        Output CSV path (will be created/overwritten)
  --timeframe(s)  One or more TF suffixes (e.g., 1H 4H 12H 1D)

Selected v2 features (by timeframe `TF`):
  - close_ret_zscore_20_`TF`      (features CSV: normalized)
  - volume_rvol_20_`TF`           (features CSV: normalized)
  - close_over_vwap_`TF`          (features CSV: normalized)
  - close_log_ratio_vwap_`TF`     (features CSV: normalized)
  - high_low_range_pct_current_`TF` (lags CSV)
  - close_open_pct_current_`TF`     (lags CSV)

The script normalizes timestamps, merges on 'timestamp', retains only
the selected columns, sorts ascending, and writes CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
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
        print(f"WARNING: Missing columns in {label}: {missing[:8]}{'...' if len(missing)>8 else ''}")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HMM v2 feature CSV by merging normalized features and lags on timestamp")
    ap.add_argument('--features-csv', type=Path, default=Path('/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_normalized.csv'))
    ap.add_argument('--lags-csv', type=Path, default=Path('/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_lags_targets.csv'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--timeframe', '--tf', dest='timeframe', default=None, help='Single timeframe (e.g., 1H, 4H)')
    ap.add_argument('--timeframes', nargs='+', default=None, help='Multiple timeframes (e.g., 1H 4H 12H)')
    ap.add_argument('--how', choices=['inner','left','right','outer'], default='inner')
    ap.add_argument('--drop-na', action='store_true', help='Drop rows with NaN in any selected feature')
    args = ap.parse_args()

    # Resolve TFs
    if args.timeframes:
        tfs = [str(tf).upper() for tf in args.timeframes]
    elif args.timeframe:
        tfs = [str(args.timeframe).upper()]
    else:
        tfs = ['1H']

    # v2 names per source
    v2_from_features = [
        *(f'close_ret_zscore_20_{tf}' for tf in tfs),
        *(f'volume_rvol_20_{tf}' for tf in tfs),
        *(f'close_over_vwap_{tf}' for tf in tfs),
        *(f'close_log_ratio_vwap_{tf}' for tf in tfs),
    ]
    v2_from_lags = [
        *(f'high_low_range_pct_current_{tf}' for tf in tfs),
        *(f'close_open_pct_current_{tf}' for tf in tfs),
    ]

    # Read inputs
    df_feat = _read_table_with_timestamp(args.features_csv)
    df_lags = _read_table_with_timestamp(args.lags_csv)

    # Validate presence
    _validate_presence(df_feat, v2_from_features, label='features_norm CSV')
    _validate_presence(df_lags, v2_from_lags, label='lags CSV')

    # Reduce columns
    left = df_feat[['timestamp'] + [c for c in v2_from_features if c in df_feat.columns]].copy()
    right = df_lags[['timestamp'] + [c for c in v2_from_lags if c in df_lags.columns]].copy()

    # Merge
    merged = pd.merge(left, right, on='timestamp', how=args.how, validate='one_to_one')

    # Order columns
    ordered_cols = ['timestamp'] + v2_from_features + v2_from_lags
    keep_cols = [c for c in ordered_cols if c in merged.columns]
    merged = merged[keep_cols].sort_values('timestamp')

    if args.drop_na:
        before = len(merged)
        merged = merged.dropna(subset=[c for c in merged.columns if c != 'timestamp'])
        print(f"Dropped {before - len(merged)} rows due to NaNs in selected features")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"HMM v2 features written: {args.output} | TFs={tfs} | rows={len(merged)} cols={len(merged.columns)}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == '__main__':
    main()

