#!/usr/bin/env python3
"""
One-off helper to merge HMM v1 and v2 feature CSVs on timestamp.

Defaults target the BINANCE_BTCUSDT.P, 60 dataset under the external drive.

Usage examples:
  python run/merge_hmm_v1_v2_features.py \
    --v1 "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v1_features.csv" \
    --v2 "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v2_features.csv" \
    --output "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v1v2_features.csv"

  # Or rely on defaults and just set output
  python run/merge_hmm_v1_v2_features.py --output \
    "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v1v2_features.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


DEFAULT_BASE = Path("/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60")


def _read_table_with_timestamp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        # Try first column as timestamp
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'timestamp'})
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    df = df.dropna(subset=['timestamp'])
    df = df[~df['timestamp'].duplicated(keep='last')]
    return df.sort_values('timestamp').reset_index(drop=True)


def merge_v1_v2(v1_path: Path, v2_path: Path, how: str = 'inner') -> Tuple[pd.DataFrame, list[str]]:
    v1 = _read_table_with_timestamp(v1_path)
    v2 = _read_table_with_timestamp(v2_path)
    # Avoid accidental overlap other than timestamp
    overlap = (set(v1.columns) & set(v2.columns)) - {'timestamp'}
    renamed: list[str] = []
    if overlap:
        # Rename overlaps from v2 with suffix
        v2 = v2.rename(columns={c: f"{c}_v2" for c in overlap})
        renamed = sorted(list(overlap))
    merged = pd.merge(v1, v2, on='timestamp', how=how, validate='one_to_one')
    return merged.sort_values('timestamp').reset_index(drop=True), renamed


def main() -> None:
    ap = argparse.ArgumentParser(description='Merge HMM v1 and v2 feature CSVs on timestamp')
    ap.add_argument('--v1', type=Path, default=DEFAULT_BASE / 'hmm_v1_features.csv', help='Path to v1 features CSV')
    ap.add_argument('--v2', type=Path, default=DEFAULT_BASE / 'hmm_v2_features.csv', help='Path to v2 features CSV')
    ap.add_argument('--output', type=Path, required=True, help='Output CSV path')
    ap.add_argument('--how', choices=['inner','left','right','outer'], default='inner', help='Join type (default: inner)')
    ap.add_argument('--drop-na', action='store_true', help='Drop rows with any NaN after merge (feature columns only)')
    args = ap.parse_args()

    merged, renamed = merge_v1_v2(args.v1, args.v2, how=args.how)
    if renamed:
        print(f"Renamed overlapping columns from v2 with suffix '_v2': {renamed[:20]}{'...' if len(renamed)>20 else ''}")

    if args.drop_na:
        before = len(merged)
        cols_no_ts = [c for c in merged.columns if c != 'timestamp']
        merged = merged.dropna(subset=cols_no_ts)
        print(f"Dropped {before - len(merged)} rows due to NaNs after merge")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote merged v1+v2: {args.output} (rows={len(merged)}, cols={merged.shape[1]})")


if __name__ == '__main__':
    main()

