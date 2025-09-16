#!/usr/bin/env python3
from __future__ import annotations

"""
Inspect a features CSV (e.g., features_all_tf.csv) and print a concise summary:
  - Shape, timestamp range, missing counts
  - Number of columns overall and by timeframe suffix (_1H/_4H/_12H/_1D)
  - Sample of column names

Usage:
  python utils/inspect_features_csv.py \
    --csv "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/features_all_tf.csv" \
    --head 3
"""

import argparse
from pathlib import Path
import sys
from collections import Counter

import pandas as pd


def _suffix(c: str) -> str | None:
    for s in ("_1H", "_4H", "_12H", "_1D"):
        if str(c).endswith(s):
            return s
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect a features CSV and summarize columns')
    parser.add_argument('--csv', required=True, help='Path to features CSV')
    parser.add_argument('--head', type=int, default=0, help='Print head(N) rows if > 0')
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f'ERROR: CSV not found: {path}')
        sys.exit(2)

    # Read minimal rows first to get columns quickly
    df_head = pd.read_csv(path, nrows=1)
    cols = list(df_head.columns)
    has_ts = 'timestamp' in cols
    print('Columns:', len(cols))
    print('Has timestamp:', has_ts)

    # Suffix distribution
    suff = [_suffix(c) for c in cols if c != 'timestamp']
    cnt = Counter([s for s in suff if s is not None])
    print('Suffix counts (by timeframe):', dict(cnt))

    # Load full CSV lazily (let pandas handle dtypes)
    df = pd.read_csv(path)
    print('Shape:', df.shape)
    if has_ts:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
        print('Timestamp range:', ts.iloc[0] if len(ts) else None, '..', ts.iloc[-1] if len(ts) else None)
        na_ts = int(ts.isna().sum())
        print('NaN timestamps:', na_ts)

    # Missing values summary (per column, top offenders)
    na_counts = df.isna().sum().sort_values(ascending=False)
    top_na = na_counts[na_counts > 0].head(10)
    if not top_na.empty:
        print('Top NaN columns:', top_na.to_dict())
    else:
        print('No NaNs detected (head analysis)')

    # Show a few example column names
    examples = [c for c in cols if c != 'timestamp'][:20]
    print('Example columns (first 20):')
    for c in examples:
        print('  -', c)

    if args.head and args.head > 0:
        print(df.head(int(args.head)).to_string(index=False))

    print('inspect_features_csv OK')


if __name__ == '__main__':
    main()

