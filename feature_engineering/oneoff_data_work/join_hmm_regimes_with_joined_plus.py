#!/usr/bin/env python3
"""
Join HMM regimes (state / p_state_*) with the 'joined_plus' feature table,
dropping all lag-related columns from the base table first.

Default targets the BTC 1H dataset:
  Base:   /Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_joined_plus_24h_tp_sl.csv
  Regimes:/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/regimes.csv

Usage examples:
  # Join probabilities only, fill missing with 0
  python run/join_hmm_regimes_with_joined_plus.py \
    --base "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_joined_plus_24h_tp_sl.csv" \
    --regimes "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/regimes.csv" \
    --mode proba \
    --output "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/joined_plus_with_hmm_proba.csv"

  # Join both state and probabilities
  python run/join_hmm_regimes_with_joined_plus.py --mode both --output /tmp/out.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_BASE = Path("/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_joined_plus_24h_tp_sl.csv")
DEFAULT_REGIMES = Path("/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/regimes.csv")


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'timestamp' not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: 'timestamp'})
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)


def _drop_lag_columns(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    pat = re.compile(pattern, flags=re.IGNORECASE)
    keep = ['timestamp']
    drop_cols: List[str] = []
    for c in df.columns:
        if c == 'timestamp':
            continue
        if pat.search(c):
            drop_cols.append(c)
        else:
            keep.append(c)
    if drop_cols:
        print(f"Dropping lag-related columns: count={len(drop_cols)}")
    else:
        print("No lag-related columns matched the pattern; keeping all columns.")
    return df[keep]


def _select_regimes(df: pd.DataFrame, mode: str, *, state_col_name: str, proba_prefix: str | None) -> pd.DataFrame:
    out = pd.DataFrame({'timestamp': df['timestamp']})
    if mode in ('state', 'both') and 'state' in df.columns:
        out[state_col_name] = df['state'].astype(int)
    if mode in ('proba', 'both'):
        pcols = [c for c in df.columns if c.startswith('p_state_')]
        for c in pcols:
            new_c = f"{proba_prefix}{c}" if proba_prefix else c
            out[new_c] = df[c]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Join HMM regimes with joined_plus features; drop lag columns first')
    ap.add_argument('--base', type=Path, default=DEFAULT_BASE)
    ap.add_argument('--regimes', type=Path, default=DEFAULT_REGIMES)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=['state','proba','both'], default='proba')
    ap.add_argument('--state-col-name', default='hmm_state')
    ap.add_argument('--proba-prefix', default='hmm_')
    ap.add_argument('--drop-pattern', default=r'_lag_', help='Regex to identify lag-related columns (default: _lag_)')
    args = ap.parse_args()

    print(f"Reading base: {args.base}")
    base = pd.read_csv(args.base)
    base = _normalize_timestamp_column(base)
    before_cols = base.shape[1]
    base = _drop_lag_columns(base, args.drop_pattern)
    print(f"Base columns: {before_cols} -> {base.shape[1]} after lag-drop")

    print(f"Reading regimes: {args.regimes}")
    regimes = pd.read_csv(args.regimes)
    regimes = _normalize_timestamp_column(regimes)

    reg_sel = _select_regimes(regimes, args.mode, state_col_name=args.state_col_name, proba_prefix=args.proba_prefix)
    print(f"Regime columns selected: {list(reg_sel.columns)}")

    merged = pd.merge(base, reg_sel, on='timestamp', how='left', validate='one_to_one')
    print(f"Merged rows={len(merged)} cols={merged.shape[1]}")

    # Fill missing probabilities with 0 as a bandaid
    if args.mode in ('proba','both'):
        pcols = [c for c in merged.columns if c.startswith('p_state_') or c.startswith(f"{args.proba_prefix}p_state_")]
        pcols = sorted(list(set(pcols)))
        if pcols:
            na_before = int(merged[pcols].isna().sum().sum())
            merged[pcols] = merged[pcols].fillna(0.0)
            print(f"Filled missing p_state values with 0: cols={len(pcols)} replaced={na_before}")
        else:
            print("No p_state_* columns found to fill.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote output: {args.output}")


if __name__ == '__main__':
    main()

