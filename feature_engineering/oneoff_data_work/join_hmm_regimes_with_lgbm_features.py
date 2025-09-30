#!/usr/bin/env python3
"""
Join HMM regimes (state / p_state_*) with LightGBM features table on timestamp.

Base (left) table: LightGBM features CSV (e.g., merged_features_targets.csv)
Right table: HMM regimes CSV (e.g., regimes.csv)

Usage examples:
  python run/join_hmm_regimes_with_lgbm_features.py \
    --lgbm-csv "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv" \
    --regimes-csv "/Volumes/Extreme SSD/training/BINANCE_BTCUSDT.P, 60/regimes.csv" \
    --mode proba \
    --output "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_with_hmm_proba.csv"

  python run/join_hmm_regimes_with_lgbm_features.py \
    --lgbm-csv .../merged_features_targets.csv \
    --regimes-csv .../regimes.csv \
    --mode state \
    --state-col-name hmm_state \
    --output .../merged_with_hmm_state.csv

Modes:
  - state: include only the single 'state' column (renamed by --state-col-name)
  - proba: include only 'p_state_*' columns (optionally prefixed)
  - both: include both state and p_state_* columns
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'timestamp' not in df.columns:
        # Try to infer from first column
        first = df.columns[0]
        df = df.rename(columns={first: 'timestamp'})
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)


def _select_regime_columns(df: pd.DataFrame, mode: str, *, state_col_name: str = 'hmm_state', proba_prefix: str | None = None) -> pd.DataFrame:
    cols: List[str] = ['timestamp']
    out = pd.DataFrame(df['timestamp']).copy()

    if mode in ('state', 'both'):
        if 'state' in df.columns:
            out[state_col_name] = df['state'].astype(int)
            cols.append(state_col_name)
        else:
            # silently skip if absent
            pass

    if mode in ('proba', 'both'):
        pcols = [c for c in df.columns if c.startswith('p_state_')]
        for c in pcols:
            new_c = f"{proba_prefix}{c}" if proba_prefix else c
            out[new_c] = df[c]
            cols.append(new_c)

    return out[['timestamp'] + [c for c in out.columns if c != 'timestamp']]


def main() -> None:
    ap = argparse.ArgumentParser(description='Join HMM regimes (state/p_state_*) with LightGBM features on timestamp')
    ap.add_argument('--lgbm-csv', type=Path, required=True, help='Path to merged_features_targets.csv (base table)')
    ap.add_argument('--regimes-csv', type=Path, required=True, help='Path to regimes.csv (from HMM run)')
    ap.add_argument('--output', type=Path, required=True, help='Output CSV path')
    ap.add_argument('--mode', choices=['state','proba','both'], default='proba', help='Which regime features to include (default: proba)')
    ap.add_argument('--state-col-name', default='hmm_state', help='Column name for joined state (default: hmm_state)')
    ap.add_argument('--proba-prefix', default='hmm_', help='Optional prefix for p_state_* columns (default: hmm_)')
    args = ap.parse_args()

    print(f"Loading LGBM features: {args.lgbm_csv}")
    base = pd.read_csv(args.lgbm_csv)
    base = _normalize_timestamp_column(base)
    print(f"Base rows={len(base)} cols={base.shape[1]}")

    print(f"Loading regimes: {args.regimes_csv}")
    reg = pd.read_csv(args.regimes_csv)
    reg = _normalize_timestamp_column(reg)
    print(f"Regimes rows={len(reg)} cols={reg.shape[1]}")

    # Select desired regimes columns
    reg_sel = _select_regime_columns(reg, args.mode, state_col_name=args.state_col_name, proba_prefix=args.proba_prefix)
    print(f"Selected regime columns: {list(reg_sel.columns)}")

    # Ensure no name collisions except timestamp
    overlap = (set(base.columns) & set(reg_sel.columns)) - {'timestamp'}
    if overlap:
        print(f"WARNING: Overlapping columns will be suffixed: {sorted(list(overlap))[:10]}")
        reg_sel = reg_sel.rename(columns={c: f"{c}__hmm" for c in overlap})

    merged = pd.merge(base, reg_sel, on='timestamp', how='left', validate='one_to_one')
    print(f"Merged rows={len(merged)} cols={merged.shape[1]}")

    # Bandaid: fill missing probabilities with 0
    if args.mode in ('proba', 'both'):
        # Identify proba columns considering optional prefix
        pcols = [c for c in merged.columns if c.startswith('p_state_')]
        if args.proba_prefix:
            pref = str(args.proba_prefix)
            pcols += [c for c in merged.columns if c.startswith(f"{pref}p_state_")]
        # De-duplicate
        pcols = sorted(list(set(pcols)))
        if pcols:
            na_before = int(merged[pcols].isna().sum().sum())
            merged[pcols] = merged[pcols].fillna(0.0)
            na_after = int(merged[pcols].isna().sum().sum())
            print(f"Filled missing p_state values with 0: cols={len(pcols)} replaced={na_before - na_after}")
        else:
            print("No p_state_* columns found to fill.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote joined table: {args.output}")


if __name__ == '__main__':
    main()
