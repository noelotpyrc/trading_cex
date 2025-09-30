"""
Join multiple merged features+targets tables on timestamp, dropping redundant columns.

Assumptions:
- Inputs share the same timestamps (inner join is used by default).
- Overlapping columns with identical values across inputs are redundant and dropped.
- If overlapping columns differ, both are kept; the later file's copy gets a suffix like '__dup2'.

Usage example:
  python feature_engineering/join_merged_datasets.py \
    --inputs \
      "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_normalized.csv" \
      "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_lag.csv" \
      "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_other.csv" \
    --output \
      "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_joined.csv"
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _read_table_with_timestamp(path: Path) -> pd.DataFrame:
    lower = str(path).lower()
    if lower.endswith('.parquet') or lower.endswith('.pq'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'timestamp'})
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        # Try first column as timestamp
        first_col = df.columns[0]
        if first_col != 'timestamp':
            ts_try = pd.to_datetime(df[first_col], errors='coerce', utc=True)
            if ts_try.notna().any():
                df = df.rename(columns={first_col: 'timestamp'})
                df['timestamp'] = ts_try.dt.tz_convert('UTC').dt.tz_localize(None)

    if 'timestamp' not in df.columns:
        raise ValueError(f"No 'timestamp' column found in {path}")

    df = df.dropna(subset=['timestamp']).copy()
    # Deduplicate timestamps, keep last occurrence
    df = df[~df['timestamp'].duplicated(keep='last')]
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def _series_equal(a: pd.Series, b: pd.Series, tol: float) -> bool:
    # Handle both numeric
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        a_vals = a.to_numpy(dtype=float, copy=False)
        b_vals = b.to_numpy(dtype=float, copy=False)
        # Treat NaNs as equal
        both_nan = np.isnan(a_vals) & np.isnan(b_vals)
        close = np.isclose(a_vals, b_vals, rtol=0, atol=tol, equal_nan=False)
        return bool(np.all(close | both_nan))
    # Fallback: compare as strings with explicit NA token
    na_token = object()
    a_obj = a.where(~a.isna(), other=na_token).astype(str)
    b_obj = b.where(~b.isna(), other=na_token).astype(str)
    return bool((a_obj == b_obj).all())


def join_merged(inputs: List[Path], how: str = 'inner', tol: float = 1e-9) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if len(inputs) < 2:
        raise ValueError('Provide at least two input files with --inputs')

    dfs = [_read_table_with_timestamp(p) for p in inputs]
    # Start with the first as base
    merged = dfs[0].copy()
    print(f"Base: rows={len(merged)}, cols={merged.shape[1]} from {inputs[0]}")
    dropped_redundant: List[str] = []
    conflicts_kept: List[str] = []

    for i, df in enumerate(dfs[1:], start=2):
        print(f"Merging input #{i}: rows={len(df)}, cols={df.shape[1]} from {inputs[i-1]}")
        before_rows = len(merged)
        merged = pd.merge(merged, df, on='timestamp', how=how, suffixes=('', f'__dup{i}'))
        print(f"  -> after join: rows={len(merged)} (delta={len(merged)-before_rows}) cols={merged.shape[1]}")

        # Identify overlapping columns in this merge
        for col in df.columns:
            if col == 'timestamp':
                continue
            dup_col = f"{col}__dup{i}"
            if dup_col in merged.columns and col in merged.columns:
                # Compare values
                try:
                    equal = _series_equal(merged[col], merged[dup_col], tol)
                except Exception:
                    equal = False
                if equal:
                    merged = merged.drop(columns=[dup_col])
                    dropped_redundant.append(dup_col)
                else:
                    conflicts_kept.append(dup_col)

    return merged, dropped_redundant, conflicts_kept


def main():
    ap = argparse.ArgumentParser(description='Join merged features+targets tables on timestamp, drop redundant overlaps')
    ap.add_argument('--inputs', nargs='+', required=True, help='Two or more CSV/Parquet files to join')
    ap.add_argument('--output', type=Path, required=True, help='Output CSV/Parquet path')
    ap.add_argument('--how', type=str, default='inner', choices=['inner', 'outer', 'left', 'right'], help='Join type')
    ap.add_argument('--tol', type=float, default=1e-9, help='Numeric equality tolerance for redundancy check')
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    merged, dropped, conflicts = join_merged(inputs, how=args.how, tol=args.tol)

    print(f"Dropped redundant columns: {len(dropped)}")
    if dropped:
        print(dropped[:50] + (['...'] if len(dropped) > 50 else []))
    print(f"Overlapping columns kept due to conflicts: {len(conflicts)}")
    if conflicts:
        print(conflicts[:50] + (['...'] if len(conflicts) > 50 else []))

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    lower = str(out).lower()
    if lower.endswith('.parquet') or lower.endswith('.pq'):
        merged.to_parquet(out)
    else:
        merged.to_csv(out, index=False)
    print(f"Wrote joined table: {out} (rows={len(merged)}, cols={merged.shape[1]})")


if __name__ == '__main__':
    main()

