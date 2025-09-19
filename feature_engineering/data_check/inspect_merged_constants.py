"""
Inspect a merged features+targets table for constant/degenerate feature columns.

Examples:
  python feature_engineering/inspect_merged_constants.py \
    --merged "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_normalized.csv" \
    --skip-prefixes y_ target_ label_ --report-dir "/tmp/merged_report"

What it reports:
  - Summary (rows/cols, timestamp range)
  - Columns that are all NA
  - Constant columns (nunique <= 1, ignoring NaN)
  - Zero-variance numeric columns (std == 0)
  - Saves CSV reports optionally
"""

import argparse
from pathlib import Path
from typing import List

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
        # Attempt to interpret first column as timestamp
        first_col = df.columns[0]
        if first_col != 'timestamp':
            ts_try = pd.to_datetime(df[first_col], errors='coerce', utc=True)
            if ts_try.notna().any():
                df = df.rename(columns={first_col: 'timestamp'})
                df['timestamp'] = ts_try.dt.tz_convert('UTC').dt.tz_localize(None)

    if 'timestamp' not in df.columns:
        raise ValueError("Could not find/parse 'timestamp' column in merged table")

    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def _basic_summary(df: pd.DataFrame) -> None:
    n_rows, n_cols = df.shape
    ts_dupes = df['timestamp'].duplicated().sum() if 'timestamp' in df.columns else 'n/a'
    ts_min = df['timestamp'].min() if 'timestamp' in df.columns else 'n/a'
    ts_max = df['timestamp'].max() if 'timestamp' in df.columns else 'n/a'
    print(f"Rows={n_rows}, Cols={n_cols}, Duplicated timestamps={ts_dupes}, Range=[{ts_min} .. {ts_max}]")


def _constant_checks(df: pd.DataFrame, skip_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in skip_cols]
    sub = df[cols]
    na_ratio = sub.isna().sum() / len(sub) if len(sub) else 0
    nunique = sub.nunique(dropna=True)
    numeric_cols = sub.select_dtypes(include=[np.number]).columns
    std = sub[numeric_cols].std(numeric_only=True)

    std_full = pd.Series(index=cols, dtype=float)
    std_full.loc[numeric_cols] = std

    out = pd.DataFrame({
        'na_ratio': na_ratio,
        'nunique': nunique,
        'std': std_full,
        'dtype': sub.dtypes.astype(str),
    }).sort_index()
    return out


def main():
    ap = argparse.ArgumentParser(description='Inspect merged features+targets for constant feature columns')
    ap.add_argument('--merged', type=Path, required=True, help='Path to merged CSV/Parquet (features + targets)')
    ap.add_argument('--report-dir', type=Path, default=None, help='Directory to write CSV reports')
    ap.add_argument('--skip-prefixes', nargs='*', default=['y_', 'target', 'label'], help='Column prefixes to skip from checks')
    args = ap.parse_args()

    df = _read_table_with_timestamp(args.merged)
    _basic_summary(df)

    # Columns to skip: timestamp + anything matching skip-prefixes
    skip_cols = ['timestamp']
    if args.skip_prefixes:
        for c in df.columns:
            for p in args.skip_prefixes:
                if str(c).startswith(p):
                    skip_cols.append(c)
                    break

    stats = _constant_checks(df, skip_cols)

    all_na = stats[stats['na_ratio'] >= 1.0]
    constants = stats[stats['nunique'] <= 1]
    zero_var = stats[(~stats['std'].isna()) & (stats['std'] == 0)]

    print("\nAll-NA columns (features only):", len(all_na))
    if not all_na.empty:
        print(all_na.index.tolist())

    print("\nConstant columns (nunique<=1):", len(constants))
    if not constants.empty:
        print(constants.index.tolist())

    print("\nZero-variance numeric columns:", len(zero_var))
    if not zero_var.empty:
        print(zero_var.index.tolist())

    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.report_dir / 'merged_feature_constant_stats.csv')
        all_na.to_csv(args.report_dir / 'merged_feature_all_na.csv')
        constants.to_csv(args.report_dir / 'merged_feature_constants.csv')
        zero_var.to_csv(args.report_dir / 'merged_feature_zero_variance.csv')
        print(f"Reports written to {args.report_dir}")


if __name__ == '__main__':
    main()

