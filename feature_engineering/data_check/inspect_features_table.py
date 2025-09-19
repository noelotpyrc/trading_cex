"""
Inspect a features table (CSV or Parquet) for missing/constant/problematic columns.

Examples:
  python feature_engineering/inspect_features_table.py \
    --features "/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60/features_all_tf_normalized.csv" \
    --report-dir "/tmp/features_report" --skip-head-rows 720 --top 50

Outputs:
  - Prints summary (rows/cols, timestamp range, dupes)
  - Prints columns with 100% NA, >=99% NA, >=50% NA
  - Prints constant and zero-variance numeric columns
  - Writes CSV reports to --report-dir (optional)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
        raise ValueError("Could not find/parse 'timestamp' column in features table")

    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def _basic_summary(df: pd.DataFrame) -> None:
    n_rows, n_cols = df.shape
    ts_dupes = df['timestamp'].duplicated().sum() if 'timestamp' in df.columns else 'n/a'
    ts_min = df['timestamp'].min() if 'timestamp' in df.columns else 'n/a'
    ts_max = df['timestamp'].max() if 'timestamp' in df.columns else 'n/a'
    print(f"Rows={n_rows}, Cols={n_cols}, Duplicated timestamps={ts_dupes}, Range=[{ts_min} .. {ts_max}]")


def _na_stats(df: pd.DataFrame, skip_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in skip_cols]
    sub = df[cols]
    na_count = sub.isna().sum()
    na_ratio = na_count / len(sub) if len(sub) else 0
    nunique = sub.nunique(dropna=True)
    numeric_cols = sub.select_dtypes(include=[np.number]).columns
    std = sub[numeric_cols].std(numeric_only=True)
    # Fill non-numeric std with NaN
    std_full = pd.Series(index=cols, dtype=float)
    std_full.loc[numeric_cols] = std
    res = pd.DataFrame({
        'na_count': na_count,
        'na_ratio': na_ratio,
        'nunique': nunique,
        'std': std_full,
        'dtype': sub.dtypes.astype(str),
    }).sort_values('na_ratio', ascending=False)
    return res


def _print_top(title: str, ser: pd.Series, top: int) -> None:
    print(f"\n{title}")
    if ser.empty:
        print("  (none)")
    else:
        to_show = ser.head(top)
        for name, val in to_show.items():
            print(f"  {name}: {val}")


def _pattern_report(df: pd.DataFrame, patterns: List[str], skip_cols: List[str], top: int) -> None:
    stats = _na_stats(df, skip_cols)
    for p in patterns:
        mask = stats.index.str.contains(p)
        subset = stats.loc[mask]
        if subset.empty:
            continue
        print(f"\nPattern '{p}': cols={len(subset)}")
        _print_top("  Top NA ratios", subset['na_ratio'], top)


def main():
    ap = argparse.ArgumentParser(description='Inspect features table for missing/constant/problematic columns')
    ap.add_argument('--features', type=Path, required=True, help='Path to features CSV or Parquet')
    ap.add_argument('--report-dir', type=Path, default=None, help='Directory to write CSV reports')
    ap.add_argument('--skip-head-rows', type=int, default=0, help='Skip first N rows (e.g., warmup) when computing stats')
    ap.add_argument('--top', type=int, default=50, help='Top N items to print for each category')
    args = ap.parse_args()

    df = _read_table_with_timestamp(args.features)
    _basic_summary(df)

    if args.skip_head_rows > 0 and len(df) > args.skip_head_rows:
        df_stats = df.iloc[args.skip_head_rows:].reset_index(drop=True)
        print(f"Using rows [{args.skip_head_rows}:{len(df)}] for stats (skipping warmup)")
    else:
        df_stats = df

    skip_cols = ['timestamp']
    stats = _na_stats(df_stats, skip_cols)

    # Buckets
    all_na = stats[stats['na_ratio'] >= 1.0]
    almost_all_na = stats[(stats['na_ratio'] >= 0.99) & (stats['na_ratio'] < 1.0)]
    half_na = stats[(stats['na_ratio'] >= 0.5) & (stats['na_ratio'] < 0.99)]
    constants = stats[stats['nunique'] <= 1]
    zero_var = stats[(~stats['std'].isna()) & (stats['std'] == 0)]

    _print_top("Columns with 100% NA", all_na['na_ratio'], args.top)
    _print_top("Columns with >=99% NA", almost_all_na['na_ratio'], args.top)
    _print_top("Columns with >=50% NA", half_na['na_ratio'], args.top)
    _print_top("Constant columns (nunique<=1)", constants['nunique'], args.top)
    _print_top("Zero-variance numeric columns", zero_var['std'], args.top)

    # Patterns for newly added normalized features
    patterns = [
        r'_over_ema_12',
        r'_log_ratio_ema_12',
        r'_dist_ema12_atr',
        r'_macd_line_12_26_over_close',
        r'_macd_histogram_12_26_9_over_close',
        r'_macd_line_12_26_over_atr14',
        r'_macd_histogram_12_26_9_over_atr14',
        r'_bb_width_pct_20_2',
        r'_obv_over_dollar_vol_20',
        r'_adl_over_dollar_vol_20',
        r'_over_vwap',
        r'_log_ratio_vwap',
        r'time_.*_(sin|cos)',
        r'_position_in_range_20$',
    ]
    _pattern_report(df_stats, patterns, skip_cols, args.top)

    # By timeframe suffix report
    suffixes = ['_1H', '_4H', '_12H', '_1D']
    for s in suffixes:
        s_stats = stats[stats.index.str.endswith(s)]
        if s_stats.empty:
            continue
        print(f"\nTimeframe {s}: cols={len(s_stats)} all_na={int((s_stats['na_ratio']>=1.0).sum())} >=99%={(s_stats['na_ratio']>=0.99).sum()} >=50%={(s_stats['na_ratio']>=0.5).sum()}")

    # Save reports
    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.report_dir / 'feature_na_stats.csv')
        all_na.to_csv(args.report_dir / 'all_na_columns.csv')
        almost_all_na.to_csv(args.report_dir / 'almost_all_na_columns.csv')
        half_na.to_csv(args.report_dir / 'half_na_columns.csv')
        constants.to_csv(args.report_dir / 'constant_columns.csv')
        zero_var.to_csv(args.report_dir / 'zero_variance_numeric.csv')
        print(f"Reports written to {args.report_dir}")


if __name__ == '__main__':
    main()

