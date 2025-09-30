import argparse
from pathlib import Path
from typing import Iterable, List, Set, Optional

import pandas as pd


# Default feature columns to remove based on EDA findings
DEFAULT_FEATURES_TO_REMOVE: List[str] = [
    # 1H
    "close_roll_spread_20_1H",
    "close_ou_halflife_100_1H",
    # 4H
    "close_roll_spread_20_4H",
    "close_ou_halflife_100_4H",
    # 4H time features (redundant with base 1H time features)
    "time_hour_of_day_4H",
    "time_day_of_week_4H",
    "time_day_of_month_4H",
    "time_month_of_year_4H",
    # 12H
    "close_hurst_100_12H",
    "close_ljung_p_5_100_12H",
    "close_ou_halflife_100_12H",
    "close_roll_spread_20_12H",
    # 12H time features (redundant with base 1H time features)
    "time_hour_of_day_12H",
    "time_day_of_week_12H",
    "time_day_of_month_12H",
    "time_month_of_year_12H",
    # 1D
    "close_vol_ratio_5_50_1D",
    "close_skew_30_1D",
    "close_kurt_30_1D",
    "close_autocorr_1_30_1D",
    "close_hurst_100_1D",
    "close_dominant_cycle_length_50_1D",
    "close_cycle_strength_50_1D",
    "close_ljung_p_5_100_1D",
    "close_ou_halflife_100_1D",
    "close_var_5_50_1D",
    "close_cvar_5_50_1D",
    "close_spectral_entropy_50_1D",
    "close_perm_entropy_3_50_1D",
    "close_roll_spread_20_1D",
    # 1D time features (redundant with base 1H time features)
    "time_hour_of_day_1D",
    "time_day_of_week_1D",
    "time_day_of_month_1D",
    "time_month_of_year_1D",
    'time_hour_cos_1D', 
    'time_hour_sin_1D'
]


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a proper 'timestamp' column of dtype datetime64[ns],
    de-duplicate on timestamp, and sort ascending.
    Handles common CSV patterns where the timestamp may be written as index
    (named 'timestamp') or as the first unnamed/index column.
    """
    data = df.copy()

    if 'timestamp' in data.columns:
        # Parse as UTC-aware then standardize to UTC-naive
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
        # Fallback: attempt to interpret the first column as timestamp
        first_col = data.columns[0]
        if first_col != 'timestamp':
            maybe_ts = pd.to_datetime(data[first_col], errors='coerce', utc=True)
            if maybe_ts.notna().any():
                data = data.rename(columns={first_col: 'timestamp'})
                data['timestamp'] = maybe_ts.dt.tz_convert('UTC').dt.tz_localize(None)

    if 'timestamp' not in data.columns:
        raise ValueError("Could not infer a 'timestamp' column from input data")

    # Drop rows where timestamp could not be parsed, drop duplicates, and sort
    data = data.dropna(subset=['timestamp'])
    data = data[~data['timestamp'].duplicated(keep='last')]
    data = data.sort_values('timestamp')
    return data


def _read_table_with_timestamp(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet and normalize/ensure a 'timestamp' column."""
    lower = str(path).lower()
    if lower.endswith('.parquet') or lower.endswith('.pq'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return _normalize_timestamp_column(df)


def merge_features_only(
    features_1h_path: Path,
    features_multi_path: Path,
    *,
    how: str = 'inner',
) -> pd.DataFrame:
    """
    Merge the two feature CSVs on 'timestamp' and return the merged feature set.
    """
    features_1h = _read_table_with_timestamp(features_1h_path)
    features_multi = _read_table_with_timestamp(features_multi_path)

    # Debug logging for timestamp ranges and counts
    def _ts_info(name: str, df: pd.DataFrame) -> None:
        dtype = df['timestamp'].dtype
        n = len(df)
        unique_n = df['timestamp'].nunique(dropna=True)
        ts_min = df['timestamp'].min() if n else None
        ts_max = df['timestamp'].max() if n else None
        print(f"{name}: rows={n}, unique_ts={unique_n}, dtype={dtype}, min={ts_min}, max={ts_max}")

    _ts_info('features_1h', features_1h)
    _ts_info('features_multi', features_multi)

    # Avoid accidental overlapping non-key columns
    overlapping_columns: Set[str] = set(features_1h.columns).intersection(features_multi.columns) - {'timestamp'}
    if overlapping_columns:
        features_multi = features_multi.rename(columns={c: f"{c}_multi" for c in overlapping_columns})

    features = pd.merge(
        features_1h,
        features_multi,
        on='timestamp',
        how=how,
        validate='one_to_one',
    )
    print(f"features merged: rows={len(features)}, cols={features.shape[1]}")

    # Intersection diagnostics
    f1_set = set(features_1h['timestamp'])
    f2_set = set(features_multi['timestamp'])
    inter_len = len(f1_set & f2_set)
    print(f"timestamp intersection (features_1h vs features_multi): {inter_len}")
    if inter_len == 0:
        only_f1 = list(f1_set - f2_set)[:3]
        only_f2 = list(f2_set - f1_set)[:3]
        print(f"sample only-in-1h: {only_f1}")
        print(f"sample only-in-multi: {only_f2}")

    return features.sort_values('timestamp')


def prepare_targets(targets_path: Path, features_for_alignment: pd.DataFrame) -> pd.DataFrame:
    """
    Read and normalize targets timestamps. If implausible, try inferring a timestamp column.
    As a last resort, align by row-order using features_for_alignment timestamps when lengths match.
    """
    def _ts_info(name: str, df: pd.DataFrame) -> None:
        dtype = df['timestamp'].dtype
        n = len(df)
        unique_n = df['timestamp'].nunique(dropna=True)
        ts_min = df['timestamp'].min() if n else None
        ts_max = df['timestamp'].max() if n else None
        print(f"{name}: rows={n}, unique_ts={unique_n}, dtype={dtype}, min={ts_min}, max={ts_max}")

    targets = _read_table_with_timestamp(targets_path)
    _ts_info('targets', targets)

    def _is_plausible_ts_range(s: pd.Series) -> bool:
        if s.empty:
            return False
        try:
            y_min = s.min().year
            y_max = s.max().year
            return 2000 <= y_min <= 2100 and 2000 <= y_max <= 2100 and s.is_monotonic_increasing
        except Exception:
            return False

    feat_ts_min = features_for_alignment['timestamp'].min() if not features_for_alignment.empty else None
    feat_ts_max = features_for_alignment['timestamp'].max() if not features_for_alignment.empty else None

    if not _is_plausible_ts_range(targets['timestamp']):
        print("Targets timestamp appears implausible. Attempting to infer correct timestamp column...")
        targets_raw = pd.read_csv(targets_path)

        inferred_ts = None
        inferred_col = None
        for col in targets_raw.columns:
            try:
                ts_try = pd.to_datetime(targets_raw[col], errors='coerce', utc=True)
                ts_try = ts_try.dt.tz_convert('UTC').dt.tz_localize(None)
                valid_ratio = ts_try.notna().mean()
                if valid_ratio > 0.95:
                    if _is_plausible_ts_range(ts_try):
                        inferred_ts = ts_try
                        inferred_col = col
                        break
                    # If not strictly in [2000,2100], check overlap with features
                    if feat_ts_min is not None and feat_ts_max is not None:
                        if ts_try.min() <= feat_ts_max and ts_try.max() >= feat_ts_min:
                            inferred_ts = ts_try
                            inferred_col = col
                            break
            except Exception:
                continue

        if inferred_ts is not None:
            print(f"Inferred timestamp column in targets: '{inferred_col}'")
            targets = targets_raw.copy()
            targets['timestamp'] = inferred_ts
            targets = targets.dropna(subset=['timestamp'])
            targets = targets.sort_values('timestamp')
            _ts_info('targets(inferred)', targets)
        elif len(targets_raw) == len(features_for_alignment):
            print("Could not infer a valid timestamp column. Falling back to row-order alignment with features timestamps.")
            targets = targets_raw.copy()
            targets = targets.reset_index(drop=True)
            features_sorted = features_for_alignment.sort_values('timestamp').reset_index(drop=True)
            targets['timestamp'] = features_sorted['timestamp']
            _ts_info('targets(row-aligned)', targets)
        elif len(targets_raw) > len(features_for_alignment):
            diff = len(targets_raw) - len(features_for_alignment)
            print(f"Row-alignment with length mismatch: trimming first {diff} target rows to match features length")
            targets = targets_raw.iloc[diff:].reset_index(drop=True)
            features_sorted = features_for_alignment.sort_values('timestamp').reset_index(drop=True)
            targets['timestamp'] = features_sorted['timestamp']
            _ts_info('targets(row-aligned-trimmed)', targets)
        else:
            print("WARNING: Unable to recover targets timestamps and targets shorter than features; later merge may be partial or empty.")

    return targets.sort_values('timestamp')


def _apply_cleanup(
    df: pd.DataFrame,
    *,
    warmup_rows: int = 720,
    features_to_remove: Iterable[str] = (),
    drop_remaining_na: bool = True,
    na_log_limit: int = 100,
    log_dropped_na: bool = True,
) -> pd.DataFrame:
    """
    Apply EDA-derived cleanup:
      1) Drop the first `warmup_rows` rows (sorted by timestamp)
      2) Remove selected problematic feature columns
      3) Drop any remaining rows with missing values
    """
    out = df.sort_values('timestamp').reset_index(drop=True)
    start_rows, start_cols = out.shape
    print(f"Cleanup start: rows={start_rows}, cols={start_cols}")

    # 1) Warmup row drop
    if warmup_rows and warmup_rows > 0:
        if len(out) > warmup_rows:
            out = out.iloc[warmup_rows:].reset_index(drop=True)
            print(f"Warmup drop: dropped_rows={warmup_rows}")
        else:
            print(f"Warmup drop: skipped (rows={len(out)} <= warmup={warmup_rows})")

    # 2) Feature column drop
    cols_to_drop = [c for c in features_to_remove if c in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)
        print(f"Feature drop: dropped_cols={len(cols_to_drop)} -> {cols_to_drop}")
    else:
        print("Feature drop: dropped_cols=0")

    # 3) Drop remaining rows with NA
    if drop_remaining_na:
        before_na_rows = len(out)
        na_mask = out.isna().any(axis=1)
        if log_dropped_na and na_mask.any():
            dropped_rows = out.loc[na_mask].copy()
            total = len(dropped_rows)
            # 3a) Summarize per-column NA counts among rows to be dropped
            per_col_na = dropped_rows.isna().sum().sort_values(ascending=False)
            if 'timestamp' in per_col_na.index:
                per_col_na = per_col_na.drop('timestamp')
            top_cols = per_col_na.head(na_log_limit)
            print(f"NA row drop candidates: total={total}")
            if not top_cols.empty:
                print("Top columns causing NA (col: count, pct_of_dropped):")
                denom = float(total) if total > 0 else 1.0
                lines = []
                for col, cnt in top_cols.items():
                    pct = (cnt / denom) * 100.0
                    lines.append(f"  {col}: {int(cnt)} ({pct:.1f}%)")
                print("\n".join(lines))
            # 3b) Show sample row-wise NA reasons
            sample = dropped_rows.head(na_log_limit)
            print(f"Sample dropped rows (up to {na_log_limit}) with NA columns:")
            for i in range(len(sample)):
                row = sample.iloc[i]
                ts_val = str(row.get('timestamp', 'n/a'))
                na_cols = [c for c in sample.columns if c != 'timestamp' and pd.isna(row[c])]
                # Trim overly long lists for readability
                na_cols_display = na_cols if len(na_cols) <= 20 else na_cols[:20] + ['...']
                print(f"  ts={ts_val} | na_cols={na_cols_display}")
        out = out.loc[~na_mask].reset_index(drop=True)
        after_na_rows = len(out)
        print(f"NA row drop: dropped_rows={before_na_rows - after_na_rows}")
    else:
        print("NA row drop: skipped (--no-drop-na)")

    final_rows, final_cols = out.shape
    print(f"Cleanup end: rows={final_rows}, cols={final_cols}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge features with targets on timestamp')
    parser.add_argument(
        '--features-dir',
        type=Path,
        default=Path('/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60'),
        help='Directory containing features CSVs',
    )
    parser.add_argument(
        '--targets-dir',
        type=Path,
        default=Path('/Volumes/Extreme SSD/trading_data/cex/targets/BINANCE_BTCUSDT.P, 60'),
        help='Directory containing targets.csv',
    )
    parser.add_argument('--features-1h', type=str, default='features_1h.csv', help='File name for 1H features CSV')
    parser.add_argument('--features-multi', type=str, default='features_4h12h1d.csv', help='File name for multi-timeframe features CSV')
    parser.add_argument(
        '--features-file',
        type=Path,
        default=None,
        help='Optional single features table (CSV/Parquet) containing all timeframes; if set, skips merging 1H and multi',
    )
    parser.add_argument('--targets-file', type=str, default='targets.csv', help='File name for targets CSV')
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Explicit output path (.csv default, or Parquet with suffix/--parquet). Overrides training-dir/dataset/output-name',
    )
    parser.add_argument('--parquet', action='store_true', help='Force Parquet output regardless of extension')
    parser.add_argument(
        '--training-dir',
        type=Path,
        default=Path('/Volumes/Extreme SSD/trading_data/cex/training'),
        help='Base directory for model training datasets',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='BINANCE_BTCUSDT.P, 60',
        help='Dataset subfolder name under training-dir',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='merged_features_targets.csv',
        help='Output filename within training-dir/dataset when --output is not specified',
    )
    parser.add_argument('--warmup-rows', type=int, default=720, help='Drop the first N rows after merge (insufficient lookback)')
    parser.add_argument('--no-drop-na', action='store_true', help="Do not drop remaining rows with any missing values")
    parser.add_argument(
        '--no-default-feature-drop',
        action='store_true',
        help='Do not drop the default EDA-identified problematic feature columns',
    )
    parser.add_argument(
        '--extra-feature-drop',
        nargs='*',
        default=None,
        help='Additional feature columns to drop (space-separated)',
    )
    args = parser.parse_args()

    features_1h_path = args.features_dir / args.features_1h
    features_multi_path = args.features_dir / args.features_multi
    targets_path = args.targets_dir / args.targets_file

    if args.features_file is not None:
        print(f"Reading single features table: {args.features_file}")
        features_merged = _read_table_with_timestamp(args.features_file)
        print(f"Single features table shape: {features_merged.shape}")
    else:
        print(f"Reading features 1H: {features_1h_path}")
        print(f"Reading features multi: {features_multi_path}")
        # Step 1: merge features only
        features_merged = merge_features_only(
            features_1h_path=features_1h_path,
            features_multi_path=features_multi_path,
            how='inner',
        )
        print(f"Features merged shape (pre-clean): {features_merged.shape}")
    print(f"Reading targets: {targets_path}")

    # Step 2: clean features
    features_to_remove: List[str] = []
    if not args.no_default_feature_drop:
        features_to_remove.extend(DEFAULT_FEATURES_TO_REMOVE)
    if args.extra_feature_drop:
        features_to_remove.extend(args.extra_feature_drop)

    features_clean = _apply_cleanup(
        features_merged,
        warmup_rows=args.warmup_rows,
        features_to_remove=features_to_remove,
        drop_remaining_na=not args.no_drop_na,
    )

    print(f"Features merged shape (post-clean): {features_clean.shape}")

    # Step 3: prepare targets and final merge
    targets_prepared = prepare_targets(targets_path=targets_path, features_for_alignment=features_clean)
    final_merged = pd.merge(
        features_clean,
        targets_prepared,
        on='timestamp',
        how='inner',
        validate='one_to_one',
    ).sort_values('timestamp')
    print(f"Final merge (features_clean + targets): rows={len(final_merged)}, cols={final_merged.shape[1]}")

    out_path: Path
    if args.output is None:
        out_path = args.training_dir / args.dataset / args.output_name
        print(f"Output (default): training_dir={args.training_dir} dataset='{args.dataset}' file='{args.output_name}'")
    else:
        out_path = Path(args.output)
        print(f"Output (explicit): {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lower = str(out_path).lower()
    if args.parquet or lower.endswith('.parquet') or lower.endswith('.pq'):
        if not lower.endswith('.parquet') and not lower.endswith('.pq'):
            out_path = out_path.with_suffix('.parquet')
        final_merged.to_parquet(out_path)
    else:
        if not lower.endswith('.csv'):
            out_path = out_path.with_suffix('.csv')
        final_merged.to_csv(out_path, index=False)

    print(f"Wrote merged dataset: {out_path}")


if __name__ == '__main__':
    main()
