"""
Persist per-timeframe lookback data in a single PKL per timeframe.

Each timeframe PKL stores a dictionary with:
  - 'timeframe': the timeframe string (e.g., '4H')
  - 'base_index': original timestamps (DatetimeIndex) of the input OHLCV
  - 'lookback_base_rows': number of base rows used for each row's lookback window
  - 'rows': dict mapping row timestamp string -> lookback DataFrame for that timeframe

This script logs progress and timings (rows processed, avg rows/sec) per timeframe
and overall to help estimate throughput for large datasets.

Usage:
  python feature_engineering/build_lookbacks.py --input data/ohlcv.csv --output data/lookbacks --timeframes 1H 4H 12H 1D
"""

import os
import argparse
import numpy as np
import pandas as pd
import time

from utils import validate_ohlcv_data, resample_ohlcv_right_closed, get_lookback_window


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if 'timestamp' in df.columns:
        df = df.set_index(pd.to_datetime(df['timestamp'], errors='coerce'))
        df = df.drop(columns=['timestamp'])
        return df
    # Fallback: try first column
    first_col = df.columns[0]
    try:
        idx = pd.to_datetime(df[first_col], errors='coerce')
        if idx.notna().all():
            df = df.set_index(idx).drop(columns=[first_col])
            return df
    except Exception:
        pass
    raise ValueError("Input must have a DatetimeIndex or a 'timestamp' column")


def _load_ohlcv(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv', '.txt'):
        df = pd.read_csv(path)
    elif ext in ('.parquet', '.pq'):
        df = pd.read_parquet(path)
    elif ext in ('.pkl', '.pickle'):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported input format: {ext}")
    df = _ensure_datetime_index(df)
    df = df.sort_index()
    # Standardize column names (common vendor variations)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Some vendors use 'volume' capitalized; lower-casing handles it
    if not validate_ohlcv_data(df):
        raise ValueError("Invalid OHLCV: require columns open, high, low, close, volume and non-empty data")
    return df


def _build_timeframe_store(df: pd.DataFrame, timeframe: str, lookback_base_rows: int) -> dict:
    tf = timeframe.upper()
    base_idx = df.index
    rows_map: dict[str, pd.DataFrame] = {}
    total = len(base_idx)
    start_time = time.time()
    last_log = start_time
    for i, ts in enumerate(base_idx):
        lb_base = get_lookback_window(df, i, lookback_base_rows)
        if tf in ("1H", "H", "60T"):
            lb_tf = lb_base
        else:
            lb_tf = resample_ohlcv_right_closed(lb_base, tf)
        # Store only if non-empty; still record empty for consistency
        rows_map[ts.strftime('%Y%m%d_%H%M%S')] = lb_tf
        if (i + 1) % 1000 == 0 or i == total - 1:
            now = time.time()
            elapsed = now - start_time
            batch_elapsed = now - last_log
            avg_rps = (i + 1) / elapsed if elapsed > 0 else float('inf')
            batch_rps = (min(1000, i + 1) / batch_elapsed) if batch_elapsed > 0 else float('inf')
            print(f"  {tf}: processed {i+1}/{total} | avg {avg_rps:.2f} rows/s | last {batch_rps:.2f} rows/s")
            last_log = now
    return {
        'timeframe': tf,
        'base_index': base_idx,
        'lookback_base_rows': int(lookback_base_rows),
        'rows': rows_map,
    }


def process_and_store(input_path: str, output_dir: str, timeframes: list[str], lookback: int) -> None:
    df = _load_ohlcv(input_path)
    total_rows = len(df)
    print(f"Rows: {total_rows}; timeframes: {timeframes}; lookback: {lookback}")
    # Derive subfolder from input file name (without extension)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    final_out_dir = os.path.join(output_dir, base_name)
    os.makedirs(final_out_dir, exist_ok=True)

    overall_start = time.time()
    total_processed = 0
    for tf in timeframes:
        print(f"Building timeframe store: {tf} ...")
        tf_start = time.time()
        store = _build_timeframe_store(df, tf, lookback)
        out_path = os.path.join(final_out_dir, f"lookbacks_{tf}.pkl")
        pd.to_pickle(store, out_path)
        tf_elapsed = time.time() - tf_start
        rows_tf = len(store['rows'])
        total_processed += rows_tf
        rps = rows_tf / tf_elapsed if tf_elapsed > 0 else float('inf')
        print(f"Wrote: {out_path} (rows={rows_tf}) | time {tf_elapsed:.2f}s | rate {rps:.2f} rows/s")

    overall_elapsed = time.time() - overall_start
    overall_rps = total_processed / overall_elapsed if overall_elapsed > 0 else float('inf')
    print(f"All timeframes done: rows={total_processed} across {len(timeframes)} TFs | total {overall_elapsed:.2f}s | avg {overall_rps:.2f} rows/s")


def main():
    parser = argparse.ArgumentParser(description="Store per-timeframe lookback data as single PKL files")
    parser.add_argument('--input', required=True, help='Path to input OHLCV file (csv/parquet/pkl)')
    parser.add_argument('--output', default='/Volumes/Extreme SSD/trading_data/cex/lookbacks', help='Base output directory (default: /Volumes/Extreme SSD/trading_data/cex/lookbacks)')
    parser.add_argument('--timeframes', nargs='+', default=['1H', '4H', '12H', '1D'], help='Timeframes to generate')
    parser.add_argument('--lookback', type=int, default=168, help='Lookback window (bars in each timeframe)')
    args = parser.parse_args()

    process_and_store(args.input, args.output, args.timeframes, args.lookback)


if __name__ == '__main__':
    main()

