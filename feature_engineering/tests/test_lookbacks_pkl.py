"""
Test script to validate per-timeframe lookback PKL files produced by store_lookbacks.py

Checks (per timeframe):
- Required keys present
- base_index length matches original data length
- For sampled rows, stored lookback equals recomputed lookback using
  get_lookback_window() and resample_ohlcv_right_closed()
"""

import os
import sys
import random
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_lookback_window, resample_ohlcv_right_closed


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    # try 'time' or 'timestamp'
    for col in ('time', 'timestamp'):
        if col in df.columns:
            df = df.set_index(pd.to_datetime(df[col], errors='coerce')).drop(columns=[col])
            return df
    # fallback: first column
    idx = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.set_index(idx).drop(columns=[df.columns[0]])
    return df


def load_input(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv', '.txt'):
        df = pd.read_csv(path)
    elif ext in ('.parquet', '.pq'):
        df = pd.read_parquet(path)
    elif ext in ('.pkl', '.pickle'):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f'Unsupported input format: {ext}')
    df = ensure_datetime_index(df)
    df = df.sort_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def ts_key(ts: pd.Timestamp) -> str:
    return ts.strftime('%Y%m%d_%H%M%S')


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a is None or b is None:
        return (a is None) and (b is None)
    if a.empty and b.empty:
        return True
    if a.shape != b.shape:
        return False
    if list(a.columns) != list(b.columns):
        return False
    # Align index types
    try:
        a_index = pd.to_datetime(a.index)
        b_index = pd.to_datetime(b.index)
    except Exception:
        a_index = a.index
        b_index = b.index
    if not a_index.equals(b_index):
        return False
    return a.equals(b)


def main():
    # Defaults for convenience; override via env vars if needed
    base_dir = os.environ.get('LOOKBACK_BASE_DIR', '/Volumes/Extreme SSD/trading_data/cex/lookbacks')
    dataset = os.environ.get('LOOKBACK_DATASET', 'BINANCE_BTCUSDT.P, 60')
    input_path = os.environ.get('LOOKBACK_INPUT', '../data/BINANCE_BTCUSDT.P, 60.csv')

    print('='*60)
    print('Loading input...')
    df = load_input(input_path)
    n = len(df)
    print(f'Rows: {n}; Columns: {list(df.columns)}')

    folder = os.path.join(base_dir, dataset)
    timeframes = ['1H', '4H', '12H', '1D']
    all_ok = True

    for tf in timeframes:
        print('\n' + '='*60)
        pkl = os.path.join(folder, f'lookbacks_{tf}.pkl')
        if not os.path.exists(pkl):
            print(f'MISSING: {pkl}')
            all_ok = False
            continue
        store = pd.read_pickle(pkl)
        print(f'Loaded {pkl}')

        # Basic structure checks
        expected_keys = {'timeframe', 'base_index', 'lookback_base_rows', 'rows'}
        missing = expected_keys - set(store.keys())
        if missing:
            print(f'ERROR: missing keys {missing}')
            all_ok = False
            continue
        if store['timeframe'].upper() != tf:
            print(f"ERROR: timeframe mismatch: {store['timeframe']} vs {tf}")
            all_ok = False
        if len(store['base_index']) != n:
            print(f"ERROR: base_index length mismatch: {len(store['base_index'])} vs {n}")
            all_ok = False
        if len(store['rows']) != n:
            print(f"ERROR: rows mapping length mismatch: {len(store['rows'])} vs {n}")
            all_ok = False

        # Sample correctness checks
        LBR = store['lookback_base_rows']
        sample_idxs = [0, min(LBR-1, n-1), min(LBR, n-1), n//2, n-1]
        # Add a few random indices beyond warmup
        rng = random.Random(0)
        for _ in range(3):
            sample_idxs.append(rng.randrange(0, n))
        checked = 0
        for i in sorted(set(x for x in sample_idxs if 0 <= x < n)):
            ts = store['base_index'][i]
            key = ts_key(pd.Timestamp(ts))
            stored_df = store['rows'].get(key)
            if stored_df is None:
                print(f'ERROR: missing stored lookback for {tf} at idx={i}, ts={ts}')
                all_ok = False
                continue
            base_lb = get_lookback_window(df, i, LBR)
            if tf in ('1H', 'H', '60T'):
                expected = base_lb
            else:
                expected = resample_ohlcv_right_closed(base_lb, tf)
            ok = compare_frames(stored_df, expected)
            print(f'Check {tf} idx={i} ts={key}:', ok, f'(stored rows={len(stored_df)})')
            checked += 1
            if not ok:
                all_ok = False

    print('\n' + '='*60)
    print('Overall OK:', all_ok)


if __name__ == '__main__':
    main()





