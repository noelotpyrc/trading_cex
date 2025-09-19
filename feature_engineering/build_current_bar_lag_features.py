"""
Run script: build Current Bar Features and their lag-N versions from saved lookbacks
and (optionally) original 1H OHLCV.

Usage examples:

1) List datasets found under the lookbacks directory:
   python feature_engineering/build_current_bar_lag_features.py --list \
     --lookbacks-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks"

2) Build features for a dataset using saved lookbacks only (1H reconstructed
   from lookbacks_1H.pkl):
   python feature_engineering/build_current_bar_lag_features.py \
     --dataset "BINANCE_BTCUSDT.P, 60" \
     --lookbacks-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks" \
     --timeframes 1H 4H 12H 1D \
     --max-lag 3 \
     --output "/Volumes/Extreme SSD/trading_data/cex/features/current_bar_with_lags.parquet"

3) Same as (2) but use an original 1H OHLCV file for the 1H path:
   python feature_engineering/build_current_bar_lag_features.py \
     --dataset "BINANCE_BTCUSDT.P, 60" \
     --lookbacks-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks" \
     --ohlcv-1h "/path/to/original_1h.csv" \
     --timeframes 1H 4H 12H 1D \
     --max-lag 3 \
     --output features.parquet
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure project root on sys.path for package imports when run as a script
import sys
from datetime import datetime

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.append(_PROJ_ROOT)

from feature_engineering.current_bar_features import (
    compute_current_bar_features,
    compute_current_and_lag_features_from_lookback,
    _rename_for_lag,
)


def list_datasets(lookbacks_dir: str) -> List[str]:
    if not os.path.isdir(lookbacks_dir):
        return []
    entries = []
    for name in os.listdir(lookbacks_dir):
        path = os.path.join(lookbacks_dir, name)
        if os.path.isdir(path):
            # consider as dataset if contains any lookbacks_*.pkl
            has_pkl = any(fn.startswith("lookbacks_") and fn.endswith(".pkl") for fn in os.listdir(path))
            if has_pkl:
                entries.append(name)
    return sorted(entries)


def load_store(lookbacks_dir: str, dataset: str, timeframe: str) -> Dict:
    pkl = os.path.join(lookbacks_dir, dataset, f"lookbacks_{timeframe}.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Missing lookback file: {pkl}")
    return pd.read_pickle(pkl)


def _apply_lag_renames(df: pd.DataFrame, tf: str, lag_k: int) -> pd.DataFrame:
    # Rename columns by inserting lag token before timeframe suffix
    renamed = {}
    suffix = f"_{tf}"
    for col in df.columns:
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            new_base = _rename_for_lag(base, lag_k)
            renamed[col] = f"{new_base}{suffix}"
        else:
            # fallback: apply rename directly
            renamed[col] = _rename_for_lag(col, lag_k)
    return df.rename(columns=renamed)


def build_1h_features(
    ohlcv_1h: pd.DataFrame,
    tf: str,
    max_lag: int,
) -> pd.DataFrame:
    feats = compute_current_bar_features(ohlcv_1h, timeframe_suffix=tf, include_original=False)
    frames = [feats]
    for k in range(1, max_lag + 1):
        lag_df = feats.shift(k)
        lag_df = _apply_lag_renames(lag_df, tf=tf, lag_k=k)
        frames.append(lag_df)
    out = pd.concat(frames, axis=1)
    return out


def reconstruct_1h_from_store(store_1h: Dict) -> pd.DataFrame:
    base_index = store_1h.get("base_index")
    rows_map = store_1h.get("rows", {})
    if base_index is None:
        raise ValueError("1H store missing 'base_index'")
    records: List[dict] = []
    for ts in base_index:
        ts_key = pd.Timestamp(ts).strftime('%Y%m%d_%H%M%S')
        lb = rows_map.get(ts_key)
        if lb is None or len(lb) == 0:
            records.append({"timestamp": pd.Timestamp(ts), "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan, "volume": np.nan})
        else:
            last = lb.iloc[-1]
            records.append({
                "timestamp": pd.Timestamp(ts),
                "open": float(last.get("open", np.nan)),
                "high": float(last.get("high", np.nan)),
                "low": float(last.get("low", np.nan)),
                "close": float(last.get("close", np.nan)),
                "volume": float(last.get("volume", np.nan)),
            })
    df = pd.DataFrame(records).set_index("timestamp")
    return df


def build_higher_tf_features_from_store(store: Dict, tf: str, max_lag: int) -> pd.DataFrame:
    base_index = store.get("base_index")
    rows_map = store.get("rows", {})
    if base_index is None:
        raise ValueError(f"store for {tf} missing 'base_index'")
    rows: List[dict] = []
    for ts in base_index:
        ts_key = pd.Timestamp(ts).strftime('%Y%m%d_%H%M%S')
        lb = rows_map.get(ts_key)
        feats = compute_current_and_lag_features_from_lookback(lb, timeframe_suffix=tf, max_lag=max_lag)
        feats["timestamp"] = pd.Timestamp(ts)
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build current-bar features and lag-N features from lookbacks and 1H OHLCV")
    parser.add_argument("--lookbacks-dir", default="/Volumes/Extreme SSD/trading_data/cex/lookbacks", help="Directory containing lookback stores")
    parser.add_argument("--dataset", help="Dataset folder name within lookbacks dir (e.g., 'BINANCE_BTCUSDT.P, 60')")
    parser.add_argument("--timeframes", nargs="+", default=["1H", "4H", "12H", "1D"], help="Timeframes to process")
    parser.add_argument("--max-lag", type=int, default=3, help="Max lag K to generate (lag-1..lag-K)")
    parser.add_argument("--ohlcv-1h", help="Optional path to original 1H OHLCV (csv/parquet/pkl)")
    parser.add_argument("--output", required=False, help="Output file (.parquet or .csv). If omitted, prints a summary and exits.")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    args = parser.parse_args()

    if args.list or not args.dataset:
        datasets = list_datasets(args.lookbacks_dir)
        if not datasets:
            print(f"No datasets found in: {args.lookbacks_dir}")
            return
        print("Available datasets under lookbacks dir:")
        for name in datasets:
            print(f"- {name}")
        if not args.dataset:
            return

    # Load stores per timeframe
    tfs = [tf.upper() for tf in args.timeframes]
    stores = {}
    for tf in tfs:
        try:
            stores[tf] = load_store(args.lookbacks_dir, args.dataset, tf)
        except FileNotFoundError as e:
            print(str(e))
            stores[tf] = None

    frames = []

    # 1H path
    if "1H" in tfs:
        df1h = None
        if args.ohlcv_1h and os.path.exists(args.ohlcv_1h):
            # lightweight loader (csv/parquet/pkl)
            ext = os.path.splitext(args.ohlcv_1h)[1].lower()
            if ext in (".csv", ".txt"):
                df1h = pd.read_csv(args.ohlcv_1h)
            elif ext in (".parquet", ".pq"):
                df1h = pd.read_parquet(args.ohlcv_1h)
            elif ext in (".pkl", ".pickle"):
                df1h = pd.read_pickle(args.ohlcv_1h)
            else:
                raise ValueError(f"Unsupported 1H input format: {ext}")
            # ensure datetime index
            if not isinstance(df1h.index, pd.DatetimeIndex):
                time_col = None
                for cand in ['timestamp', 'time', 'datetime', 'date']:
                    if cand in df1h.columns:
                        time_col = cand
                        break
                if time_col is None and len(df1h.columns) > 0 and str(df1h.columns[0]).lower() in ('time','timestamp','datetime','date'):
                    time_col = df1h.columns[0]
                if time_col is None:
                    raise ValueError("1H OHLCV must have a DatetimeIndex or a 'timestamp'/'time' column")
                df1h = df1h.set_index(pd.to_datetime(df1h[time_col], errors='coerce'))
                df1h = df1h.drop(columns=[time_col])
            df1h = df1h.sort_index()
            # standardize columns
            df1h.columns = [str(c).lower() for c in df1h.columns]
            missing = [c for c in ['open','high','low','close','volume'] if c not in df1h.columns]
            if missing:
                raise ValueError(f"1H OHLCV missing columns: {missing}")
        else:
            # Fallback: reconstruct 1H OHLCV from lookbacks_1H.pkl
            store_1h = stores.get("1H")
            if store_1h is None:
                raise FileNotFoundError(
                    "1H timeframe requested but neither --ohlcv-1h nor lookbacks_1H.pkl is available."
                )
            df1h = reconstruct_1h_from_store(store_1h)

        feats_1h = build_1h_features(df1h, tf="1H", max_lag=args.max_lag)
        frames.append(feats_1h)

    # Higher timeframes
    for tf in tfs:
        if tf == "1H":
            continue
        store = stores.get(tf)
        if store is None:
            print(f"Skipping timeframe {tf}: store not found")
            continue
        feats_tf = build_higher_tf_features_from_store(store, tf=tf, max_lag=args.max_lag)
        frames.append(feats_tf)

    if not frames:
        print("No features produced.")
        return

    # Outer-join on index
    df_all = frames[0]
    for f in frames[1:]:
        df_all = df_all.join(f, how="outer")

    print(f"Built features: rows={len(df_all)}, cols={len(df_all.columns)}")
    # Compose default output path inside the same lookbacks dataset folder, include TFs and lag count
    tf_part = "-".join(tfs)
    default_name = f"current_bar_with_lags_{tf_part}_lags{args.max_lag}.csv"
    out_path = args.output if args.output else os.path.join(
        args.lookbacks_dir, args.dataset, default_name
    )

    # Never overwrite: if path exists, append a run timestamp before extension
    def _unique_out_path(p: str) -> str:
        if not os.path.exists(p):
            return p
        root, ext = os.path.splitext(p)
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        candidate = f"{root}_{stamp}{ext or '.csv'}"
        if not os.path.exists(candidate):
            return candidate
        i = 1
        while True:
            candidate_i = f"{root}_{stamp}_{i}{ext or '.csv'}"
            if not os.path.exists(candidate_i):
                return candidate_i
            i += 1

    out_path = _unique_out_path(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.lower().endswith('.csv'):
        df_all.to_csv(out_path, index=True)
    else:
        df_all.to_parquet(out_path, index=True)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

