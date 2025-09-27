"""
E2E tests for higher timeframes (4H/12H/1D) using two processes:

1) On-the-fly resampling from 1H OHLCV: pick a random timestamp from the
   reference CSV where the TF v1 columns are finite; compute v1 features from
   the raw OHLCV via right-closed resampling and compare values.

2) Precomputed lookbacks: load TF lookback PKLs from filesystem and compute v1
   features for all timestamps; compare with overlapping timestamps in the
   reference CSV.

Skips gracefully if required files are missing.
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feature_engineering.utils import resample_ohlcv_right_closed
from feature_engineering.hmm_features import HMMFeatureConfig, latest_hmm_observation_from_lookbacks
from feature_engineering.current_bar_features import compute_current_bar_features_from_lookback
from feature_engineering.multi_timeframe_features import calculate_parkinson_volatility


REF_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v1_features.csv")
DATA_PATH = Path("data/BINANCE_BTCUSDT.P, 60.csv")
LOOKBACKS_BASE = Path("/Volumes/Extreme SSD/trading_data/cex/lookbacks")
DATASET = "BINANCE_BTCUSDT.P, 60"


def _read_ref(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return df.sort_values('timestamp').reset_index(drop=True)


def _read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={'time': 'timestamp', 'Volume': 'volume'})
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for c in keep:
        if c not in df.columns:
            raise AssertionError(f"Data CSV missing required column: {c}")
    return df[keep].dropna().sort_values('timestamp').reset_index(drop=True)


def _pick_tf_present(ref: pd.DataFrame, tfs=("4H","12H","1D")) -> str:
    for tf in tfs:
        cols = [f"close_logret_current_{tf}", f"log_volume_delta_current_{tf}", f"close_parkinson_20_{tf}"]
        if all(c in ref.columns for c in cols):
            return tf
    raise AssertionError("No high timeframe v1 columns found in reference CSV")


def _pick_random_ts(ref: pd.DataFrame, tf: str) -> pd.Timestamp:
    cols = [f"close_logret_current_{tf}", f"log_volume_delta_current_{tf}", f"close_parkinson_20_{tf}"]
    cand = ref.dropna(subset=cols)
    if cand.empty:
        raise AssertionError(f"No non-NaN rows for TF={tf} in reference CSV")
    i = random.randrange(len(cand))
    return pd.Timestamp(cand.iloc[i]['timestamp'])


def _load_tf_store(tf: str) -> dict:
    pkl = LOOKBACKS_BASE / DATASET / f"lookbacks_{tf}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(str(pkl))
    return pd.read_pickle(pkl)


def run_tests():
    if not REF_PATH.exists():
        print(f"SKIP: reference file not found: {REF_PATH}")
        return
    if not DATA_PATH.exists():
        print(f"SKIP: data file not found: {DATA_PATH}")
        return

    ref = _read_ref(REF_PATH)
    raw = _read_ohlcv(DATA_PATH)
    tf = _pick_tf_present(ref)
    print("Selected TF:", tf)

    # ---------- Process 1: on-the-fly resampling for one random timestamp ----------
    ts = _pick_random_ts(ref, tf)
    print("Random test timestamp:", ts)
    # Slice raw up to ts, resample to TF
    df_upto = raw[raw['timestamp'] <= ts].set_index('timestamp')
    tf_df = resample_ohlcv_right_closed(df_upto, tf)
    # Current-bar features and Parkinson
    cb = compute_current_bar_features_from_lookback(tf_df, timeframe_suffix=tf)
    pv = calculate_parkinson_volatility(tf_df['high'], tf_df['low'], 20, 'close')["close_parkinson_20"]
    row_computed = {
        f"close_logret_current_{tf}": cb.get(f"close_logret_current_{tf}"),
        f"log_volume_delta_current_{tf}": cb.get(f"log_volume_delta_current_{tf}"),
        f"close_parkinson_20_{tf}": pv,
    }
    # Reference row
    row_ref = ref.loc[ref['timestamp'] == ts, [
        f"close_logret_current_{tf}", f"log_volume_delta_current_{tf}", f"close_parkinson_20_{tf}"
    ]].iloc[0]
    tol = 1e-10
    for k, v in row_computed.items():
        rv = float(row_ref[k])
        if np.isnan(rv):
            # if ref is NaN (e.g., insufficient TF bars), allow NaN
            assert np.isnan(v)
        else:
            assert abs(v - rv) < tol, f"Mismatch {k}: got {v} vs ref {rv} at {ts}"
    print("Process 1 passed.")

    # ---------- Process 2: precomputed lookbacks for all timestamps ----------
    try:
        store = _load_tf_store(tf)
    except FileNotFoundError as e:
        print(f"SKIP: TF store not found for {tf}: {e}")
        return

    base_index = [pd.Timestamp(x) for x in store.get('base_index', [])]
    rows_map = store.get('rows', {})
    if not base_index or not rows_map:
        print(f"SKIP: Empty store for TF={tf}")
        return

    cfg = HMMFeatureConfig(timeframes=[tf], parkinson_window=20)
    records = []
    for ts_i in base_index:
        ts_key = pd.Timestamp(ts_i).strftime('%Y%m%d_%H%M%S')
        lb = rows_map.get(ts_key)
        if lb is None or lb.empty:
            continue
        row, _, ordered = latest_hmm_observation_from_lookbacks({tf: lb}, scaler=None, config=cfg)
        rec = {c: float(row[c]) if pd.notna(row[c]) else np.nan for c in ordered}
        tsi = pd.Timestamp(ts_i)
        # normalize to UTC-naive to match reference
        if tsi.tzinfo is not None:
            tsi = tsi.tz_convert('UTC').tz_localize(None)
        rec['timestamp'] = tsi
        records.append(rec)
    pred = pd.DataFrame(records).sort_values('timestamp')
    if pred.empty:
        print("SKIP: No predictions built from store")
        return

    # Merge with reference and compare on overlap
    cols = [f"close_logret_current_{tf}", f"log_volume_delta_current_{tf}", f"close_parkinson_20_{tf}"]
    merged = pd.merge(
        ref[['timestamp'] + cols],
        pred[['timestamp'] + cols],
        on='timestamp', how='inner', validate='one_to_one', suffixes=('_ref','_pred')
    )
    if merged.empty:
        print("SKIP: No overlapping timestamps between reference and store predictions")
        return

    mismatches = {}
    for c in cols:
        diff = (merged[f"{c}_ref"] - merged[f"{c}_pred"]).abs()
        # Treat both NaN as match
        both_nan = merged[f"{c}_ref"].isna() & merged[f"{c}_pred"].isna()
        bad = (diff > 1e-10) & (~both_nan)
        mismatches[c] = int(bad.sum())

    print("Process 2 compared rows:", len(merged))
    print("mismatches:", mismatches)
    assert all(v == 0 for v in mismatches.values()), f"Mismatches found: {mismatches}"
    print("Process 2 passed.")
    print("High-TF e2e tests passed.")


if __name__ == '__main__':
    run_tests()
