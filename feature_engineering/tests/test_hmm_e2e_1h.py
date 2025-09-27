"""
E2E test for hmm_features v1 on real data.

Compares 1H v1 features computed from data/BINANCE_BTCUSDT.P, 60.csv
against the reference CSV produced by the builder script:
  /Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v1_features.csv

Skips gracefully if the reference file is not present.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feature_engineering.hmm_features import HMMFeatureConfig, build_hmm_observations_1h


REF_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v1_features.csv")
DATA_PATH = Path("data/BINANCE_BTCUSDT.P, 60.csv")


def _read_ref(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize timestamp to UTC-naive
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    # Keep only 1H v1 columns
    cols = [
        'timestamp',
        'close_logret_current_1H',
        'log_volume_delta_current_1H',
        'close_parkinson_20_1H',
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Reference missing expected columns: {missing}")
    return df[cols].sort_values('timestamp').reset_index(drop=True)


def _read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Source uses 'time' and 'Volume' columns among others
    if 'time' not in df.columns:
        raise AssertionError("Data CSV missing 'time' column")
    df = df.rename(columns={'time': 'timestamp', 'Volume': 'volume'})
    # Normalize timestamp to UTC-naive
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    # Keep only needed columns but preserve timestamp
    keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for c in keep:
        if c not in df.columns:
            raise AssertionError(f"Data CSV missing required column: {c}")
    return df[keep].dropna().sort_values('timestamp').reset_index(drop=True)


def run_tests():
    if not REF_PATH.exists():
        print(f"SKIP: reference file not found: {REF_PATH}")
        return
    if not DATA_PATH.exists():
        print(f"SKIP: data file not found: {DATA_PATH}")
        return

    ref = _read_ref(REF_PATH)
    raw = _read_ohlcv(DATA_PATH)

    # Build observations from 1H OHLCV
    cfg = HMMFeatureConfig(timeframes=['1H'], parkinson_window=20, drop_warmup=True)
    obs, meta = build_hmm_observations_1h(raw, cfg)

    # Align on intersection of timestamps
    cols = ['close_logret_current_1H', 'log_volume_delta_current_1H', 'close_parkinson_20_1H']
    merged = pd.merge(
        ref, obs[['timestamp'] + cols],
        on='timestamp', how='inner', validate='one_to_one', suffixes=('_ref', '_hmm')
    )
    assert len(merged) > 0, "No overlapping timestamps between reference and computed features"

    # Compare each feature within tight tolerance
    tol = 1e-10
    mismatches = {}
    for c in cols:
        diff = (merged[f"{c}_ref"] - merged[f"{c}_hmm"]).abs()
        bad = diff > tol
        mismatches[c] = int(bad.sum())

    print("rows compared:", len(merged))
    print("mismatches (counts per column):", mismatches)
    assert all(v == 0 for v in mismatches.values()), f"Mismatches found: {mismatches}"

    print("hmm_features e2e (1H v1) test passed.")


if __name__ == '__main__':
    run_tests()

