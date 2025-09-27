"""
Unit tests for feature_engineering/hmm_features.py

Focus: individual functions — vectorized 1H builder, scaler, and latest-row
builder from lookbacks. Uses small synthetic OHLCV samples.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feature_engineering.hmm_features import (
    HMMFeatureConfig,
    build_hmm_observations_1h,
    scale_observations,
    latest_hmm_observation_from_lookbacks,
    _vectorized_parkinson,
)
from feature_engineering.multi_timeframe_features import calculate_parkinson_volatility
from feature_engineering.utils import resample_ohlcv_right_closed
from feature_engineering.current_bar_features import compute_current_bar_features_from_lookback


def _mk_ohlcv(n: int, start: str = "2025-01-01 00:00:00", freq: str = "1H") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq=freq)
    # Construct simple but non-degenerate series
    base = np.linspace(100.0, 120.0, n)
    close = base + np.sin(np.linspace(0, 4 * np.pi, n))
    open_ = close - np.random.default_rng(0).normal(0.2, 0.05, n)
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 1000 + (np.arange(n) % 10) * 50
    return pd.DataFrame({
        'open': open_.astype(float),
        'high': high.astype(float),
        'low': low.astype(float),
        'close': close.astype(float),
        'volume': volume.astype(float),
    }, index=idx)


def run_tests():
    print("\n== hmm_features: build_hmm_observations_1h / vectorized Parkinson ==")
    df = _mk_ohlcv(30)
    cfg = HMMFeatureConfig(timeframes=['1H'], parkinson_window=20, drop_warmup=True)
    obs, meta = build_hmm_observations_1h(df, cfg)
    print("Columns:", list(obs.columns))
    assert list(obs.columns) == ['timestamp', 'close_logret_current_1H', 'log_volume_delta_current_1H', 'close_parkinson_20_1H']
    # Warmup drop: first 19 rows (requires 20-window)
    assert len(obs) == len(df) - 19
    assert not obs[['close_logret_current_1H', 'log_volume_delta_current_1H', 'close_parkinson_20_1H']].isna().any().any()

    # Vectorized vs scalar Parkinson at last row
    parc_vec = _vectorized_parkinson(df['high'], df['low'], 20).iloc[-1]
    parc_scalar = calculate_parkinson_volatility(df['high'], df['low'], 20, 'close')["close_parkinson_20"]
    assert np.isfinite(parc_vec) and np.isfinite(parc_scalar)
    assert abs(parc_vec - parc_scalar) < 1e-12

    print("\n== hmm_features: scale_observations ==")
    obs_scaled, scaler = scale_observations(obs)
    cols = ['close_logret_current_1H', 'log_volume_delta_current_1H', 'close_parkinson_20_1H']
    means = obs_scaled[cols].mean().abs().values
    stds = obs_scaled[cols].std(ddof=0).values
    # Means ~ 0, stds ~ 1
    assert np.all(means < 1e-7)
    assert np.all(np.abs(stds - 1.0) < 1e-6)

    print("\n== hmm_features: latest_hmm_observation_from_lookbacks (1H only) ==")
    # Use last 30 rows as lookback
    lb_1h = df.tail(30)
    lookbacks = {'1H': lb_1h}
    cfg2 = HMMFeatureConfig(timeframes=['1H'], parkinson_window=20)
    row, ts, ordered = latest_hmm_observation_from_lookbacks(lookbacks, scaler=None, config=cfg2)
    assert ts == lb_1h.index[-1]
    assert ordered == ['close_logret_current_1H', 'log_volume_delta_current_1H', 'close_parkinson_20_1H']
    # Expected current-bar from lookback map
    cb_map = compute_current_bar_features_from_lookback(lb_1h, timeframe_suffix='1H')
    assert abs(row['close_logret_current_1H'] - cb_map['close_logret_current_1H']) < 1e-12
    assert abs(row['log_volume_delta_current_1H'] - cb_map['log_volume_delta_current_1H']) < 1e-12
    # Expected Parkinson via scalar function
    parc = calculate_parkinson_volatility(lb_1h['high'], lb_1h['low'], 20, 'close')["close_parkinson_20"]
    assert abs(row['close_parkinson_20_1H'] - parc) < 1e-12

    print("\n== hmm_features: latest_hmm_observation_from_lookbacks (1H + 4H) ==")
    lb_4h = resample_ohlcv_right_closed(lb_1h, '4H')
    lookbacks2 = {'1H': lb_1h, '4H': lb_4h}
    cfg3 = HMMFeatureConfig(timeframes=['1H', '4H'], parkinson_window=20)
    row2, ts2, ordered2 = latest_hmm_observation_from_lookbacks(lookbacks2, scaler=None, config=cfg3)
    assert ts2 == lb_1h.index[-1]
    assert ordered2 == [
        'close_logret_current_1H', 'close_logret_current_4H',
        'log_volume_delta_current_1H', 'log_volume_delta_current_4H',
        'close_parkinson_20_1H', 'close_parkinson_20_4H',
    ]
    # Validate 4H components
    cb_map_4h = compute_current_bar_features_from_lookback(lb_4h, timeframe_suffix='4H')
    assert abs(row2['close_logret_current_4H'] - cb_map_4h['close_logret_current_4H']) < 1e-12
    assert abs(row2['log_volume_delta_current_4H'] - cb_map_4h['log_volume_delta_current_4H']) < 1e-12
    parc_4h = calculate_parkinson_volatility(lb_4h['high'], lb_4h['low'], 20, 'close')["close_parkinson_20"]
    # If the 4H lookback has fewer than 20 bars, both should be NaN; otherwise compare numerically
    if np.isnan(parc_4h):
        assert np.isnan(row2['close_parkinson_20_4H'])
    else:
        assert abs(row2['close_parkinson_20_4H'] - parc_4h) < 1e-10

    print("All hmm_features tests passed.")


if __name__ == "__main__":
    run_tests()
