"""
HMM feature engineering utilities (compact, stationary observations).

Primary goal: build a small multivariate observation vector per bar, suitable
for Gaussian HMM regime detection, while reusing existing feature code.

Default v1 observation set per timeframe TF (e.g., 1H):
  - close_logret_current_TF          (from current_bar_features)
  - log_volume_delta_current_TF      (from current_bar_features)
  - close_parkinson_20_TF            (vectorized Parkinson(20))

Provides batch builders for 1H OHLCV and a latest-row builder from lookbacks
(one or more timeframes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .current_bar_features import (
    compute_current_bar_features,
    compute_current_bar_features_from_lookback,
)
from .multi_timeframe_features import calculate_parkinson_volatility


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class HMMFeatureConfig:
    """Configuration for building HMM observations.

    Attributes:
      timeframes: List of timeframe suffixes to include (training typically ['1H']).
      parkinson_window: Rolling window for Parkinson volatility.
      drop_warmup: Drop initial rows where any selected feature is NaN.
    """

    timeframes: List[str] = field(default_factory=lambda: ["1H"])
    parkinson_window: int = 20
    drop_warmup: bool = True

    def v1_columns(self) -> List[str]:
        cols: List[str] = [
            *(f"close_logret_current_{tf}" for tf in self.timeframes),
            *(f"log_volume_delta_current_{tf}" for tf in self.timeframes),
            *(f"close_parkinson_{self.parkinson_window}_{tf}" for tf in self.timeframes),
        ]
        return ["timestamp", *cols]


# -----------------------------------------------------------------------------
# Vectorized helpers (1H batch build)
# -----------------------------------------------------------------------------


def _vectorized_parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Vectorized Parkinson volatility per bar using the canonical formula.

    close_parkinson_{window} = sqrt((1 / (4 * ln(2))) * mean( (ln(High/Low))^2, last window ))
    """
    if high is None or low is None:
        return pd.Series(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl2 = np.log(high / low) ** 2
    ma = log_hl2.rolling(window=window, min_periods=window).mean()
    factor = 1.0 / (4.0 * np.log(2.0))
    out = np.sqrt(factor * ma)
    return out.astype(float)


def build_hmm_observations_1h(
    ohlcv_1h: pd.DataFrame,
    config: Optional[HMMFeatureConfig] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Build vectorized HMM observations for 1H OHLCV data.

    Returns a DataFrame with columns matching config.v1_columns() for 1H only.
    If config.timeframes includes multiple TFs, only '1H' is produced here.
    For multi-TF at inference, use latest_hmm_observation_from_lookbacks.
    """
    if config is None:
        config = HMMFeatureConfig()

    if ohlcv_1h is None or len(ohlcv_1h) == 0:
        raise ValueError("ohlcv_1h is empty")

    # Ensure index and required columns
    df = ohlcv_1h.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    # Normalize column names
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in ohlcv_1h")

    # Current-bar features (vectorized) for 1H
    cb = compute_current_bar_features(df, timeframe_suffix='1H', include_original=False)

    # Parkinson(20) (vectorized)
    parkinson_series = _vectorized_parkinson(df['high'], df['low'], config.parkinson_window)
    parkinson_name = f"close_parkinson_{config.parkinson_window}_1H"
    parkinson_series = parkinson_series.rename(parkinson_name)

    # Assemble
    out = pd.concat([
        cb[['close_logret_current_1H', 'log_volume_delta_current_1H']],
        parkinson_series,
    ], axis=1)
    out = out.reset_index().rename(columns={'index': 'timestamp'})

    # Optionally drop warmup rows where any feature is NaN
    dropped_rows = 0
    if config.drop_warmup:
        before = len(out)
        out = out.dropna(subset=[c for c in out.columns if c != 'timestamp'])
        dropped_rows = before - len(out)

    # Enforce column order (timestamp first)
    desired_cols = ['timestamp', 'close_logret_current_1H', 'log_volume_delta_current_1H', parkinson_name]
    existing = [c for c in desired_cols if c in out.columns]
    out = out[existing]

    meta = {
        'timeframes': ['1H'],
        'parkinson_window': config.parkinson_window,
        'dropped_rows': dropped_rows,
        'feature_columns': existing[1:],
    }
    return out, meta


# -----------------------------------------------------------------------------
# Scaling
# -----------------------------------------------------------------------------


def scale_observations(
    obs_df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    *,
    copy: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Fit/apply StandardScaler on observation columns (excludes 'timestamp')."""
    if obs_df is None or obs_df.empty:
        raise ValueError('obs_df is empty')
    cols = [c for c in obs_df.columns if c != 'timestamp']
    X = obs_df[cols].values
    if scaler is None:
        scaler = StandardScaler(copy=copy)
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)
    df_scaled = obs_df.copy()
    df_scaled[cols] = Xs
    return df_scaled, scaler


# -----------------------------------------------------------------------------
# Latest-row builder from lookbacks (supports multiple TFs)
# -----------------------------------------------------------------------------


def _select_v1_from_current_bar_map(feat_map: Dict[str, float], tf: str) -> Dict[str, float]:
    keys = [
        f"close_logret_current_{tf}",
        f"log_volume_delta_current_{tf}",
    ]
    return {k: feat_map.get(k, np.nan) for k in keys}


def latest_hmm_observation_from_lookbacks(
    lookbacks_by_tf: Dict[str, pd.DataFrame],
    scaler: Optional[StandardScaler],
    config: Optional[HMMFeatureConfig] = None,
) -> Tuple[pd.Series, pd.Timestamp, List[str]]:
    """Compute the v1 observation at the latest timestamp from per-TF lookbacks.

    - Uses current-bar features for log return and log volume delta.
    - Uses calculate_parkinson_volatility on each TF lookback with config.parkinson_window.
    - Applies an optional prefit StandardScaler (if provided).

    Returns: (row_series, timestamp, ordered_feature_columns)
    """
    if config is None:
        config = HMMFeatureConfig()

    if not lookbacks_by_tf:
        raise ValueError('lookbacks_by_tf is empty')

    # Determine timestamp: prefer 1H if present, else max last index
    if '1H' in lookbacks_by_tf and lookbacks_by_tf['1H'] is not None and not lookbacks_by_tf['1H'].empty:
        ts = pd.Timestamp(lookbacks_by_tf['1H'].index[-1])
    else:
        ts = max(pd.Timestamp(df.index[-1]) for df in lookbacks_by_tf.values() if df is not None and not df.empty)

    # Build per-TF features
    feat_values: Dict[str, float] = {}
    selected_tfs: List[str] = []
    for tf in config.timeframes:
        if tf not in lookbacks_by_tf:
            continue
        lb = lookbacks_by_tf[tf]
        if lb is None or lb.empty:
            continue
        selected_tfs.append(tf)

        # Current-bar map (t, t-1 in that TF)
        cb_map = compute_current_bar_features_from_lookback(lb, timeframe_suffix=tf)
        feat_values.update(_select_v1_from_current_bar_map(cb_map, tf))

        # Parkinson (window W over that TF lookback)
        pv = calculate_parkinson_volatility(lb['high'], lb['low'], window=config.parkinson_window, column_name='close')
        key = f"close_parkinson_{config.parkinson_window}_{tf}"
        feat_values[key] = pv.get(f"close_parkinson_{config.parkinson_window}", np.nan)

    # Order columns
    ordered_cols = [
        *(f"close_logret_current_{tf}" for tf in selected_tfs),
        *(f"log_volume_delta_current_{tf}" for tf in selected_tfs),
        *(f"close_parkinson_{config.parkinson_window}_{tf}" for tf in selected_tfs),
    ]
    row = pd.Series([feat_values.get(c, np.nan) for c in ordered_cols], index=ordered_cols, dtype=float)

    # Scale if provided
    if scaler is not None:
        arr = scaler.transform(row.values.reshape(1, -1))[0]
        row = pd.Series(arr, index=ordered_cols)

    return row, ts, ordered_cols


__all__ = [
    'HMMFeatureConfig',
    'build_hmm_observations_1h',
    'scale_observations',
    'latest_hmm_observation_from_lookbacks',
]

