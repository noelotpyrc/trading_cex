"""
Vectorized generators for Current Bar Features (t and t−1 only).

These functions operate on OHLCV lookback DataFrames (DatetimeIndex, columns
include 'open','high','low','close','volume') and return an intermediate
DataFrame with current-bar features computed for every row using only the
current bar t and the immediately previous bar t−1.

Column naming follows docs/feature_engineering.md (Current Bar Features):

- close_logret_current_<TF> = log(C_t / C_{t−1})
- high_low_range_pct_current_<TF> = (H_t − L_t) / O_t
- close_open_pct_current_<TF> = (C_t − O_t) / O_t
- log_volume_<TF> = log1p(V_t)
- log_volume_delta_current_<TF> = log1p(V_t) − log1p(V_{t−1})
- sign_close_logret_current_<TF> = sign(close_logret_current_<TF>)

Interactions (feature1_x_feature2):
- close_logret_current_<TF>_x_log_volume_<TF>
- close_logret_current_<TF>_x_log_volume_delta_current_<TF>
- sign_close_logret_current_<TF>_x_log_volume_<TF>
- high_low_range_pct_current_<TF>_x_log_volume_<TF>
- close_open_pct_current_<TF>_x_log_volume_<TF>

All operations are leakage-safe (use at most t and t−1). Deeper lags should be
produced by shifting the returned table (self-joins) as needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .utils import validate_ohlcv_data


def _ensure_ohlcv_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    # Map any case-variant columns to lower-case canonical names
    rename_map = {}
    for col in df.columns:
        low = str(col).lower()
        if low in {"open", "high", "low", "close", "volume"}:
            rename_map[col] = low
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _append_tf_suffix(columns: list[str], timeframe_suffix: Optional[str]) -> list[str]:
    if timeframe_suffix is None:
        return columns
    suffix = f"_{timeframe_suffix}"
    return [f"{c}{suffix}" for c in columns]


def compute_current_bar_features(
    ohlcv: pd.DataFrame,
    timeframe_suffix: Optional[str] = None,
    include_original: bool = False,
) -> pd.DataFrame:
    """
    Compute Current Bar Features for every row of the given OHLCV DataFrame.

    Parameters
    - ohlcv: DataFrame with columns 'open','high','low','close','volume' and a
      DatetimeIndex (or indexable). Only t and t−1 are used.
    - timeframe_suffix: Optional suffix to append to feature column names, e.g.,
      '1H', '4H', '12H', '1D'. When None, no suffix is appended.
    - include_original: If True, original OHLCV columns are included in the
      returned DataFrame. Otherwise, only the feature columns are returned.

    Returns
    - DataFrame indexed like `ohlcv`, with per-row current-bar features and
      optional original columns.
    """
    if ohlcv is None:
        raise ValueError("ohlcv is None")

    df = _ensure_ohlcv_lowercase(ohlcv.copy())
    if not validate_ohlcv_data(df):
        raise ValueError("Input DataFrame must contain non-empty OHLCV columns: 'open','high','low','close','volume'.")

    # Extract core series
    open_s = pd.to_numeric(df["open"], errors="coerce")
    high_s = pd.to_numeric(df["high"], errors="coerce")
    low_s = pd.to_numeric(df["low"], errors="coerce")
    close_s = pd.to_numeric(df["close"], errors="coerce")
    volume_s = pd.to_numeric(df["volume"], errors="coerce")

    # Close log return (t vs t−1) — log(C_t / C_{t−1})
    prev_close = close_s.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        close_logret = np.where(
            (close_s > 0) & (prev_close > 0),
            np.log(close_s / prev_close),
            np.nan,
        )
    close_logret = pd.Series(close_logret, index=df.index, dtype=float, name="close_logret_current")

    # High–Low range (percent of Open) — (H_t − L_t) / O_t
    denom_open = open_s.replace(0.0, np.nan)
    high_low_range_pct = (high_s - low_s) / denom_open
    high_low_range_pct.name = "high_low_range_pct_current"

    # Close–Open percent (of Open) — (C_t − O_t) / O_t
    close_open_pct = (close_s - open_s) / denom_open
    close_open_pct.name = "close_open_pct_current"

    # Volume transforms
    log_volume = np.log1p(volume_s)
    log_volume.name = "log_volume"
    log_volume_delta = log_volume - log_volume.shift(1)
    log_volume_delta.name = "log_volume_delta_current"

    # Sign of close log return
    sign_close_logret = np.sign(close_logret)
    sign_close_logret.name = "sign_close_logret_current"

    # Interactions
    close_logret_x_log_volume = close_logret * log_volume
    close_logret_x_log_volume.name = "close_logret_current_x_log_volume"

    close_logret_x_log_volume_delta = close_logret * log_volume_delta
    close_logret_x_log_volume_delta.name = "close_logret_current_x_log_volume_delta_current"

    sign_close_logret_x_log_volume = sign_close_logret * log_volume
    sign_close_logret_x_log_volume.name = "sign_close_logret_current_x_log_volume"

    high_low_range_pct_x_log_volume = high_low_range_pct * log_volume
    high_low_range_pct_x_log_volume.name = "high_low_range_pct_current_x_log_volume"

    close_open_pct_x_log_volume = close_open_pct * log_volume
    close_open_pct_x_log_volume.name = "close_open_pct_current_x_log_volume"

    features = pd.concat(
        [
            close_logret,
            high_low_range_pct,
            close_open_pct,
            log_volume,
            log_volume_delta,
            sign_close_logret,
            close_logret_x_log_volume,
            close_logret_x_log_volume_delta,
            sign_close_logret_x_log_volume,
            high_low_range_pct_x_log_volume,
            close_open_pct_x_log_volume,
        ],
        axis=1,
    )

    # Apply timeframe suffix if requested
    if timeframe_suffix is not None:
        features.columns = _append_tf_suffix(list(features.columns), timeframe_suffix)

    if include_original:
        # Preserve original columns first, then add features
        out = df.copy()
        # Avoid accidental overwrite by ensuring no name collisions
        overlap = set(out.columns) & set(features.columns)
        if overlap:
            raise ValueError(f"Column name collision between original and features: {sorted(overlap)}")
        out = pd.concat([out, features], axis=1)
        return out

    return features


def with_current_bar_features(
    ohlcv: pd.DataFrame,
    timeframe_suffix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: return original OHLCV with current-bar features appended.
    """
    return compute_current_bar_features(
        ohlcv=ohlcv,
        timeframe_suffix=timeframe_suffix,
        include_original=True,
    )


def compute_current_bar_features_from_lookback(
    lookback: pd.DataFrame,
    timeframe_suffix: Optional[str] = None,
) -> dict[str, float]:
    """
    Compute Current Bar Features for a single timestamp using that timestamp's
    higher-timeframe lookback window (right-closed). Only the last two bars in
    the lookback are used to form t and t−1 within that timeframe.

    Returns a dict of feature_name -> value (optionally suffixed with timeframe).
    """
    if lookback is None or len(lookback) == 0:
        return {}

    lb = _ensure_ohlcv_lowercase(lookback)
    if not set(["open","high","low","close","volume"]).issubset(lb.columns):
        return {}

    # Last (t) and previous (t−1) bars in this timeframe
    last = lb.iloc[-1]
    prev = lb.iloc[-2] if len(lb) >= 2 else None

    O_t = float(last.get("open", np.nan))
    H_t = float(last.get("high", np.nan))
    L_t = float(last.get("low", np.nan))
    C_t = float(last.get("close", np.nan))
    V_t = float(last.get("volume", np.nan))

    C_tm1 = float(prev.get("close", np.nan)) if prev is not None else np.nan
    V_tm1 = float(prev.get("volume", np.nan)) if prev is not None else np.nan

    # close_logret_current = log(C_t / C_{t−1})
    with np.errstate(divide="ignore", invalid="ignore"):
        close_logret = np.log(C_t / C_tm1) if (C_t > 0 and C_tm1 > 0) else np.nan

    # high_low_range_pct_current = (H_t − L_t) / O_t
    denom_O = np.nan if O_t == 0 else O_t
    hl_range_pct = (H_t - L_t) / denom_O if denom_O and np.isfinite(denom_O) else np.nan

    # close_open_pct_current = (C_t − O_t) / O_t
    co_pct = (C_t - O_t) / denom_O if denom_O and np.isfinite(denom_O) else np.nan

    # log_volume and delta
    log_vol = np.log1p(V_t) if np.isfinite(V_t) else np.nan
    log_vol_tm1 = np.log1p(V_tm1) if np.isfinite(V_tm1) else np.nan
    log_vol_delta = log_vol - log_vol_tm1 if (np.isfinite(log_vol) and np.isfinite(log_vol_tm1)) else np.nan

    sign_close_logret = float(np.sign(close_logret)) if np.isfinite(close_logret) else np.nan

    # Interactions
    cl_x_lv = close_logret * log_vol if (np.isfinite(close_logret) and np.isfinite(log_vol)) else np.nan
    cl_x_lvd = close_logret * log_vol_delta if (np.isfinite(close_logret) and np.isfinite(log_vol_delta)) else np.nan
    sgncl_x_lv = sign_close_logret * log_vol if (np.isfinite(sign_close_logret) and np.isfinite(log_vol)) else np.nan
    hlpct_x_lv = hl_range_pct * log_vol if (np.isfinite(hl_range_pct) and np.isfinite(log_vol)) else np.nan
    copct_x_lv = co_pct * log_vol if (np.isfinite(co_pct) and np.isfinite(log_vol)) else np.nan

    out = {
        "close_logret_current": close_logret,
        "high_low_range_pct_current": hl_range_pct,
        "close_open_pct_current": co_pct,
        "log_volume": log_vol,
        "log_volume_delta_current": log_vol_delta,
        "sign_close_logret_current": sign_close_logret,
        "close_logret_current_x_log_volume": cl_x_lv,
        "close_logret_current_x_log_volume_delta_current": cl_x_lvd,
        "sign_close_logret_current_x_log_volume": sgncl_x_lv,
        "high_low_range_pct_current_x_log_volume": hlpct_x_lv,
        "close_open_pct_current_x_log_volume": copct_x_lv,
    }

    if timeframe_suffix is not None:
        out = {f"{k}_{timeframe_suffix}": v for k, v in out.items()}
    return out


def build_current_bar_features_from_store(store: dict, timeframe: str) -> pd.DataFrame:
    """
    For a timeframe store produced by build_lookbacks.py, compute Current Bar
    Features per base timestamp by evaluating each row's lookback and keeping
    only the last-bar features.
    """
    tf = str(timeframe)
    base_index = store.get("base_index")
    rows_map = store.get("rows", {})
    if base_index is None:
        raise ValueError("store missing 'base_index'")
    results: list[dict[str, float]] = []
    for ts in base_index:
        ts_key = pd.Timestamp(ts).strftime('%Y%m%d_%H%M%S')
        lb = rows_map.get(ts_key)
        feats = compute_current_bar_features_from_lookback(lb, timeframe_suffix=tf)
        feats["timestamp"] = pd.Timestamp(ts)
        results.append(feats)
    df = pd.DataFrame(results).set_index("timestamp")
    return df


def _rename_for_lag(base_name: str, lag_k: int) -> str:
    """Rename a current-bar feature base name to its lag-k counterpart.

    - If the name encodes an interaction with `_x_`, apply lag renaming to both sides.
    - If the token `_current_` or suffix `_current` is present, convert to `_lag_k_` or `_lag_k`.
    - Otherwise append `_lag_k` to the base name.
    """
    if "_x_" in base_name:
        left, right = base_name.split("_x_", 1)
        return f"{_rename_for_lag(left, lag_k)}_x_{_rename_for_lag(right, lag_k)}"
    if "_current_" in base_name:
        return base_name.replace("_current_", f"_lag_{lag_k}_")
    if base_name.endswith("_current"):
        return base_name[: -len("_current")] + f"_lag_{lag_k}"
    return f"{base_name}_lag_{lag_k}"


def compute_current_and_lag_features_from_lookback(
    lookback: pd.DataFrame,
    timeframe_suffix: str,
    max_lag: int = 3,
) -> dict[str, float]:
    """
    For a single timeframe lookback DataFrame (multiple rows), compute current
    bar features for ALL rows using compute_current_bar_features, then extract:
      - current (last row) values
      - lag-1..lag-N values from previous rows within this lookback
    and return a flat dict of feature_name_with_suffix -> value.
    """
    if lookback is None or len(lookback) == 0:
        return {}

    feats_df = compute_current_bar_features(
        ohlcv=lookback,
        timeframe_suffix=None,
        include_original=False,
    )
    if feats_df is None or feats_df.empty:
        return {}

    out: dict[str, float] = {}
    t_idx = len(feats_df) - 1
    # current values
    for col in feats_df.columns:
        key = f"{col}_{timeframe_suffix}"
        out[key] = float(feats_df.iloc[t_idx][col]) if pd.notna(feats_df.iloc[t_idx][col]) else np.nan

    # lag values 1..max_lag
    for k in range(1, max_lag + 1):
        src_idx = t_idx - k
        if src_idx < 0:
            # no further history within this lookback
            break
        for col in feats_df.columns:
            lag_name = _rename_for_lag(col, k)
            key = f"{lag_name}_{timeframe_suffix}"
            val = feats_df.iloc[src_idx][col]
            out[key] = float(val) if pd.notna(val) else np.nan

    return out


