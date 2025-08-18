import numpy as np
import pandas as pd
from typing import Dict, Iterable


def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """Validate that `data` has required OHLCV columns and is non-empty.

    Required columns: 'open', 'high', 'low', 'close', 'volume'.
    Returns True if valid; False otherwise.
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if data is None or not isinstance(data, pd.DataFrame):
        return False
    if len(data) == 0:
        return False
    if not all(col in data.columns for col in required_cols):
        return False
    return True


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Safely divide two numbers.

    - Returns `default` if denominator is zero or either input is NaN.
    - Otherwise returns numerator / denominator.
    """
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return default
    return numerator / denominator


# -----------------------------------------------------------------------------
# Lookback and timeframe utilities (leakage-safe)
# -----------------------------------------------------------------------------

def get_lookback_window(data: pd.DataFrame, current_idx: int, window_size: int) -> pd.DataFrame:
    """Return the right-aligned lookback window [current_idx-window_size+1, current_idx].

    - No leakage: strictly ends at current_idx
    - Assumes a DatetimeIndex (regularly spaced)
    """
    start = max(0, current_idx + 1 - window_size)
    return data.iloc[start: current_idx + 1]


def resample_ohlcv_right_closed(lookback: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Right-closed/right-labeled OHLCV resample that avoids future leakage.

    - Uses label='right', closed='right' so each aggregated bar represents (t-Δ, t]
    - Drops any trailing bin that ends after the lookback last timestamp
    - Aggregation: open=first, high=max, low=min, close=last, volume=sum
    """
    if lookback is None or lookback.empty:
        return lookback
    if not isinstance(lookback.index, (pd.DatetimeIndex,)):
        raise TypeError("resample_ohlcv_right_closed requires a DatetimeIndex")

    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    current_ts = lookback.index[-1]
    out = (
        lookback.resample(rule, label='right', closed='right')
               .agg(agg)
    )
    # Keep only bins that end on/before current_ts (excludes any future/partial bin)
    out = out.loc[out.index <= current_ts]
    return out


def generate_timeframe_lookbacks(
    data: pd.DataFrame,
    current_idx: int,
    window_size: int,
    timeframes: Iterable[str] = ("1H", "4H", "12H", "1D"),
) -> Dict[str, pd.DataFrame]:
    """Build per-timeframe lookback windows ending at the current row.

    - '1H' (or the native frequency) returns the raw lookback slice
    - Higher timeframes are resampled right-closed to avoid using future bars
    - Returns a dict like { '1H': df1h, '4H': df4h, '12H': df12h, '1D': df1d }
    """
    lb = get_lookback_window(data, current_idx, window_size)
    results: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        tfu = str(tf).upper()
        if tfu in ("1H", "H", "60T"):
            results[tf] = lb
        else:
            results[tf] = resample_ohlcv_right_closed(lb, tfu)
    return results


