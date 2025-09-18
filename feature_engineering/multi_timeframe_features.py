"""
Core feature calculation functions.

These are pure functions that take data and return calculated features.
Each function works on any timeframe data (1H, 4H, 1D, etc.).

Based on 52 features from docs/feature_engineering.md
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, List
from scipy import stats
from scipy.fft import fft
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# BASIC TRANSFORMATIONS AND LAGS (Features 1-8)
# =============================================================================

def get_lags(data: pd.Series, lags: List[int], column_name: str = "close") -> Dict[str, float]:
    """1. Generic Lags: Get lagged values for any column with configurable naming"""
    result = {}
    for lag in lags:
        if len(data) > lag:
            result[f'{column_name}_lag_{lag}'] = data.iloc[-1-lag]
        else:
            result[f'{column_name}_lag_{lag}'] = np.nan
    return result


def calculate_price_differences(open_s: pd.Series, high_s: pd.Series, low_s: pd.Series, close_s: pd.Series, lag: int = 0) -> Dict[str, float]:
    """2. Price Differences: Calculate differences for a specific lag in OHLC data

    Apply on: open, high, low, close series (aligned and same length)
    """
    if len(close_s) == 0:
        return {}

    # Calculate the actual row index based on lag
    # lag 0 = current row (last), lag 1 = previous row, lag 2 = two back, etc.
    row_idx = -1 - lag

    # Check if the row exists
    if abs(row_idx) > len(close_s):
        return {
            'close_open_diff': np.nan,
            'high_low_range': np.nan,
            'price_change': np.nan
        }

    # Create column names with lag terminology
    if lag == 0:
        row_suffix = "_current"
    else:
        row_suffix = f"_lag_{lag}"

    result = {
        f'close_open_diff{row_suffix}': close_s.iloc[row_idx] - open_s.iloc[row_idx],
        f'high_low_range{row_suffix}': high_s.iloc[row_idx] - low_s.iloc[row_idx]
    }

    # Calculate close change for the specific row we're analyzing
    if abs(row_idx - 1) <= len(close_s):
        prev_close = close_s.iloc[row_idx - 1]
        result[f'close_change{row_suffix}'] = close_s.iloc[row_idx] - prev_close
    else:
        result[f'close_change{row_suffix}'] = np.nan

    return result


def calculate_log_transform(series: pd.Series, column_name: str = 'close') -> Dict[str, float]:
    """3. Log Transformation: log(series)

    Apply on: open, high, low, close, volume (positive values only)
    """
    if len(series) == 0:
        return {}

    value = series.iloc[-1]
    if value > 0:
        return {f'log_{column_name}': np.log(value)}
    return {f'log_{column_name}': np.nan}


def calculate_percentage_changes(data: pd.Series, lag: int = 0, column_name: str = "close") -> Dict[str, float]:
    """4. Percentage Changes: (Close - Close_{t-1}) / Close_{t-1} * 100 for a specific lag"""
    if len(data) == 0:
        return {}
    
    # Calculate the actual row index based on lag
    # lag 0 = current row (last), lag 1 = previous row, lag 2 = two back, etc.
    row_idx = -1 - lag
    
    # Check if the row exists
    if abs(row_idx) > len(data):
        return {f'{column_name}_pct_change': np.nan}
    
    current_price = data.iloc[row_idx]
    
    # Create column names with lag terminology
    if lag == 0:
        row_suffix = "_current"
    else:
        row_suffix = f"_lag_{lag}"
    
    # Calculate percentage change from current row to previous row (lag 1)
    if abs(row_idx - 1) <= len(data):
        prev_price = data.iloc[row_idx - 1]  # Move backward one row
        if prev_price != 0:
            result = {f'{column_name}_pct_change{row_suffix}': (current_price - prev_price) / prev_price * 100}
        else:
            result = {f'{column_name}_pct_change{row_suffix}': np.nan}
    else:
        result = {f'{column_name}_pct_change{row_suffix}': np.nan}
    
    return result


def calculate_cumulative_returns(data: pd.Series, windows: List[int], column_name: str = "close") -> Dict[str, float]:
    """5. Cumulative Returns: Sum of log returns over rolling windows

    Apply on: close
    """
    result = {}
    
    if len(data) < 2:
        for window in windows:
            result[f'{column_name}_cum_return_{window}'] = np.nan
        return result
    
    log_returns = np.log(data / data.shift(1)).dropna()
    
    for window in windows:
        if len(log_returns) >= window:
            result[f'{column_name}_cum_return_{window}'] = log_returns.tail(window).sum()
        else:
            result[f'{column_name}_cum_return_{window}'] = np.nan
    
    return result


def calculate_zscore(data: pd.Series, window: int, column_name: str = "close") -> Dict[str, float]:
    """6. Z-Scores: (price - rolling_mean) / rolling_std

    Apply on: any numeric series (e.g., close, volume)
    Returns: { '{column_name}_zscore_{window}': value }
    """
    key = f"{column_name}_zscore_{window}"
    if len(data) < window:
        return {key: np.nan}

    rolling_data = data.tail(window)
    mean = rolling_data.mean()
    std = rolling_data.std()

    if std == 0:
        return {key: np.nan}

    value = (data.iloc[-1] - mean) / std
    return {key: float(value)}


def calculate_volume_lags_and_changes(volume: pd.Series, lags: List[int]) -> Dict[str, float]:
    """7-8. Volume Lags and Changes"""
    result = {}
    
    # Get volume lags using the generic function
    volume_lags = get_lags(volume, lags, "volume")
    result.update(volume_lags)
    
    # Volume changes
    if len(volume) > 1:
        current_vol = volume.iloc[-1]
        prev_vol = volume.iloc[-2]
        result['volume_change'] = current_vol - prev_vol
    else:
        result['volume_change'] = np.nan
    
    # Get volume percentage changes using the generic function
    volume_pct_changes = calculate_percentage_changes(volume, 0, "volume")
    result.update(volume_pct_changes)
    
    return result


# =============================================================================
# MOVING AVERAGES AND TREND FEATURES (Features 9-15)
# =============================================================================

def calculate_sma(series: pd.Series, window: int, column_name: str = 'close') -> Dict[str, float]:
    """9. Simple Moving Average

    Apply on: close, open, high, low, volume
    Returns: { '{column_name}_sma_{window}': value }
    """
    key = f"{column_name}_sma_{window}"
    if len(series) < window:
        return {key: np.nan}
    value = series.rolling(window=window).mean().iloc[-1]
    return {key: float(value)}


def calculate_ema(series: pd.Series, span: int, column_name: str = 'close') -> Dict[str, float]:
    """10. Exponential Moving Average

    Apply on: close, open, high, low, volume
    Returns: { '{column_name}_ema_{span}': value }
    """
    key = f"{column_name}_ema_{span}"
    if len(series) < span:
        return {key: np.nan}
    value = series.ewm(span=span).mean().iloc[-1]
    return {key: float(value)}


def calculate_wma(series: pd.Series, window: int, column_name: str = 'close') -> Dict[str, float]:
    """11. Weighted Moving Average

    Apply on: close, open, high, low, volume
    Returns: { '{column_name}_wma_{window}': value }
    """
    key = f"{column_name}_wma_{window}"
    if len(series) < window:
        return {key: np.nan}
    weights = np.arange(1, window + 1)
    values = series.tail(window).values
    value = float(np.sum(weights * values) / np.sum(weights))
    return {key: value}


def calculate_ma_crossovers(series: pd.Series, fast_window: int, slow_window: int, column_name: str = 'close') -> Dict[str, float]:
    """12. Moving Average Crossovers

    Apply on: close (typically)
    Returns keys with windows encoded, e.g., '{col}_ma_cross_diff_5_20'
    """
    if len(series) < max(fast_window, slow_window):
        return {
            f'{column_name}_ma_cross_diff_{fast_window}_{slow_window}': np.nan,
            f'{column_name}_ma_cross_ratio_{fast_window}_{slow_window}': np.nan,
            f'{column_name}_ma_cross_signal_{fast_window}_{slow_window}': np.nan
        }
    ma_fast = series.rolling(window=fast_window).mean().iloc[-1]
    ma_slow = series.rolling(window=slow_window).mean().iloc[-1]
    diff = float(ma_fast - ma_slow) if not (np.isnan(ma_fast) or np.isnan(ma_slow)) else np.nan
    ratio = (float(ma_fast / ma_slow) if ma_slow != 0 else np.nan) if not (np.isnan(ma_fast) or np.isnan(ma_slow)) else np.nan
    signal = (1 if (not np.isnan(ma_fast) and not np.isnan(ma_slow) and ma_fast > ma_slow) else 0)
    return {
        f'{column_name}_ma_cross_diff_{fast_window}_{slow_window}': diff,
        f'{column_name}_ma_cross_ratio_{fast_window}_{slow_window}': ratio,
        f'{column_name}_ma_cross_signal_{fast_window}_{slow_window}': signal
    }


def calculate_ma_distance(current_value: float, ma_value: float, column_name: str = 'close', ma_label: str = 'ma') -> Dict[str, float]:
    """13. Distance to Moving Averages

    Apply on: close (or any series scalar)
    Returns keys: '{col}_ma_distance_{label}', '{col}_ma_distance_pct_{label}'
    """
    dist_key = f"{column_name}_ma_distance_{ma_label}"
    pct_key = f"{column_name}_ma_distance_pct_{ma_label}"
    if np.isnan(ma_value) or ma_value == 0:
        return {dist_key: np.nan, pct_key: np.nan}
    distance = float(current_value - ma_value)
    distance_pct = float((distance / ma_value) * 100)
    return {dist_key: distance, pct_key: distance_pct}


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, column_name: str = 'close') -> Dict[str, float]:
    """14. MACD: EMA_12 - EMA_26, Signal line, Histogram

    Apply on: close
    Keys: '{col}_macd_line_{fast}_{slow}', '{col}_macd_signal_{signal}', '{col}_macd_histogram_{fast}_{slow}_{signal}'
    """
    if len(series) < max(fast, slow, signal):
        return {
            f'{column_name}_macd_line_{fast}_{slow}': np.nan,
            f'{column_name}_macd_signal_{signal}': np.nan,
            f'{column_name}_macd_histogram_{fast}_{slow}_{signal}': np.nan
        }
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = float((ema_fast - ema_slow).iloc[-1])
    macd_series = ema_fast - ema_slow
    if len(macd_series) >= signal:
        macd_signal = float(macd_series.ewm(span=signal).mean().iloc[-1])
        macd_histogram = float(macd_line - macd_signal)
    else:
        macd_signal = np.nan
        macd_histogram = np.nan
    return {
        f'{column_name}_macd_line_{fast}_{slow}': macd_line,
        f'{column_name}_macd_signal_{signal}': macd_signal,
        f'{column_name}_macd_histogram_{fast}_{slow}_{signal}': macd_histogram
    }


def calculate_volume_ma(volume: pd.Series, window: int, column_name: str = 'volume') -> Dict[str, float]:
    """15. Moving Average of Volume

    Apply on: volume (SMA)
    Returns: { 'volume_sma_{window}': value }
    """
    return calculate_sma(volume, window, column_name)


# =============================================================================
# MOMENTUM AND OSCILLATOR FEATURES (Features 16-22)
# =============================================================================

def calculate_rsi(series: pd.Series, window: int = 14, column_name: str = 'close') -> Dict[str, float]:
    """16. Relative Strength Index (Wilder's RSI)

    Apply on: close
    Returns: { '{column_name}_rsi_{window}': value }
    """
    key = f"{column_name}_rsi_{window}"
    if len(series) < window + 1:
        return {key: np.nan}

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain_series = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss_series = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    last_avg_gain = avg_gain_series.iloc[-1]
    last_avg_loss = avg_loss_series.iloc[-1]

    if np.isnan(last_avg_gain) or np.isnan(last_avg_loss):
        return {key: np.nan}

    if last_avg_loss == 0:
        return {key: 100.0}

    rs = last_avg_gain / last_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return {key: float(rsi)}


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_window: int = 14, d_window: int = 3, column_name: str = 'close') -> Dict[str, float]:
    """17. Stochastic Oscillator: %K and %D

    Apply on: high, low, close
    Returns: { '{col}_stoch_k_{k}', '{col}_stoch_d_{k}_{d}' }
    """
    if len(close) < k_window:
        return {f'{column_name}_stoch_k_{k_window}': np.nan, f'{column_name}_stoch_d_{k_window}_{d_window}': np.nan}
    
    lowest_low = low.rolling(window=k_window).min().iloc[-1]
    highest_high = high.rolling(window=k_window).max().iloc[-1]
    
    if highest_high == lowest_low:
        k_percent = 50.0
    else:
        k_percent = 100 * (close.iloc[-1] - lowest_low) / (highest_high - lowest_low)
    
    # Calculate %D (SMA of %K)
    if len(close) >= k_window + d_window - 1:
        k_values = []
        for i in range(d_window):
            idx = len(close) - 1 - i
            if idx < k_window - 1:
                break
            ll = low.iloc[idx-k_window+1:idx+1].min()
            hh = high.iloc[idx-k_window+1:idx+1].max()
            if hh == ll:
                k_val = 50.0
            else:
                k_val = 100 * (close.iloc[idx] - ll) / (hh - ll)
            k_values.append(k_val)
        
        d_percent = np.mean(k_values) if len(k_values) == d_window else np.nan
    else:
        d_percent = np.nan
    
    return {f'{column_name}_stoch_k_{k_window}': float(k_percent), f'{column_name}_stoch_d_{k_window}_{d_window}': float(d_percent) if not np.isnan(d_percent) else np.nan}


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """18. Commodity Channel Index

    Apply on: high, low, close
    Returns: { '{col}_cci_{window}': value }
    """
    if len(close) < window:
        return {f'{column_name}_cci_{window}': np.nan}
    
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean().iloc[-1]
    
    # Mean deviation
    rolling_tp = typical_price.tail(window)
    mean_deviation = np.mean(np.abs(rolling_tp - rolling_tp.mean()))
    
    if mean_deviation == 0:
        return {f'{column_name}_cci_{window}': np.nan}
    cci = (typical_price.iloc[-1] - sma_tp) / (0.015 * mean_deviation)
    return {f'{column_name}_cci_{window}': float(cci)}


def calculate_roc(series: pd.Series, period: int, column_name: str = 'close') -> Dict[str, float]:
    """19. Rate of Change

    Apply on: close or volume
    Returns: { '{col}_roc_{period}': value }
    """
    key = f"{column_name}_roc_{period}"
    if len(series) <= period:
        return {key: np.nan}
    current = series.iloc[-1]
    past = series.iloc[-1-period]
    if past == 0:
        return {key: np.nan}
    return {key: float(((current - past) / past) * 100)}


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, column_name: str = 'close') -> Dict[str, float]:
    """20. Williams %R

    Apply on: high, low, close
    Returns: { '{col}_williams_r_{window}': value }
    """
    if len(close) < window:
        return {f'{column_name}_williams_r_{window}': np.nan}
    
    highest_high = high.rolling(window=window).max().iloc[-1]
    lowest_low = low.rolling(window=window).min().iloc[-1]
    
    if highest_high == lowest_low:
        return {f'{column_name}_williams_r_{window}': -50.0}
    williams_r = -100 * (highest_high - close.iloc[-1]) / (highest_high - lowest_low)
    return {f'{column_name}_williams_r_{window}': float(williams_r)}


def calculate_ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                                periods: List[int] = [7, 14, 28], 
                                weights: List[float] = [4, 2, 1],
                                column_name: str = 'close') -> Dict[str, float]:
    """21. Ultimate Oscillator

    Apply on: high, low, close
    Returns: { '{col}_uo_{p1}_{p2}_{p3}': value }
    """
    if len(close) < max(periods):
        return {f'{column_name}_uo_{"_".join(map(str, periods))}': np.nan}
    
    def buying_pressure(h, l, c):
        return c - np.minimum(l, c.shift(1))
    
    def true_range_calc(h, l, c):
        return np.maximum(h, c.shift(1)) - np.minimum(l, c.shift(1))
    
    bp = buying_pressure(high, low, close)
    tr = true_range_calc(high, low, close)
    
    averages = []
    for i, period in enumerate(periods):
        if len(bp) >= period:
            avg = bp.rolling(window=period).sum().iloc[-1] / tr.rolling(window=period).sum().iloc[-1]
            averages.append(avg * weights[i])
        else:
            return {f'{column_name}_uo_{"_".join(map(str, periods))}': np.nan}
    uo = 100 * sum(averages) / sum(weights)
    return {f'{column_name}_uo_{"_".join(map(str, periods))}': float(uo)}


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                 volume: pd.Series, window: int = 14, column_name: str = 'close') -> Dict[str, float]:
    """22. Money Flow Index

    Apply on: high, low, close, volume
    Returns: { '{col}_mfi_{window}': value }
    """
    key = f"{column_name}_mfi_{window}"
    if len(close) < window + 1:
        return {key: np.nan}
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    price_change = typical_price.diff()
    positive_flow = money_flow.where(price_change > 0, 0)
    negative_flow = money_flow.where(price_change < 0, 0)
    
    positive_mf = positive_flow.rolling(window=window).sum().iloc[-1]
    negative_mf = negative_flow.rolling(window=window).sum().iloc[-1]
    
    if negative_mf == 0:
        return {key: 100.0}
    mfr = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfr))
    return {key: float(mfi)}


# =============================================================================
# VOLATILITY FEATURES (Features 23-28)
# =============================================================================

def calculate_historical_volatility(series: pd.Series, window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """23. Historical Volatility: std of log returns (annualized)

    Apply on: close
    Returns: { '{col}_hv_{window}': value }
    """
    key = f"{column_name}_hv_{window}"
    if len(series) < window + 1:
        return {key: np.nan}
    returns = np.log(series / series.shift(1)).dropna()
    if len(returns) < window:
        return {key: np.nan}
    value = float(returns.tail(window).std() * np.sqrt(365))
    return {key: value}


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, column_name: str = 'close') -> Dict[str, float]:
    """24. Average True Range

    Apply on: high, low, close
    Returns: { '{col}_atr_{window}': value }
    """
    key = f"{column_name}_atr_{window}"
    if len(close) < window + 1:
        return {key: np.nan}
    
    tr_list = []
    for i in range(1, len(close)):
        h = high.iloc[i]
        l = low.iloc[i]
        c_prev = close.iloc[i-1]
        
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    
    if len(tr_list) < window:
        return {key: np.nan}
    value = float(np.mean(tr_list[-window:]))
    return {key: value}


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0, column_name: str = 'close') -> Dict[str, float]:
    """25. Bollinger Bands: Upper, Lower, Width, %B

    Apply on: close
    Returns: keys prefixed with column and window/num_std
    """
    def fmt_std(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else str(x).replace('.', '_')
    std_tag = fmt_std(num_std)
    keys = {
        'upper': f'{column_name}_bb_upper_{window}_{std_tag}',
        'middle': f'{column_name}_bb_middle_{window}',
        'lower': f'{column_name}_bb_lower_{window}_{std_tag}',
        'width': f'{column_name}_bb_width_{window}_{std_tag}',
        'percent': f'{column_name}_bb_percent_{window}_{std_tag}',
    }
    if len(series) < window:
        return {keys['upper']: np.nan, keys['middle']: np.nan, keys['lower']: np.nan, keys['width']: np.nan, keys['percent']: np.nan}
    rolling_data = series.tail(window)
    middle = rolling_data.mean()
    std = rolling_data.std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    width = upper - lower
    current_price = series.iloc[-1]
    bb_percent = 0.5 if width == 0 else (current_price - lower) / width
    return {keys['upper']: float(upper), keys['middle']: float(middle), keys['lower']: float(lower), keys['width']: float(width), keys['percent']: float(bb_percent)}


def calculate_volatility_ratio(series: pd.Series, short_window: int = 5, long_window: int = 50, column_name: str = 'close') -> Dict[str, float]:
    """26. Volatility Ratios: short-term vol / long-term vol

    Apply on: close
    Returns: { '{col}_vol_ratio_{short}_{long}': value }
    """
    key = f"{column_name}_vol_ratio_{short_window}_{long_window}"
    if len(series) < long_window + 1:
        return {key: np.nan}
    returns = np.log(series / series.shift(1)).dropna()
    if len(returns) < long_window:
        return {key: np.nan}
    short_vol = returns.tail(short_window).std()
    long_vol = returns.tail(long_window).std()
    if long_vol == 0:
        return {key: np.nan}
    return {key: float(short_vol / long_vol)}


def calculate_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """27. Parkinson Volatility: based on High-Low ranges

    Apply on: high, low
    Returns: { '{col}_parkinson_{window}': value }
    """
    key = f"{column_name}_parkinson_{window}"
    if len(high) < window:
        return {key: np.nan}
    log_hl_ratio = np.log(high / low)
    parkinson_values = log_hl_ratio ** 2
    if len(parkinson_values) < window:
        return {key: np.nan}
    value = float(np.sqrt((1 / (4 * np.log(2))) * parkinson_values.tail(window).mean()))
    return {key: value}


def calculate_garman_klass_volatility(high: pd.Series, low: pd.Series, open_s: pd.Series, close: pd.Series, window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """28. Garman-Klass Volatility: incorporates OHLC

    Apply on: high, low, open, close
    Returns: { '{col}_gk_{window}': value }
    """
    key = f"{column_name}_gk_{window}"
    if len(close) < window:
        return {key: np.nan}
    term1 = 0.5 * (np.log(high / low)) ** 2
    term2 = (2 * np.log(2) - 1) * (np.log(close / open_s)) ** 2
    gk_values = term1 - term2
    if len(gk_values) < window:
        return {key: np.nan}
    value = float(np.sqrt(gk_values.tail(window).mean()))
    return {key: value}


# =============================================================================
# VOLUME-INTEGRATED FEATURES (Features 29-33)
# =============================================================================

def calculate_obv(close: pd.Series, volume: pd.Series, column_name: str = 'close') -> Dict[str, float]:
    """29. On-Balance Volume

    Apply on: close, volume
    Returns: { '{col}_obv': value }
    """
    key = f"{column_name}_obv"
    if len(close) < 2:
        return {key: np.nan}

    obv_value = 0.0
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv_value += float(volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv_value -= float(volume.iloc[i])

    return {key: obv_value}


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, column_name: str = 'close') -> Dict[str, float]:
    """30. Volume Weighted Average Price

    Apply on: high, low, close, volume
    Returns: { '{col}_vwap': value }
    """
    key = f"{column_name}_vwap"
    if len(close) == 0:
        return {key: np.nan}

    typical_price = (high + low + close) / 3
    total_volume_price = float((typical_price * volume).sum())
    total_volume = float(volume.sum())

    if total_volume == 0:
        return {key: np.nan}

    return {key: total_volume_price / total_volume}


def calculate_adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, column_name: str = 'close') -> Dict[str, float]:
    """31. Accumulation/Distribution Line

    Apply on: high, low, close, volume
    Returns: { '{col}_adl': value }
    """
    key = f"{column_name}_adl"
    if len(close) == 0:
        return {key: np.nan}

    adl_value = 0.0
    for i in range(len(close)):
        if high.iloc[i] != low.iloc[i]:
            clv = ((close.iloc[i] - low.iloc[i]) - (high.iloc[i] - close.iloc[i])) / (high.iloc[i] - low.iloc[i])
            adl_value += float(clv * volume.iloc[i])

    return {key: adl_value}


def calculate_chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                               volume: pd.Series, fast: int = 3, slow: int = 10, column_name: str = 'close') -> Dict[str, float]:
    """32. Chaikin Oscillator: EMA_3 - EMA_10 of ADL

    Apply on: high, low, close, volume
    Returns: { '{col}_chaikin_{fast}_{slow}': value }
    """
    key = f"{column_name}_chaikin_{fast}_{slow}"
    if len(close) < max(fast, slow):
        return {key: np.nan}

    # Calculate ADL series
    adl_series: List[float] = []
    adl_cumulative: float = 0.0

    for i in range(len(close)):
        if high.iloc[i] != low.iloc[i]:
            clv = ((close.iloc[i] - low.iloc[i]) - (high.iloc[i] - close.iloc[i])) / (high.iloc[i] - low.iloc[i])
            adl_cumulative += float(clv * volume.iloc[i])
        adl_series.append(adl_cumulative)

    adl_pd = pd.Series(adl_series)

    if len(adl_pd) < max(fast, slow):
        return {key: np.nan}

    ema_fast = float(adl_pd.ewm(span=fast).mean().iloc[-1])
    ema_slow = float(adl_pd.ewm(span=slow).mean().iloc[-1])

    return {key: ema_fast - ema_slow}


def calculate_volume_roc(volume: pd.Series, period: int, column_name: str = 'volume') -> Dict[str, float]:
    """33. Volume Rate of Change

    Apply on: volume
    Returns: { '{col}_roc_{period}': value }
    """
    return calculate_roc(volume, period, column_name)


# =============================================================================
# STATISTICAL AND DISTRIBUTIONAL FEATURES (Features 34-38)
# =============================================================================

def calculate_rolling_percentiles(data: pd.Series, window: int,
                                percentiles: List[float] = [25, 50, 75],
                                column_name: str = 'close') -> Dict[str, float]:
    """34. Rolling Medians and Percentiles

    Apply on: any numeric series
    Returns keys like '{col}_percentile_{p}_{window}'
    """
    if len(data) < window:
        return {f'{column_name}_percentile_{p}_{window}': np.nan for p in percentiles}

    rolling_data = data.tail(window)
    return {f'{column_name}_percentile_{p}_{window}': float(np.percentile(rolling_data, p)) for p in percentiles}


def calculate_distribution_features(data: pd.Series, window: int = 30, column_name: str = 'close') -> Dict[str, float]:
    """35. Kurtosis and Skewness of returns

    Apply on: any price/volume series (uses returns)
    Returns: '{col}_skew_{window}', '{col}_kurt_{window}'
    """
    skew_key = f'{column_name}_skew_{window}'
    kurt_key = f'{column_name}_kurt_{window}'
    if len(data) < window + 1:
        return {skew_key: np.nan, kurt_key: np.nan}

    returns = data.pct_change().dropna()
    if len(returns) < window:
        return {skew_key: np.nan, kurt_key: np.nan}

    rolling_returns = returns.tail(window)
    return {skew_key: float(stats.skew(rolling_returns)), kurt_key: float(stats.kurtosis(rolling_returns))}


def calculate_autocorrelation(data: pd.Series, lag: int = 1, window: int = 30, column_name: str = 'close') -> Dict[str, float]:
    """36. Autocorrelation of returns

    Apply on: any price/volume series (uses returns)
    Returns: '{col}_autocorr_{lag}_{window}'
    """
    key = f'{column_name}_autocorr_{lag}_{window}'
    if len(data) < window + lag + 1:
        return {key: np.nan}

    returns = data.pct_change().dropna()
    if len(returns) < window + lag:
        return {key: np.nan}

    rolling_returns = returns.tail(window + lag)
    return {key: float(rolling_returns.autocorr(lag=lag))}


def calculate_hurst_exponent(data: pd.Series, window: int = 100, column_name: str = 'close') -> Dict[str, float]:
    """37. Hurst Exponent via rescaled range analysis

    Apply on: any price series
    Returns: '{col}_hurst_{window}'
    """
    key = f'{column_name}_hurst_{window}'
    if len(data) < window:
        return {key: np.nan}

    try:
        log_returns = np.log(data / data.shift(1)).dropna().tail(window)
        if len(log_returns) < 10:
            return {key: np.nan}

        lags = range(2, min(20, len(log_returns) // 2))
        rs_values = []

        for lag in lags:
            Y = log_returns.values
            mean_Y = np.mean(Y)
            Z = np.cumsum(Y - mean_Y)
            R = np.max(Z) - np.min(Z)
            S = np.std(Y)
            if S > 0:
                rs_values.append(R / S)

        if len(rs_values) < 3:
            return {key: np.nan}

        log_lags = np.log(list(lags[:len(rs_values)]))
        log_rs = np.log(rs_values)
        hurst = float(np.polyfit(log_lags, log_rs, 1)[0])
        return {key: hurst}

    except Exception:
        return {key: np.nan}


def calculate_entropy(data: pd.Series, window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """38. Approximate entropy of price series

    Apply on: any price series
    Returns: '{col}_entropy_{window}'
    """
    key = f'{column_name}_entropy_{window}'
    if len(data) < window:
        return {key: np.nan}

    try:
        rolling_data = data.tail(window).values
        n_bins = min(10, len(rolling_data) // 2)
        hist, _ = np.histogram(rolling_data, bins=n_bins)
        hist = hist / np.sum(hist)
        entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
        return {key: entropy}
    except Exception:
        return {key: np.nan}


# =============================================================================
# RATIO AND HYBRID FEATURES (Features 39-44)
# =============================================================================

def calculate_price_volume_ratios(close: pd.Series,
                                 high: pd.Series,
                                 volume: pd.Series,
                                 close_name: str = 'close',
                                 high_name: str = 'high') -> Dict[str, float]:
    """39. Price-to-Volume Ratios

    Apply on: close, high, volume
    Returns: { '{close_name}_volume_ratio', '{high_name}_volume_ratio' }
    """
    result = {}
    if len(close) == 0 or len(high) == 0 or len(volume) == 0:
        result[f'{close_name}_volume_ratio'] = np.nan
        result[f'{high_name}_volume_ratio'] = np.nan
        return result

    current_volume = float(volume.iloc[-1])
    if current_volume == 0 or np.isnan(current_volume):
        result[f'{close_name}_volume_ratio'] = np.nan
        result[f'{high_name}_volume_ratio'] = np.nan
        return result

    result[f'{close_name}_volume_ratio'] = float(close.iloc[-1]) / current_volume
    result[f'{high_name}_volume_ratio'] = float(high.iloc[-1]) / current_volume
    return result


def calculate_candle_patterns(open_s: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                              column_name: str = 'close') -> Dict[str, float]:
    """40-41. Candle Body and Shadow Ratios

    Apply on: open, high, low, close
    Returns: { '{col}_candle_body_ratio', '{col}_candle_upper_shadow_ratio', '{col}_candle_lower_shadow_ratio' }
    """
    keys = {
        'body': f'{column_name}_candle_body_ratio',
        'upper': f'{column_name}_candle_upper_shadow_ratio',
        'lower': f'{column_name}_candle_lower_shadow_ratio',
    }
    if len(close) == 0:
        return {keys['body']: np.nan, keys['upper']: np.nan, keys['lower']: np.nan}

    o = float(open_s.iloc[-1])
    h = float(high.iloc[-1])
    l = float(low.iloc[-1])
    c = float(close.iloc[-1])
    
    range_val = h - l
    if range_val == 0:
        return {keys['body']: np.nan, keys['upper']: np.nan, keys['lower']: np.nan}

    body_ratio = abs(c - o) / range_val
    upper_shadow_ratio = (h - max(o, c)) / range_val
    lower_shadow_ratio = (min(o, c) - l) / range_val

    return {keys['body']: body_ratio, keys['upper']: upper_shadow_ratio, keys['lower']: lower_shadow_ratio}


def calculate_typical_price(high: pd.Series, low: pd.Series, close: pd.Series, column_name: str = 'close') -> Dict[str, float]:
    """42. Typical Price: (High + Low + Close) / 3

    Apply on: high, low, close
    Returns: { '{col}_typical_price': value }
    """
    key = f'{column_name}_typical_price'
    if len(close) == 0:
        return {key: np.nan}
    return {key: float((high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3)}


def calculate_ohlc_average(open_s: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                           column_name: str = 'close') -> Dict[str, float]:
    """43. OHLC Average: (Open + High + Low + Close) / 4

    Apply on: open, high, low, close
    Returns: { '{col}_ohlc_average': value }
    """
    key = f'{column_name}_ohlc_average'
    if len(close) == 0:
        return {key: np.nan}
    value = float((open_s.iloc[-1] + high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 4)
    return {key: value}


def calculate_volatility_adjusted_returns(data: pd.Series, atr_value: float, column_name: str = 'close', atr_label: str = 'atr') -> Dict[str, float]:
    """44. Volatility-Adjusted Returns: log_return / sqrt(ATR)

    Apply on: close (or any price series), with external ATR measure
    Returns: { '{col}_vol_adj_return_{atr_label}': value }
    """
    key = f'{column_name}_vol_adj_return_{atr_label}'
    if len(data) < 2 or atr_value <= 0 or np.isnan(atr_value):
        return {key: np.nan}
    log_return = np.log(data.iloc[-1] / data.iloc[-2])
    return {key: float(log_return / np.sqrt(atr_value))}


# =============================================================================
# TIME-BASED AND CYCLICAL FEATURES (Features 45-48)
# =============================================================================

def calculate_time_features(timestamp: pd.Timestamp) -> Dict[str, float]:
    """45. Time of Day/Week features

    Apply on: timestamp
    Returns: { 'time_hour_of_day', 'time_day_of_week', 'time_day_of_month', 'time_month_of_year' }
    """
    return {
        'time_hour_of_day': timestamp.hour,
        'time_day_of_week': timestamp.dayofweek,
        'time_day_of_month': timestamp.day,
        'time_month_of_year': timestamp.month
    }


def calculate_rolling_extremes(data: pd.Series, window: int = 10, column_name: str = 'close') -> Dict[str, float]:
    """46. Rolling Min/Max and position relative to range

    Apply on: any price series
    Returns keys with column and window, e.g., '{col}_rolling_min_{window}'
    """
    keys = {
        'min': f'{column_name}_rolling_min_{window}',
        'max': f'{column_name}_rolling_max_{window}',
        'pos': f'{column_name}_position_in_range_{window}',
    }
    if len(data) < window:
        return {keys['min']: np.nan, keys['max']: np.nan, keys['pos']: np.nan}

    rolling_data = data.tail(window)
    rolling_min = float(rolling_data.min())
    rolling_max = float(rolling_data.max())
    current_price = float(data.iloc[-1])

    if rolling_max == rolling_min:
        position_in_range = 0.5
    else:
        position_in_range = (current_price - rolling_min) / (rolling_max - rolling_min)

    return {keys['min']: rolling_min, keys['max']: rolling_max, keys['pos']: float(position_in_range)}


def calculate_dominant_cycle(data: pd.Series, window: int = 50, column_name: str = 'close') -> Dict[str, float]:
    """47. Simple Fourier-based dominant cycle detection

    Apply on: any price series
    Returns: '{col}_dominant_cycle_length_{window}', '{col}_cycle_strength_{window}'
    """
    keys = {
        'len': f'{column_name}_dominant_cycle_length_{window}',
        'strength': f'{column_name}_cycle_strength_{window}',
    }
    if len(data) < window:
        return {keys['len']: np.nan, keys['strength']: np.nan}

    try:
        rolling_data = data.tail(window).values
        detrended = rolling_data - np.mean(rolling_data)
        fft_values = np.abs(fft(detrended))
        dominant_freq_idx = np.argmax(fft_values[1:len(fft_values)//2]) + 1
        if dominant_freq_idx == 0:
            return {keys['len']: np.nan, keys['strength']: np.nan}
        dominant_cycle_length = len(rolling_data) / dominant_freq_idx
        denom = np.sum(fft_values[1:len(fft_values)//2])
        cycle_strength = float(fft_values[dominant_freq_idx] / denom) if denom != 0 else np.nan
        return {keys['len']: float(dominant_cycle_length), keys['strength']: cycle_strength}
    except Exception:
        return {keys['len']: np.nan, keys['strength']: np.nan}


# =============================================================================
# ENSEMBLE AND DERIVED FEATURES (Features 49-52) 
# =============================================================================

def calculate_binary_thresholds(values: Dict[str, float], 
                               thresholds: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """51. Binary Threshold Features"""
    result = {}
    
    for indicator, value in values.items():
        if indicator in thresholds and not np.isnan(value):
            thresh_config = thresholds[indicator]
            
            if 'oversold' in thresh_config:
                result[f'{indicator}_oversold'] = 1 if value < thresh_config['oversold'] else 0
            
            if 'overbought' in thresh_config:
                result[f'{indicator}_overbought'] = 1 if value > thresh_config['overbought'] else 0
        else:
            if indicator in thresholds:
                result[f'{indicator}_oversold'] = np.nan
                result[f'{indicator}_overbought'] = np.nan
    
    return result


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, window: int = 20,
                                  series1_name: str = 'close', series2_name: str = 'volume') -> Dict[str, float]:
    """52. Rolling Correlation between two series

    Apply on: any two aligned series
    Returns: '{name1}_{name2}_rolling_corr_{window}'
    """
    key = f'{series1_name}_{series2_name}_rolling_corr_{window}'
    if len(series1) < window or len(series2) < window:
        return {key: np.nan}
    value = series1.tail(window).corr(series2.tail(window))
    return {key: float(value)}


def calculate_interaction_terms(features: Dict[str, float], 
                              interactions: List[tuple]) -> Dict[str, float]:
    """50. Interaction Terms: RSI * Volatility, etc."""
    result = {}
    
    for feature1, feature2 in interactions:
        if feature1 in features and feature2 in features:
            val1, val2 = features[feature1], features[feature2]
            if not (np.isnan(val1) or np.isnan(val2)):
                result[f'{feature1}_x_{feature2}'] = val1 * val2
            else:
                result[f'{feature1}_x_{feature2}'] = np.nan
        else:
            result[f'{feature1}_x_{feature2}'] = np.nan
    
    return result


# =============================================================================
# ADDITIONAL HIGH-IMPACT FEATURES (Trend, Volatility, Volume, Breakout)
# =============================================================================

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                  window: int = 14, column_name: str = 'close') -> Dict[str, float]:
    """ADX/DMI (Wilder) trend strength and directional indicators

    Apply on: high, low, close
    Returns: '{col}_adx_{w}', '{col}_di_plus_{w}', '{col}_di_minus_{w}'
    """
    keys = {
        'adx': f'{column_name}_adx_{window}',
        'dip': f'{column_name}_di_plus_{window}',
        'dim': f'{column_name}_di_minus_{window}',
    }
    if len(close) < window + 1:
        return {keys['adx']: np.nan, keys['dip']: np.nan, keys['dim']: np.nan}

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    rma_tr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rma_plus_dm = plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rma_minus_dm = minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    di_plus = 100 * (rma_plus_dm / rma_tr)
    di_minus = 100 * (rma_minus_dm / rma_tr)
    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).replace([np.inf, -np.inf], np.nan)
    adx_series = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    return {
        keys['adx']: float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else np.nan,
        keys['dip']: float(di_plus.iloc[-1]) if not np.isnan(di_plus.iloc[-1]) else np.nan,
        keys['dim']: float(di_minus.iloc[-1]) if not np.isnan(di_minus.iloc[-1]) else np.nan,
    }


def calculate_rogers_satchell_volatility(high: pd.Series, low: pd.Series, open_s: pd.Series, close: pd.Series,
                                         window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """Rogers–Satchell volatility estimator

    Apply on: open, high, low, close
    Returns: '{col}_rs_{window}'
    """
    key = f'{column_name}_rs_{window}'
    if len(close) < window:
        return {key: np.nan}
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (np.log(high / close) * np.log(high / open_s) +
                np.log(low / close) * np.log(low / open_s))
    rs_var = term.tail(window).mean()
    if np.isnan(rs_var) or rs_var < 0:
        return {key: np.nan}
    return {key: float(np.sqrt(rs_var))}


def calculate_yang_zhang_volatility(open_s: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                                    window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """Yang–Zhang volatility estimator combining overnight, intraday, and RS components

    Apply on: open, high, low, close
    Returns: '{col}_yz_{window}'
    """
    key = f'{column_name}_yz_{window}'
    if len(close) < window + 1:
        return {key: np.nan}

    with np.errstate(divide='ignore', invalid='ignore'):
        r_o = np.log(open_s / close.shift(1))
        r_c = np.log(close / open_s)
        rs_term = (np.log(high / close) * np.log(high / open_s) +
                   np.log(low / close) * np.log(low / open_s))

    r_o_w = r_o.tail(window)
    r_c_w = r_c.tail(window)
    rs_w = rs_term.tail(window)
    if len(r_o_w) < window or len(r_c_w) < window or len(rs_w) < window:
        return {key: np.nan}

    sigma_o2 = float(np.var(r_o_w, ddof=1))
    sigma_c2 = float(np.var(r_c_w, ddof=1))
    sigma_rs2 = float(rs_w.mean())

    if window <= 1:
        k = 0.34
    else:
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2
    if yz_var < 0:
        return {key: np.nan}
    return {key: float(np.sqrt(yz_var))}


def calculate_rvol(volume: pd.Series, window: int = 20, column_name: str = 'volume') -> Dict[str, float]:
    """Relative Volume: current volume vs. rolling mean volume

    Apply on: volume
    Returns: '{col}_rvol_{window}'
    """
    key = f'{column_name}_rvol_{window}'
    if len(volume) < window:
        return {key: np.nan}
    mean_vol = volume.tail(window).mean()
    if mean_vol == 0 or np.isnan(mean_vol):
        return {key: np.nan}
    return {key: float(volume.iloc[-1] / mean_vol)}


def calculate_donchian_distance(high: pd.Series, low: pd.Series, close: pd.Series,
                                window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """Donchian channel position and distances

    Apply on: high, low, close
    Returns: '{col}_donchian_pos_{w}', '{col}_donchian_upper_dist_{w}', '{col}_donchian_lower_dist_{w}'
    """
    keys = {
        'pos': f'{column_name}_donchian_pos_{window}',
        'ud': f'{column_name}_donchian_upper_dist_{window}',
        'ld': f'{column_name}_donchian_lower_dist_{window}',
    }
    if len(close) < window:
        return {keys['pos']: np.nan, keys['ud']: np.nan, keys['ld']: np.nan}
    highest = high.tail(window).max()
    lowest = low.tail(window).min()
    rng = highest - lowest
    if rng == 0:
        return {keys['pos']: 0.5, keys['ud']: 0.0, keys['ld']: 0.0}
    current = close.iloc[-1]
    pos = (current - lowest) / rng
    upper_dist = (highest - current) / rng
    lower_dist = (current - lowest) / rng
    return {keys['pos']: float(pos), keys['ud']: float(upper_dist), keys['ld']: float(lower_dist)}


def calculate_aroon(high: pd.Series, low: pd.Series, window: int = 14, column_name: str = 'close') -> Dict[str, float]:
    """Aroon Up/Down and oscillator

    Apply on: high, low
    Returns: '{col}_aroon_up_{w}', '{col}_aroon_down_{w}', '{col}_aroon_osc_{w}'
    """
    keys = {
        'up': f'{column_name}_aroon_up_{window}',
        'down': f'{column_name}_aroon_down_{window}',
        'osc': f'{column_name}_aroon_osc_{window}',
    }
    if len(high) < window or len(low) < window:
        return {keys['up']: np.nan, keys['down']: np.nan, keys['osc']: np.nan}
    h_w = high.tail(window).reset_index(drop=True)
    l_w = low.tail(window).reset_index(drop=True)
    idx_high = int(h_w.idxmax())
    idx_low = int(l_w.idxmin())
    periods_since_high = (window - 1) - idx_high
    periods_since_low = (window - 1) - idx_low
    aroon_up = 100 * (window - periods_since_high) / window
    aroon_down = 100 * (window - periods_since_low) / window
    osc = aroon_up - aroon_down
    return {keys['up']: float(aroon_up), keys['down']: float(aroon_down), keys['osc']: float(osc)}


def calculate_return_zscore(series: pd.Series, window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """Z-score of the latest log return over last `window` returns

    Apply on: close (or any price series)
    Returns: '{col}_ret_zscore_{w}'
    """
    key = f'{column_name}_ret_zscore_{window}'
    if len(series) < window + 1:
        return {key: np.nan}
    returns = np.log(series / series.shift(1)).dropna()
    r_window = returns.tail(window)
    mean = r_window.mean()
    std = r_window.std()
    if std == 0 or np.isnan(std):
        return {key: np.nan}
    z = (returns.iloc[-1] - mean) / std
    return {key: float(z)}


def calculate_atr_normalized_distance(current_value: float, ref_value: float, atr_value: float,
                                      column_name: str = 'close', ref_label: str = 'ref') -> Dict[str, float]:
    """Distance between current and reference value normalized by ATR

    Apply on: any scalar values with external ATR
    Returns: '{col}_dist_{label}_atr'
    """
    key = f'{column_name}_dist_{ref_label}_atr'
    if atr_value <= 0 or np.isnan(atr_value) or np.isnan(current_value) or np.isnan(ref_value):
        return {key: np.nan}
    return {key: float((current_value - ref_value) / atr_value)}


# =============================================================================
# ADVANCED MATH/LIQUIDITY/STAT FEATURES (low overlap with classic TA)
# =============================================================================

def calculate_roll_spread(close: pd.Series, window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """Roll spread estimate using negative first-order autocovariance of price changes

    Apply on: close
    Returns: '{col}_roll_spread_{window}' (NaN if cov >= 0)
    """
    key = f'{column_name}_roll_spread_{window}'
    if len(close) < window + 1:
        return {key: np.nan}
    dp = close.diff().dropna().tail(window)
    if len(dp) < 2:
        return {key: np.nan}
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    if cov >= 0 or np.isnan(cov):
        return {key: np.nan}
    spread = 2.0 * np.sqrt(-cov)
    return {key: float(spread)}


def calculate_amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """Amihud illiquidity: mean(|return| / dollar_volume) over window

    Apply on: close, volume
    Returns: '{col}_amihud_{window}'
    """
    key = f'{column_name}_amihud_{window}'
    if len(close) < window + 1:
        return {key: np.nan}
    ret = np.log(close / close.shift(1)).abs().dropna().tail(window)
    dollar_vol = (close * volume).dropna().tail(window)
    if len(ret) < window or len(dollar_vol) < window:
        return {key: np.nan}
    div = ret / dollar_vol.replace(0, np.nan)
    value = float(div.mean()) if div.notna().any() else np.nan
    return {key: value}


def calculate_turnover_zscore(close: pd.Series, volume: pd.Series, window: int = 20, column_name: str = 'turnover') -> Dict[str, float]:
    """Turnover z-score: z of (close*volume) over window

    Apply on: close, volume
    Returns: '{col}_z_{window}' where col defaults to 'turnover'
    """
    key = f'{column_name}_z_{window}'
    if len(close) < window or len(volume) < window:
        return {key: np.nan}
    turnover = (close * volume).tail(window)
    mean = turnover.mean()
    std = turnover.std()
    if std == 0 or np.isnan(std):
        return {key: np.nan}
    z = (turnover.iloc[-1] - mean) / std
    return {key: float(z)}


def calculate_ljung_box_pvalue(series: pd.Series, lags: int = 5, window: int = 100, column_name: str = 'close') -> Dict[str, float]:
    """Ljung–Box test p-value on recent returns (null: no autocorrelation up to lags)

    Apply on: close (or any price series)
    Returns: '{col}_ljung_p_{lags}_{window}'
    """
    from scipy.stats import chi2
    key = f'{column_name}_ljung_p_{lags}_{window}'
    if len(series) < window + 1:
        return {key: np.nan}
    r = np.log(series / series.shift(1)).dropna().tail(window)
    n = len(r)
    if n <= lags:
        return {key: np.nan}
    # autocorrelation estimates
    r_mean = r.mean()
    acf = []
    arr = r - r_mean
    denom = np.sum(arr**2)
    for k in range(1, lags + 1):
        num = np.sum(arr[k:] * arr[:-k])
        acf.append(num / denom)
    Q = n * (n + 2) * np.sum([(acf[k - 1] ** 2) / (n - k) for k in range(1, lags + 1)])
    pval = float(chi2.sf(Q, df=lags))
    return {key: pval}


def calculate_permutation_entropy(series: pd.Series, window: int = 50, m: int = 3, column_name: str = 'close') -> Dict[str, float]:
    """Permutation entropy (ordinal pattern entropy) of recent prices

    Apply on: any price series
    Returns: '{col}_perm_entropy_{m}_{window}' (normalized to [0,1])
    """
    key = f'{column_name}_perm_entropy_{m}_{window}'
    if len(series) < window:
        return {key: np.nan}
    x = series.tail(window).values
    n = len(x)
    if n < m:
        return {key: np.nan}
    # build ordinal patterns
    patterns = {}
    for i in range(n - m + 1):
        pattern = tuple(np.argsort(x[i:i + m]))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    return {key: float(ent)}


def calculate_ou_half_life(series: pd.Series, window: int = 100, column_name: str = 'close') -> Dict[str, float]:
    """Ornstein–Uhlenbeck half-life estimated via Δp_t = β p_{t-1} + ε over window

    Apply on: any price series
    Returns: '{col}_ou_halflife_{window}' (NaN if β >= 0)
    """
    key = f'{column_name}_ou_halflife_{window}'
    if len(series) < window + 1:
        return {key: np.nan}
    p = series.tail(window + 1).values
    dp = np.diff(p)
    x = p[:-1]
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return {key: np.nan}
    beta = float(np.dot(x, dp) / denom)
    if beta >= 0:
        return {key: np.nan}
    halflife = float(-np.log(2) / beta)
    return {key: halflife}


def calculate_var_cvar(series: pd.Series, window: int = 50, alpha: float = 0.05, column_name: str = 'close') -> Dict[str, float]:
    """Historical VaR and CVaR of returns over window

    Apply on: close (or any price series)
    Returns: '{col}_var_{pct}_{w}', '{col}_cvar_{pct}_{w}'
    """
    pct = int(alpha * 100)
    var_key = f'{column_name}_var_{pct}_{window}'
    cvar_key = f'{column_name}_cvar_{pct}_{window}'
    if len(series) < window + 1:
        return {var_key: np.nan, cvar_key: np.nan}
    r = np.log(series / series.shift(1)).dropna().tail(window)
    if len(r) < window:
        return {var_key: np.nan, cvar_key: np.nan}
    var = float(np.quantile(r, alpha))
    tail = r[r <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else np.nan
    return {var_key: var, cvar_key: cvar}


def calculate_spectral_entropy(series: pd.Series, window: int = 50, column_name: str = 'close') -> Dict[str, float]:
    """Spectral entropy of detrended recent prices using FFT power spectrum

    Apply on: any price series
    Returns: '{col}_spectral_entropy_{window}' in [0,1]
    """
    key = f'{column_name}_spectral_entropy_{window}'
    if len(series) < window:
        return {key: np.nan}
    x = series.tail(window).values
    x = x - x.mean()
    fft_vals = np.abs(np.fft.rfft(x))
    power = fft_vals**2
    if power.sum() == 0:
        return {key: np.nan}
    p = power / power.sum()
    ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    return {key: float(ent)}


# =============================================================================
# NORMALIZED / STATIONARY FEATURE VARIANTS (new)
# =============================================================================

def calculate_price_ema_ratios(series: pd.Series, span: int = 12, column_name: str = 'close') -> Dict[str, float]:
    """Price vs EMA ratios for stationarity

    Returns two dimensionless features:
      - '{col}_over_ema_{span}' = close / EMA(span)
      - '{col}_log_ratio_ema_{span}' = log(close / EMA(span))
    """
    ratio_key = f"{column_name}_over_ema_{span}"
    log_key = f"{column_name}_log_ratio_ema_{span}"
    if len(series) < span:
        return {ratio_key: np.nan, log_key: np.nan}
    ema_val = series.ewm(span=span).mean().iloc[-1]
    last = series.iloc[-1]
    if np.isnan(ema_val) or ema_val == 0 or np.isnan(last) or last <= 0:
        return {ratio_key: np.nan, log_key: np.nan}
    ratio = float(last / ema_val)
    log_ratio = float(np.log(ratio)) if ratio > 0 else np.nan
    return {ratio_key: ratio, log_key: log_ratio}


def calculate_price_ema_atr_distance(high: pd.Series, low: pd.Series, close: pd.Series,
                                     span: int = 12, atr_window: int = 14,
                                     column_name: str = 'close') -> Dict[str, float]:
    """ATR-normalized distance between price and its EMA(span)

    Returns: '{col}_dist_ema{span}_atr' = (close - EMA(span)) / ATR(atr_window)
    """
    if len(close) < max(span, atr_window) or len(high) < atr_window or len(low) < atr_window:
        return {f"{column_name}_dist_ema{span}_atr": np.nan}
    ema_val = close.ewm(span=span).mean().iloc[-1]
    atr_key = f"{column_name}_atr_{atr_window}"
    atr_val = calculate_atr(high, low, close, atr_window, column_name).get(atr_key, np.nan)
    if np.isnan(ema_val) or np.isnan(atr_val) or atr_val <= 0 or len(close) == 0:
        return {f"{column_name}_dist_ema{span}_atr": np.nan}
    current = float(close.iloc[-1])
    out = calculate_atr_normalized_distance(current, float(ema_val), float(atr_val), column_name, f"ema{span}")
    # Key is '{col}_dist_ema{span}_atr'
    return out


def calculate_macd_normalized_by_close(series: pd.Series, fast: int = 12, slow: int = 26,
                                       signal: int = 9, column_name: str = 'close') -> Dict[str, float]:
    """Normalize MACD components by current close to reduce scale effects

    Returns:
      - '{col}_macd_line_{fast}_{slow}_over_close'
      - '{col}_macd_histogram_{fast}_{slow}_{signal}_over_close'
    """
    if len(series) < max(fast, slow, signal) or len(series) == 0:
        return {
            f"{column_name}_macd_line_{fast}_{slow}_over_close": np.nan,
            f"{column_name}_macd_histogram_{fast}_{slow}_{signal}_over_close": np.nan,
        }
    macd = calculate_macd(series, fast, slow, signal, column_name)
    close_val = series.iloc[-1]
    if close_val == 0 or np.isnan(close_val):
        return {
            f"{column_name}_macd_line_{fast}_{slow}_over_close": np.nan,
            f"{column_name}_macd_histogram_{fast}_{slow}_{signal}_over_close": np.nan,
        }
    line_key = f"{column_name}_macd_line_{fast}_{slow}"
    hist_key = f"{column_name}_macd_histogram_{fast}_{slow}_{signal}"
    line = macd.get(line_key, np.nan)
    hist = macd.get(hist_key, np.nan)
    out = {
        f"{column_name}_macd_line_{fast}_{slow}_over_close": (float(line) / float(close_val)) if not np.isnan(line) else np.nan,
        f"{column_name}_macd_histogram_{fast}_{slow}_{signal}_over_close": (float(hist) / float(close_val)) if not np.isnan(hist) else np.nan,
    }
    return out


def calculate_macd_over_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                             fast: int = 12, slow: int = 26, signal: int = 9,
                             atr_window: int = 14, column_name: str = 'close') -> Dict[str, float]:
    """Normalize MACD components by ATR to make them regime invariant

    Returns:
      - '{col}_macd_line_{fast}_{slow}_over_atr{atr_window}'
      - '{col}_macd_histogram_{fast}_{slow}_{signal}_over_atr{atr_window}'
    """
    if len(close) < max(fast, slow, signal) or len(close) < atr_window or len(high) < atr_window or len(low) < atr_window:
        return {
            f"{column_name}_macd_line_{fast}_{slow}_over_atr{atr_window}": np.nan,
            f"{column_name}_macd_histogram_{fast}_{slow}_{signal}_over_atr{atr_window}": np.nan,
        }
    macd = calculate_macd(close, fast, slow, signal, column_name)
    atr_key = f"{column_name}_atr_{atr_window}"
    atr_val = calculate_atr(high, low, close, atr_window, column_name).get(atr_key, np.nan)
    if np.isnan(atr_val) or atr_val <= 0:
        return {
            f"{column_name}_macd_line_{fast}_{slow}_over_atr{atr_window}": np.nan,
            f"{column_name}_macd_histogram_{fast}_{slow}_{signal}_over_atr{atr_window}": np.nan,
        }
    line_key = f"{column_name}_macd_line_{fast}_{slow}"
    hist_key = f"{column_name}_macd_histogram_{fast}_{slow}_{signal}"
    line = macd.get(line_key, np.nan)
    hist = macd.get(hist_key, np.nan)
    out = {
        f"{column_name}_macd_line_{fast}_{slow}_over_atr{atr_window}": (float(line) / float(atr_val)) if not np.isnan(line) else np.nan,
        f"{column_name}_macd_histogram_{fast}_{slow}_{signal}_over_atr{atr_window}": (float(hist) / float(atr_val)) if not np.isnan(hist) else np.nan,
    }
    return out


def calculate_bollinger_width_pct(series: pd.Series, window: int = 20, num_std: float = 2.0,
                                  column_name: str = 'close') -> Dict[str, float]:
    """Bollinger width as a fraction of the middle band or price

    Returns: '{col}_bb_width_pct_{window}_{std}' = width / middle
    """
    def _fmt_std(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else str(x).replace('.', '_')
    std_tag = _fmt_std(num_std)
    key = f"{column_name}_bb_width_pct_{window}_{std_tag}"
    if len(series) < window:
        return {key: np.nan}
    bb = calculate_bollinger_bands(series, window, num_std, column_name)
    middle = bb.get(f"{column_name}_bb_middle_{window}", np.nan)
    width = bb.get(f"{column_name}_bb_width_{window}_{std_tag}", np.nan)
    if np.isnan(middle) or middle == 0 or np.isnan(width):
        return {key: np.nan}
    return {key: float(width) / float(middle)}


def calculate_obv_over_dollar_vol(close: pd.Series, volume: pd.Series, window: int = 20,
                                   column_name: str = 'close') -> Dict[str, float]:
    """OBV normalized by rolling dollar volume sum to reduce scale dependence

    Returns: '{col}_obv_over_dollar_vol_{window}'
    """
    key = f"{column_name}_obv_over_dollar_vol_{window}"
    if len(close) < 2 or len(close) < window or len(volume) < window:
        return {key: np.nan}
    obv_val = calculate_obv(close, volume, column_name).get(f"{column_name}_obv", np.nan)
    dv = (close * volume).tail(window)
    denom = float(dv.sum()) if len(dv) == window else np.nan
    if np.isnan(obv_val) or denom == 0 or np.isnan(denom):
        return {key: np.nan}
    return {key: float(obv_val) / denom}


def calculate_adl_over_dollar_vol(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                                   window: int = 20, column_name: str = 'close') -> Dict[str, float]:
    """ADL normalized by rolling dollar volume sum

    Returns: '{col}_adl_over_dollar_vol_{window}'
    """
    key = f"{column_name}_adl_over_dollar_vol_{window}"
    if len(close) < 2 or len(close) < window or len(volume) < window:
        return {key: np.nan}
    adl_val = calculate_adl(high, low, close, volume, column_name).get(f"{column_name}_adl", np.nan)
    dv = (close * volume).tail(window)
    denom = float(dv.sum()) if len(dv) == window else np.nan
    if np.isnan(adl_val) or denom == 0 or np.isnan(denom):
        return {key: np.nan}
    return {key: float(adl_val) / denom}


def calculate_vwap_ratios(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                          column_name: str = 'close') -> Dict[str, float]:
    """Price vs VWAP ratios for stationarity

    Returns:
      - '{col}_over_vwap' = close / vwap
      - '{col}_log_ratio_vwap' = log(close / vwap)
    """
    ratio_key = f"{column_name}_over_vwap"
    log_key = f"{column_name}_log_ratio_vwap"
    if len(close) == 0:
        return {ratio_key: np.nan, log_key: np.nan}
    vwap_val = calculate_vwap(high, low, close, volume, column_name).get(f"{column_name}_vwap", np.nan)
    last = close.iloc[-1]
    if np.isnan(vwap_val) or vwap_val == 0 or np.isnan(last) or last <= 0:
        return {ratio_key: np.nan, log_key: np.nan}
    ratio = float(last / vwap_val)
    log_ratio = float(np.log(ratio)) if ratio > 0 else np.nan
    return {ratio_key: ratio, log_key: log_ratio}


def calculate_time_features_cyc(timestamp: pd.Timestamp) -> Dict[str, float]:
    """Cyclical encodings for time features (normalized)

    Returns: hour/day-of-week/month sin/cos pairs
      - 'time_hour_sin', 'time_hour_cos'
      - 'time_dow_sin', 'time_dow_cos'
      - 'time_month_sin', 'time_month_cos'
    """
    hour = timestamp.hour
    dow = timestamp.dayofweek
    month = timestamp.month
    two_pi = 2.0 * np.pi
    return {
        'time_hour_sin': float(np.sin(two_pi * hour / 24.0)),
        'time_hour_cos': float(np.cos(two_pi * hour / 24.0)),
        'time_dow_sin': float(np.sin(two_pi * dow / 7.0)),
        'time_dow_cos': float(np.cos(two_pi * dow / 7.0)),
        'time_month_sin': float(np.sin(two_pi * month / 12.0)),
        'time_month_cos': float(np.cos(two_pi * month / 12.0)),
    }
