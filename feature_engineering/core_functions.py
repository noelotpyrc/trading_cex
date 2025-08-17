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


def calculate_price_differences(ohlcv: pd.DataFrame, lag: int = 0) -> Dict[str, float]:
    """2. Price Differences: Calculate price differences for a specific lag in OHLCV data"""
    if len(ohlcv) == 0:
        return {}
    
    # Check required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in ohlcv.columns for col in required_cols):
        return {
            'close_open_diff': np.nan,
            'high_low_range': np.nan,
            'price_change': np.nan
        }
    
    # Calculate the actual row index based on lag
    # lag 0 = current row (last), lag 1 = previous row, lag 2 = two back, etc.
    row_idx = -1 - lag
    
    # Check if the row exists
    if abs(row_idx) > len(ohlcv):
        return {
            'close_open_diff': np.nan,
            'high_low_range': np.nan,
            'price_change': np.nan
        }
    
    current_row = ohlcv.iloc[row_idx]
    
    # Create column names with lag terminology
    if lag == 0:
        row_suffix = "_current"
    else:
        row_suffix = f"_lag_{lag}"
    
    result = {
        f'close_open_diff{row_suffix}': current_row['close'] - current_row['open'],
        f'high_low_range{row_suffix}': current_row['high'] - current_row['low']
    }
    
    # Calculate close change for the specific row we're analyzing
    # Check if this row has a previous row to calculate close - close_previous
    # Since row_idx is always negative, we check if row_idx - 1 is within bounds
    if abs(row_idx - 1) <= len(ohlcv):
        prev_row = ohlcv.iloc[row_idx - 1]  # Move backward one row (e.g., -2 - 1 = -3)
        result[f'close_change{row_suffix}'] = current_row['close'] - prev_row['close']
    else:
        result[f'close_change{row_suffix}'] = np.nan
    
    return result


def calculate_log_transforms(ohlcv: pd.DataFrame, columns: List[str] = ['open', 'high', 'low', 'close', 'volume']) -> Dict[str, float]:
    """3. Log Transformations: log(columns) for specified columns"""
    if len(ohlcv) == 0:
        return {}
    
    current = ohlcv.iloc[-1]
    result = {}
    
    for col in columns:
        if col in ohlcv.columns and current[col] > 0:
            result[f'log_{col}'] = np.log(current[col])
        else:
            result[f'log_{col}'] = np.nan
    
    return result


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


def calculate_cumulative_returns(data: pd.Series, windows: List[int]) -> Dict[str, float]:
    """5. Cumulative Returns: Sum of log returns over rolling windows"""
    result = {}
    
    if len(data) < 2:
        for window in windows:
            result[f'cum_return_{window}'] = np.nan
        return result
    
    log_returns = np.log(data / data.shift(1)).dropna()
    
    for window in windows:
        if len(log_returns) >= window:
            result[f'cum_return_{window}'] = log_returns.tail(window).sum()
        else:
            result[f'cum_return_{window}'] = np.nan
    
    return result


def calculate_zscore(data: pd.Series, window: int) -> float:
    """6. Z-Scores: (price - rolling_mean) / rolling_std"""
    if len(data) < window:
        return np.nan
    
    rolling_data = data.tail(window)
    mean = rolling_data.mean()
    std = rolling_data.std()
    
    if std == 0:
        return np.nan
    
    return (data.iloc[-1] - mean) / std


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

def calculate_sma(data: pd.Series, window: int) -> float:
    """9. Simple Moving Average"""
    if len(data) < window:
        return np.nan
    return data.rolling(window=window).mean().iloc[-1]


def calculate_ema(data: pd.Series, span: int) -> float:
    """10. Exponential Moving Average"""
    if len(data) < span:
        return np.nan
    return data.ewm(span=span).mean().iloc[-1]


def calculate_wma(data: pd.Series, window: int) -> float:
    """11. Weighted Moving Average"""
    if len(data) < window:
        return np.nan
    
    weights = np.arange(1, window + 1)
    values = data.tail(window).values
    return np.sum(weights * values) / np.sum(weights)


def calculate_ma_crossovers(data: pd.Series, fast_window: int, slow_window: int) -> Dict[str, float]:
    """12. Moving Average Crossovers"""
    if len(data) < max(fast_window, slow_window):
        return {
            'ma_cross_diff': np.nan,
            'ma_cross_ratio': np.nan,
            'ma_cross_signal': np.nan
        }
    
    ma_fast = calculate_sma(data, fast_window)
    ma_slow = calculate_sma(data, slow_window)
    
    diff = ma_fast - ma_slow
    ratio = ma_fast / ma_slow if ma_slow != 0 else np.nan
    signal = 1 if ma_fast > ma_slow else 0
    
    return {
        'ma_cross_diff': diff,
        'ma_cross_ratio': ratio, 
        'ma_cross_signal': signal
    }


def calculate_ma_distance(price: float, ma_value: float) -> Dict[str, float]:
    """13. Distance to Moving Averages"""
    if np.isnan(ma_value) or ma_value == 0:
        return {'ma_distance': np.nan, 'ma_distance_pct': np.nan}
    
    distance = price - ma_value
    distance_pct = (distance / ma_value) * 100
    
    return {'ma_distance': distance, 'ma_distance_pct': distance_pct}


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """14. MACD: EMA_12 - EMA_26, Signal line, Histogram"""
    if len(data) < max(fast, slow, signal):
        return {
            'macd_line': np.nan,
            'macd_signal': np.nan,
            'macd_histogram': np.nan
        }
    
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd_line = (ema_fast - ema_slow).iloc[-1]
    
    macd_series = ema_fast - ema_slow
    if len(macd_series) >= signal:
        macd_signal = macd_series.ewm(span=signal).mean().iloc[-1]
        macd_histogram = macd_line - macd_signal
    else:
        macd_signal = np.nan
        macd_histogram = np.nan
    
    return {
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_histogram': macd_histogram
    }


def calculate_volume_ma(volume: pd.Series, window: int) -> float:
    """15. Moving Average of Volume"""
    return calculate_sma(volume, window)


# =============================================================================
# MOMENTUM AND OSCILLATOR FEATURES (Features 16-22)
# =============================================================================

def calculate_rsi(data: pd.Series, window: int = 14) -> float:
    """16. Relative Strength Index"""
    if len(data) < window + 1:
        return np.nan
    
    delta = data.diff().dropna()
    if len(delta) < window:
        return np.nan
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean().iloc[-1]
    avg_loss = loss.rolling(window=window).mean().iloc[-1]
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_window: int = 14, d_window: int = 3) -> Dict[str, float]:
    """17. Stochastic Oscillator: %K and %D"""
    if len(close) < k_window:
        return {'stoch_k': np.nan, 'stoch_d': np.nan}
    
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
    
    return {'stoch_k': k_percent, 'stoch_d': d_percent}


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> float:
    """18. Commodity Channel Index"""
    if len(close) < window:
        return np.nan
    
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean().iloc[-1]
    
    # Mean deviation
    rolling_tp = typical_price.tail(window)
    mean_deviation = np.mean(np.abs(rolling_tp - rolling_tp.mean()))
    
    if mean_deviation == 0:
        return np.nan
    
    cci = (typical_price.iloc[-1] - sma_tp) / (0.015 * mean_deviation)
    return cci


def calculate_roc(data: pd.Series, period: int) -> float:
    """19. Rate of Change"""
    if len(data) <= period:
        return np.nan
    
    current = data.iloc[-1]
    past = data.iloc[-1-period]
    
    if past == 0:
        return np.nan
    
    return ((current - past) / past) * 100


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    """20. Williams %R"""
    if len(close) < window:
        return np.nan
    
    highest_high = high.rolling(window=window).max().iloc[-1]
    lowest_low = low.rolling(window=window).min().iloc[-1]
    
    if highest_high == lowest_low:
        return -50.0
    
    williams_r = -100 * (highest_high - close.iloc[-1]) / (highest_high - lowest_low)
    return williams_r


def calculate_ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                                periods: List[int] = [7, 14, 28], 
                                weights: List[float] = [4, 2, 1]) -> float:
    """21. Ultimate Oscillator"""
    if len(close) < max(periods):
        return np.nan
    
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
            return np.nan
    
    uo = 100 * sum(averages) / sum(weights)
    return uo


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                 volume: pd.Series, window: int = 14) -> float:
    """22. Money Flow Index"""
    if len(close) < window + 1:
        return np.nan
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    price_change = typical_price.diff()
    positive_flow = money_flow.where(price_change > 0, 0)
    negative_flow = money_flow.where(price_change < 0, 0)
    
    positive_mf = positive_flow.rolling(window=window).sum().iloc[-1]
    negative_mf = negative_flow.rolling(window=window).sum().iloc[-1]
    
    if negative_mf == 0:
        return 100.0
    
    mfr = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfr))
    return mfi


# =============================================================================
# VOLATILITY FEATURES (Features 23-28)
# =============================================================================

def calculate_historical_volatility(data: pd.Series, window: int = 20) -> float:
    """23. Historical Volatility: std of log returns"""
    if len(data) < window + 1:
        return np.nan
    
    returns = np.log(data / data.shift(1)).dropna()
    if len(returns) < window:
        return np.nan
    
    return returns.tail(window).std() * np.sqrt(252)  # Annualized


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    """24. Average True Range"""
    if len(close) < window + 1:
        return np.nan
    
    tr_list = []
    for i in range(1, len(close)):
        h = high.iloc[i]
        l = low.iloc[i]
        c_prev = close.iloc[i-1]
        
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    
    if len(tr_list) < window:
        return np.nan
    
    return np.mean(tr_list[-window:])


def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, float]:
    """25. Bollinger Bands: Upper, Lower, Width, %B"""
    if len(data) < window:
        return {
            'bb_upper': np.nan, 'bb_middle': np.nan, 'bb_lower': np.nan,
            'bb_width': np.nan, 'bb_percent': np.nan
        }
    
    rolling_data = data.tail(window)
    middle = rolling_data.mean()
    std = rolling_data.std()
    
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    width = upper - lower
    
    current_price = data.iloc[-1]
    if width == 0:
        bb_percent = 0.5
    else:
        bb_percent = (current_price - lower) / width
    
    return {
        'bb_upper': upper, 'bb_middle': middle, 'bb_lower': lower,
        'bb_width': width, 'bb_percent': bb_percent
    }


def calculate_volatility_ratio(data: pd.Series, short_window: int = 5, long_window: int = 50) -> float:
    """26. Volatility Ratios: short-term vol / long-term vol"""
    if len(data) < long_window + 1:
        return np.nan
    
    returns = np.log(data / data.shift(1)).dropna()
    
    if len(returns) < long_window:
        return np.nan
    
    short_vol = returns.tail(short_window).std()
    long_vol = returns.tail(long_window).std()
    
    if long_vol == 0:
        return np.nan
    
    return short_vol / long_vol


def calculate_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> float:
    """27. Parkinson Volatility: based on High-Low ranges"""
    if len(high) < window:
        return np.nan
    
    log_hl_ratio = np.log(high / low)
    parkinson_values = log_hl_ratio ** 2
    
    if len(parkinson_values) < window:
        return np.nan
    
    return np.sqrt((1 / (4 * np.log(2))) * parkinson_values.tail(window).mean())


def calculate_garman_klass_volatility(ohlc: pd.DataFrame, window: int = 20) -> float:
    """28. Garman-Klass Volatility: incorporates OHLC"""
    if len(ohlc) < window:
        return np.nan
    
    high, low, open_, close = ohlc['high'], ohlc['low'], ohlc['open'], ohlc['close']
    
    term1 = 0.5 * (np.log(high / low)) ** 2
    term2 = (2 * np.log(2) - 1) * (np.log(close / open_)) ** 2
    
    gk_values = term1 - term2
    
    if len(gk_values) < window:
        return np.nan
    
    return np.sqrt(gk_values.tail(window).mean())


# =============================================================================
# VOLUME-INTEGRATED FEATURES (Features 29-33)
# =============================================================================

def calculate_obv(close: pd.Series, volume: pd.Series) -> float:
    """29. On-Balance Volume"""
    if len(close) < 2:
        return np.nan
    
    obv = 0
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv += volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv -= volume.iloc[i]
    
    return obv


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
    """30. Volume Weighted Average Price"""
    if len(close) == 0:
        return np.nan
    
    typical_price = (high + low + close) / 3
    total_volume_price = (typical_price * volume).sum()
    total_volume = volume.sum()
    
    if total_volume == 0:
        return np.nan
    
    return total_volume_price / total_volume


def calculate_adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
    """31. Accumulation/Distribution Line"""
    if len(close) == 0:
        return np.nan
    
    adl = 0
    for i in range(len(close)):
        if high.iloc[i] != low.iloc[i]:
            clv = ((close.iloc[i] - low.iloc[i]) - (high.iloc[i] - close.iloc[i])) / (high.iloc[i] - low.iloc[i])
            adl += clv * volume.iloc[i]
    
    return adl


def calculate_chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                               volume: pd.Series, fast: int = 3, slow: int = 10) -> float:
    """32. Chaikin Oscillator: EMA_3 - EMA_10 of ADL"""
    if len(close) < max(fast, slow):
        return np.nan
    
    # Calculate ADL series
    adl_series = []
    adl_cumulative = 0
    
    for i in range(len(close)):
        if high.iloc[i] != low.iloc[i]:
            clv = ((close.iloc[i] - low.iloc[i]) - (high.iloc[i] - close.iloc[i])) / (high.iloc[i] - low.iloc[i])
            adl_cumulative += clv * volume.iloc[i]
        adl_series.append(adl_cumulative)
    
    adl_pd = pd.Series(adl_series)
    
    if len(adl_pd) < max(fast, slow):
        return np.nan
    
    ema_fast = adl_pd.ewm(span=fast).mean().iloc[-1]
    ema_slow = adl_pd.ewm(span=slow).mean().iloc[-1]
    
    return ema_fast - ema_slow


def calculate_volume_roc(volume: pd.Series, period: int) -> float:
    """33. Volume Rate of Change"""
    return calculate_roc(volume, period)


# =============================================================================
# STATISTICAL AND DISTRIBUTIONAL FEATURES (Features 34-38)
# =============================================================================

def calculate_rolling_percentiles(data: pd.Series, window: int, 
                                percentiles: List[float] = [25, 50, 75]) -> Dict[str, float]:
    """34. Rolling Medians and Percentiles"""
    if len(data) < window:
        return {f'percentile_{p}': np.nan for p in percentiles}
    
    rolling_data = data.tail(window)
    return {f'percentile_{p}': np.percentile(rolling_data, p) for p in percentiles}


def calculate_distribution_features(data: pd.Series, window: int = 30) -> Dict[str, float]:
    """35. Kurtosis and Skewness of returns"""
    if len(data) < window + 1:
        return {'skewness': np.nan, 'kurtosis': np.nan}
    
    returns = data.pct_change().dropna()
    if len(returns) < window:
        return {'skewness': np.nan, 'kurtosis': np.nan}
    
    rolling_returns = returns.tail(window)
    return {
        'skewness': stats.skew(rolling_returns),
        'kurtosis': stats.kurtosis(rolling_returns)
    }


def calculate_autocorrelation(data: pd.Series, lag: int = 1, window: int = 30) -> float:
    """36. Autocorrelation of returns"""
    if len(data) < window + lag + 1:
        return np.nan
    
    returns = data.pct_change().dropna()
    if len(returns) < window + lag:
        return np.nan
    
    rolling_returns = returns.tail(window + lag)
    return rolling_returns.autocorr(lag=lag)


def calculate_hurst_exponent(data: pd.Series, window: int = 100) -> float:
    """37. Hurst Exponent via rescaled range analysis"""
    if len(data) < window:
        return np.nan
    
    try:
        log_returns = np.log(data / data.shift(1)).dropna().tail(window)
        if len(log_returns) < 10:
            return np.nan
        
        lags = range(2, min(20, len(log_returns) // 2))
        rs_values = []
        
        for lag in lags:
            # Calculate R/S statistic
            Y = log_returns.values
            mean_Y = np.mean(Y)
            
            # Cumulative deviations
            Z = np.cumsum(Y - mean_Y)
            R = np.max(Z) - np.min(Z)  # Range
            S = np.std(Y)  # Standard deviation
            
            if S > 0:
                rs_values.append(R / S)
        
        if len(rs_values) < 3:
            return np.nan
        
        # Linear regression on log(lag) vs log(R/S)
        log_lags = np.log(list(lags[:len(rs_values)]))
        log_rs = np.log(rs_values)
        
        hurst = np.polyfit(log_lags, log_rs, 1)[0]
        return hurst
        
    except:
        return np.nan


def calculate_entropy(data: pd.Series, window: int = 20) -> float:
    """38. Approximate entropy of price series"""
    if len(data) < window:
        return np.nan
    
    try:
        rolling_data = data.tail(window).values
        
        # Simple binning approach for entropy
        n_bins = min(10, len(rolling_data) // 2)
        hist, _ = np.histogram(rolling_data, bins=n_bins)
        
        # Normalize to probabilities
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))  # Add small value to avoid log(0)
        return entropy
        
    except:
        return np.nan


# =============================================================================
# RATIO AND HYBRID FEATURES (Features 39-44)
# =============================================================================

def calculate_price_volume_ratios(ohlc: pd.DataFrame, volume: pd.Series) -> Dict[str, float]:
    """39. Price-to-Volume Ratios"""
    if len(ohlc) == 0 or volume.iloc[-1] == 0:
        return {'close_volume_ratio': np.nan, 'high_volume_ratio': np.nan}
    
    current = ohlc.iloc[-1]
    current_volume = volume.iloc[-1]
    
    return {
        'close_volume_ratio': current['close'] / current_volume,
        'high_volume_ratio': current['high'] / current_volume
    }


def calculate_candle_patterns(ohlc: pd.DataFrame) -> Dict[str, float]:
    """40-41. Candle Body and Shadow Ratios"""
    if len(ohlc) == 0:
        return {
            'body_ratio': np.nan, 'upper_shadow_ratio': np.nan, 
            'lower_shadow_ratio': np.nan
        }
    
    # Check required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in ohlc.columns for col in required_cols):
        return {
            'body_ratio': np.nan, 'upper_shadow_ratio': np.nan, 
            'lower_shadow_ratio': np.nan
        }
    
    current = ohlc.iloc[-1]
    o, h, l, c = current['open'], current['high'], current['low'], current['close']
    
    range_val = h - l
    if range_val == 0:
        return {
            'body_ratio': np.nan, 'upper_shadow_ratio': np.nan, 
            'lower_shadow_ratio': np.nan
        }
    
    body_ratio = abs(c - o) / range_val
    upper_shadow_ratio = (h - max(o, c)) / range_val
    lower_shadow_ratio = (min(o, c) - l) / range_val
    
    return {
        'body_ratio': body_ratio,
        'upper_shadow_ratio': upper_shadow_ratio,
        'lower_shadow_ratio': lower_shadow_ratio
    }


def calculate_typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """42. Typical Price: (High + Low + Close) / 3"""
    if len(close) == 0:
        return np.nan
    
    return (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3


def calculate_ohlc_average(ohlc: pd.DataFrame) -> float:
    """43. OHLC Average: (Open + High + Low + Close) / 4"""
    if len(ohlc) == 0:
        return np.nan
    
    current = ohlc.iloc[-1]
    return (current['open'] + current['high'] + current['low'] + current['close']) / 4


def calculate_volatility_adjusted_returns(data: pd.Series, atr_value: float) -> float:
    """44. Volatility-Adjusted Returns: log_return / sqrt(ATR)"""
    if len(data) < 2 or atr_value <= 0 or np.isnan(atr_value):
        return np.nan
    
    log_return = np.log(data.iloc[-1] / data.iloc[-2])
    return log_return / np.sqrt(atr_value)


# =============================================================================
# TIME-BASED AND CYCLICAL FEATURES (Features 45-48)
# =============================================================================

def calculate_time_features(timestamp: pd.Timestamp) -> Dict[str, float]:
    """45. Time of Day/Week features"""
    return {
        'hour_of_day': timestamp.hour,
        'day_of_week': timestamp.dayofweek,
        'day_of_month': timestamp.day,
        'month_of_year': timestamp.month
    }


def calculate_rolling_extremes(data: pd.Series, window: int = 10) -> Dict[str, float]:
    """46. Rolling Min/Max and position relative to range"""
    if len(data) < window:
        return {
            'rolling_min': np.nan, 'rolling_max': np.nan,
            'position_in_range': np.nan
        }
    
    rolling_data = data.tail(window)
    rolling_min = rolling_data.min()
    rolling_max = rolling_data.max()
    
    current_price = data.iloc[-1]
    
    if rolling_max == rolling_min:
        position_in_range = 0.5
    else:
        position_in_range = (current_price - rolling_min) / (rolling_max - rolling_min)
    
    return {
        'rolling_min': rolling_min,
        'rolling_max': rolling_max,
        'position_in_range': position_in_range
    }


def calculate_dominant_cycle(data: pd.Series, window: int = 50) -> Dict[str, float]:
    """47. Simple Fourier-based dominant cycle detection"""
    if len(data) < window:
        return {'dominant_cycle_length': np.nan, 'cycle_strength': np.nan}
    
    try:
        rolling_data = data.tail(window).values
        
        # Detrend the data
        detrended = rolling_data - np.mean(rolling_data)
        
        # Apply FFT
        fft_values = np.abs(fft(detrended))
        
        # Find dominant frequency (excluding DC component)
        freqs = np.arange(1, len(fft_values) // 2)
        dominant_freq_idx = np.argmax(fft_values[1:len(fft_values)//2]) + 1
        
        # Convert to cycle length
        dominant_cycle_length = len(rolling_data) / dominant_freq_idx
        cycle_strength = fft_values[dominant_freq_idx] / np.sum(fft_values[1:len(fft_values)//2])
        
        return {
            'dominant_cycle_length': dominant_cycle_length,
            'cycle_strength': cycle_strength
        }
    except:
        return {'dominant_cycle_length': np.nan, 'cycle_strength': np.nan}


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


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, window: int = 20) -> float:
    """52. Rolling Correlation between Close and Volume"""
    if len(series1) < window or len(series2) < window:
        return np.nan
    
    return series1.tail(window).corr(series2.tail(window))


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
# UTILITY FUNCTIONS
# =============================================================================

def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """Basic validation of OHLCV data"""
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_cols):
        return False
    
    if len(data) == 0:
        return False
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Safe division with default value for zero division"""
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return default
    return numerator / denominator