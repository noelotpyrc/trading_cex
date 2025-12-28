"""
Base primitive functions for feature engineering.

These are reusable calculation functions that return pd.Series or scalar values.
All functions work on 1H bar data.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# =============================================================================
# DEFAULT WINDOW PARAMS
# =============================================================================

FAST_WINDOW = 24       # 1 day
SLOW_WINDOW = 168      # 1 week
LONG_WINDOW = 720      # 1 month
ZSCORE_MIN_PERIODS = 168


# =============================================================================
# ROLLING STATS
# =============================================================================

def rolling_mean(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Rolling mean."""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods).mean()


def rolling_std(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Rolling standard deviation."""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods).std()


def rolling_corr(series1: pd.Series, series2: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Rolling correlation between two series."""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series1.rolling(window, min_periods=min_periods).corr(series2)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage change."""
    return series.pct_change(periods, fill_method=None)


# =============================================================================
# VOLATILITY
# =============================================================================

def parkinson_volatility(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Parkinson volatility estimator using high-low range.
    More efficient than close-to-close volatility.
    
    Formula: sqrt((1 / (4 * ln(2))) * mean((ln(High/Low))^2, window))
    """
    log_hl_ratio = np.log(high / low)
    parkinson_values = log_hl_ratio ** 2
    result = np.sqrt((1 / (4 * np.log(2))) * parkinson_values.rolling(window).mean())
    return result


def historical_volatility(close: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """
    Historical volatility (standard deviation of log returns).
    
    Args:
        close: Close price series
        window: Rolling window
        annualize: If True, multiply by sqrt(365) for crypto
    """
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(365 * 24)  # 1H bars, 365 days
    return vol


# =============================================================================
# INDICATORS
# =============================================================================

def rsi(close: pd.Series, window: int) -> pd.Series:
    """
    Relative Strength Index (Wilder's RSI).
    
    Formula: 100 - 100/(1 + AvgGain/AvgLoss)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder's smoothing (EMA with alpha = 1/window)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """
    Average Directional Index (Wilder).
    Measures trend strength regardless of direction.
    """
    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # True range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed values (Wilder's EMA)
    alpha = 1 / window
    rma_tr = tr.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    rma_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    rma_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    
    # Directional indicators
    di_plus = 100 * (rma_plus_dm / rma_tr)
    di_minus = 100 * (rma_minus_dm / rma_tr)
    
    # DX and ADX
    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).replace([np.inf, -np.inf], np.nan)
    adx_result = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    
    return adx_result


# =============================================================================
# NORMALIZATION
# =============================================================================

def zscore(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Rolling z-score: (value - rolling_mean) / rolling_std
    
    min_periods defaults to 94% of window to tolerate small gaps.
    """
    if min_periods is None:
        min_periods = int(window * 0.94)  # 94% of window required
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


# =============================================================================
# PRICE ANCHORS
# =============================================================================

def rolling_vwap(open_: pd.Series, high: pd.Series, low: pd.Series, 
                  close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """
    Rolling Volume Weighted Average Price using typical price.
    
    Formula: sum(typical_price * volume, window) / sum(volume, window)
    where typical_price = (open + high + low + close) / 4
    """
    typical_price = (open_ + high + low + close) / 4
    pv = typical_price * volume
    return pv.rolling(window).sum() / volume.rolling(window).sum()


def ema_distance(price: pd.Series, span: int) -> pd.Series:
    """
    Percentage distance from EMA.
    
    Formula: (price - EMA) / EMA
    """
    ema_val = ema(price, span)
    return (price - ema_val) / ema_val


# =============================================================================
# TIME ENCODING
# =============================================================================

def cyclical_encode(values: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    """
    Cyclical encoding using sin/cos.
    
    Args:
        values: Series of values (e.g., day of week 0-6)
        period: Period of cycle (e.g., 7 for days, 12 for months)
    
    Returns:
        Tuple of (sin_encoded, cos_encoded)
    """
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def time_features(timestamp: pd.Series) -> pd.DataFrame:
    """
    Generate cyclical time features from timestamp.
    
    Returns DataFrame with sin/cos encodings for:
    - day_of_week (period=7)
    - month_of_year (period=12)
    - week_of_month (period=5)
    """
    dt = pd.to_datetime(timestamp)
    
    dow = dt.dt.dayofweek
    month = dt.dt.month
    week_of_month = ((dt.dt.day - 1) // 7) + 1
    
    dow_sin, dow_cos = cyclical_encode(dow, 7)
    month_sin, month_cos = cyclical_encode(month, 12)
    wom_sin, wom_cos = cyclical_encode(week_of_month, 5)
    
    return pd.DataFrame({
        'day_of_week_sin': dow_sin,
        'day_of_week_cos': dow_cos,
        'month_of_year_sin': month_sin,
        'month_of_year_cos': month_cos,
        'week_of_month_sin': wom_sin,
        'week_of_month_cos': wom_cos,
    })


# =============================================================================
# RISK METRICS
# =============================================================================

def var_cvar(returns: pd.Series, window: int, alpha: float = 0.05) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling Value at Risk and Conditional VaR (Expected Shortfall).
    
    Args:
        returns: Log returns series
        window: Rolling window
        alpha: Quantile (0.05 = 5% worst cases)
    
    Returns:
        Tuple of (VaR, CVaR)
    """
    var = returns.rolling(window).quantile(alpha)
    
    # CVaR: mean of returns below VaR
    def cvar_calc(x):
        threshold = np.quantile(x, alpha)
        tail = x[x <= threshold]
        return tail.mean() if len(tail) > 0 else np.nan
    
    cvar = returns.rolling(window).apply(cvar_calc, raw=True)
    
    return var, cvar


def linear_regression_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS slope (trend direction).
    """
    def slope(x):
        if len(x) < 2:
            return np.nan
        y = np.arange(len(x))
        try:
            coeffs = np.polyfit(y, x, 1)
            return coeffs[0]  # slope
        except:
            return np.nan
    
    return series.rolling(window).apply(slope, raw=True)

def kalman_filter_slope(series: pd.Series, 
                         process_variance: float = 1e-5,
                         measurement_variance: float = 1e-2) -> pd.Series:
    """
    Kalman filter to estimate slope (velocity) with minimal lag.
    
    Uses a 2-state model: [position, velocity]
    The velocity state gives a lag-minimized estimate of trend direction.
    
    Args:
        series: Price series
        process_variance: How much we expect the true state to change (Q)
        measurement_variance: How noisy our measurements are (R)
    
    Returns:
        Series of velocity (slope) estimates
    """
    n = len(series)
    values = series.values
    
    # State: [position, velocity]
    # Initialize
    x = np.array([values[0], 0.0])  # [position, velocity]
    P = np.array([[1.0, 0.0], [0.0, 1.0]])  # Covariance matrix
    
    # Process noise covariance (Q)
    Q = np.array([[process_variance, 0.0], [0.0, process_variance]])
    
    # Measurement noise (R)
    R = measurement_variance
    
    # State transition matrix: position = prev_position + velocity
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    
    # Measurement matrix (we only observe position)
    H = np.array([[1.0, 0.0]])
    
    velocities = np.full(n, np.nan)
    
    for i in range(1, n):
        if np.isnan(values[i]):
            velocities[i] = np.nan
            continue
            
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Update
        z = values[i]  # measurement
        y = z - H @ x_pred  # innovation
        S = H @ P_pred @ H.T + R  # innovation covariance
        K = P_pred @ H.T / S  # Kalman gain
        
        x = x_pred + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P_pred
        
        velocities[i] = x[1]  # velocity state
    
    return pd.Series(velocities, index=series.index)
