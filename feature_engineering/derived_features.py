"""
Derived feature functions organized by Physics Framework.

Model: r_{t+1} ≈ ρ_t * v_t - γ_t * d_t

Categories:
- State Variables: displacement, speed (price OHLC only)
- Momentum Persistence (ρ): signals that momentum continues
- Mean-Reversion Strength (γ): signals that reversion is likely
- Regime Indicators: trending vs choppy environment
- Interactions: cross-variable products
"""

import pandas as pd
import numpy as np

from .primitives import (
    FAST_WINDOW, SLOW_WINDOW, LONG_WINDOW, ZSCORE_MIN_PERIODS,
    rolling_mean, rolling_std, rolling_corr, ema, pct_change,
    parkinson_volatility, historical_volatility, rsi, adx, zscore,
    rolling_vwap, time_features, var_cvar, linear_regression_slope,
    kalman_filter_slope
)


# =============================================================================
# STATE VARIABLES (Price OHLC only)
# =============================================================================

# Displacement (distance from equilibrium)
def price_vwap_distance_zscore(close: pd.Series, volume: pd.Series, 
                                vwap_window: int, zscore_window: int) -> pd.Series:
    """Displacement: price distance from VWAP, z-scored"""
    vwap = rolling_vwap(close, volume, vwap_window)
    vwap_dist = (close - vwap) / vwap
    return zscore(vwap_dist, zscore_window)
# try vwap window of 24, 168 and 720, then zscore window always 168

# Speed / SNR (signal-to-noise)
def _ema_distance_zscore(series: pd.Series, ema_span: int, zscore_window: int) -> pd.Series:
    """Internal: Distance from EMA, z-scored"""
    ema_val = ema(series, ema_span)
    ema_dist = (series - ema_val) / ema_val
    return zscore(ema_dist, zscore_window)


def price_ema_distance_zscore(close: pd.Series, ema_span: int, zscore_window: int) -> pd.Series:
    """Speed/SNR: price distance from EMA, z-scored"""
    return _ema_distance_zscore(close, ema_span, zscore_window)
# try ema span of 24, then zscore window of 168 and 720

def price_roc_over_volatility(close: pd.Series, high: pd.Series, low: pd.Series, 
                               roc_window: int, vol_window: int) -> pd.Series:
    """Speed/SNR: Price ROC divided by Parkinson volatility"""
    price_roc = pct_change(close, roc_window)
    vol = parkinson_volatility(high, low, vol_window)
    return price_roc / vol.replace(0, np.nan)
# try roc window of 24, then vol window of 24, 168 and 720

# =============================================================================
# MOMENTUM PERSISTENCE (ρ signals)
# =============================================================================

# --- Price-based momentum ---

def return_autocorr(close: pd.Series, window: int) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(window).corr(r.shift(1))
# try window of 48 and 168

def run_fraction(close: pd.Series, window: int) -> pd.Series:
    r = np.log(close).diff()
    s = np.sign(r)

    def same_sign_frac(x: pd.Series) -> float:
        cur = x.iloc[-1]
        if pd.isna(cur) or cur == 0:
            return np.nan  # undefined if latest return is exactly 0
        # optionally ignore zeros in the window
        x_nz = x[x != 0]
        if len(x_nz) == 0:
            return np.nan
        return (x_nz == cur).mean()

    return s.rolling(window).apply(same_sign_frac, raw=False)
# try window of 24 and 168


def variance_ratio(close: pd.Series, q: int, window: int) -> pd.Series:
    y = np.log(close)
    r1 = y.diff(1)
    rq = y.diff(q)

    var_1 = r1.rolling(window).var()
    var_q = rq.rolling(window).var()

    denom = (q * var_1).where(var_1 > 0)
    return var_q / denom
# use q = 24 and window = 48, 168 and 720


# --- OI-based momentum ---

def oi_price_accel_product(oi: pd.Series, close: pd.Series, span: int) -> pd.Series:
    """Aligned OI and price accelerations (positive = momentum)"""
    oi_accel = scaled_acceleration(oi, span)
    price_accel = scaled_acceleration(close, span)
    return oi_accel * price_accel
# try span as 168

def oi_price_momentum(oi: pd.Series, close: pd.Series, span: int) -> pd.Series:
    """Product of OI and price EMA slopes"""
    oi_ema = ema(oi, span)
    price_ema = ema(close, span)
    oi_slope = pct_change(oi_ema)
    price_slope = pct_change(price_ema)
    return oi_slope * price_slope
# try span as 168

# --- Flow-based momentum ---

def cvd_slope(taker_buy: pd.Series, spot_volume: pd.Series, window: int) -> pd.Series:
    """Slope of cumulative volume delta (directional pressure)"""
    imbalance = taker_buy - (spot_volume - taker_buy)
    cvd = imbalance.cumsum()
    return (cvd - cvd.shift(window)) / window  # fast endpoint slope
# try window as 24, 168

def net_taker_volume_zscore(taker_buy: pd.Series, spot_volume: pd.Series, 
                             window: int) -> pd.Series:
    """Z-score of net taker volume (aggressive flow)"""
    imbalance = taker_buy - (spot_volume - taker_buy)
    return zscore(imbalance, window)
# try window as 168

def relative_volume(volume: pd.Series, timestamp: pd.Series, window: int) -> pd.Series:
    """Volume relative to same hour-of-day mean"""
    ts = pd.to_datetime(timestamp)
    hour = ts.dt.hour
    df = pd.DataFrame({'volume': volume, 'hour': hour})
    vol_by_hour = df.groupby('hour')['volume'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return volume / vol_by_hour.replace(0, np.nan)
# try window 7, 14, 30

def trade_count_lead_price_corr(num_trades: pd.Series, close: pd.Series,
                                 lead_periods: int, window: int) -> pd.Series:
    """Correlation between lagged trade count and price ROC"""
    tc_shifted_roc = pct_change(num_trades.shift(lead_periods))
    price_roc = pct_change(close)
    return rolling_corr(tc_shifted_roc, price_roc, window)
# try lead periods as 24 and window as 168

# =============================================================================
# MEAN-REVERSION STRENGTH (γ signals)
# =============================================================================

# --- Price-based mean reversion ---

def pullback_slope(close: pd.Series, ema_span: int, window: int) -> pd.Series:
    y = np.log(close)
    m = ema(y, ema_span)
    d = y - m
    r = y.diff(1)

    x = (-d.shift(1))  # predictor
    cov = x.rolling(window).cov(r)
    var = x.rolling(window).var()

    return cov / var.where(var > 0)
# try ema span as 24, 168 and 720, and window as 48 and 168


def mean_cross_rate(close: pd.Series, ema_span: int, window: int) -> pd.Series:
    y = np.log(close)
    m = ema(y, ema_span)
    d = y - m

    s = np.sign(d).replace(0, np.nan).ffill()  # avoid 0-sign jitter
    cross = (s != s.shift(1)).astype(float)

    # exact normalization by (window - 1)
    return cross.rolling(window).sum() / (window - 1)
# try ema span as 24, 168 and 720, and window as 48 and 168


# --- OI-based reversion signals ---

def oi_zscore(oi: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """OI z-score: extreme = stretched, may revert"""
    return zscore(oi, window, min_periods)
# try window as 168

def oi_ema_distance_zscore(oi: pd.Series, ema_span: int, zscore_window: int) -> pd.Series:
    """OI distance from EMA: extreme = stretched"""
    return _ema_distance_zscore(oi, ema_span, zscore_window)
# try ema span as 24, 168 and 720, and zscore window 168

# --- Premium-based reversion signals ---

def premium_zscore(premium: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Premium z-score: extreme = crowded, may revert"""
    return zscore(premium, window, min_periods)
# try window as 168

# --- L/S ratio reversion signals ---

def long_short_ratio_zscore(ls_ratio: pd.Series, window: int) -> pd.Series:
    """L/S ratio z-score: extreme = crowded, may revert"""
    return zscore(ls_ratio, window)
# try window as 48 and 168

# --- Spot reversion signals ---

def spot_vol_zscore(spot_volume: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Spot volume z-score"""
    return zscore(spot_volume, window, min_periods)
# try window as 168

def avg_trade_size_zscore(quote_volume: pd.Series, num_trades: pd.Series, 
                           window: int) -> pd.Series:
    """Average trade size z-score"""
    avg_size = quote_volume / num_trades.replace(0, np.nan)
    return zscore(avg_size, window)
# try window as 48 and 168

# --- Flow reversion signals ---

def imbalance_price_corr(taker_buy: pd.Series, spot_volume: pd.Series, 
                          close: pd.Series, window: int) -> pd.Series:
    """Imbalance-price correlation (exhaustion when weak)"""
    imbalance = taker_buy - (spot_volume - taker_buy)
    imb_z = zscore(imbalance, window)
    price_roc = pct_change(close)
    return rolling_corr(imb_z, price_roc, window)
# try window as 168

def trade_size_price_corr(quote_volume: pd.Series, num_trades: pd.Series,
                           close: pd.Series, window: int) -> pd.Series:
    """Trade size-price correlation"""
    avg_size = quote_volume / num_trades.replace(0, np.nan)
    avg_size_roc = pct_change(avg_size)
    price_roc = pct_change(close)
    return rolling_corr(avg_size_roc, price_roc, window)
# try window as 168

# =============================================================================
# REGIME INDICATORS
# =============================================================================

# --- Price-based regime ---

def efficiency_avg(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                   window: int, eps: float = 1e-12, min_periods: int | None = None) -> pd.Series:
    """Windowed efficiency: sum(|log(C/O)|) / sum(|log(H/L)|)."""
    if min_periods is None:
        min_periods = window

    log_range = (np.log(high) - np.log(low)).abs()
    log_net = (np.log(close) - np.log(open_)).abs()

    num = log_net.rolling(window, min_periods=min_periods).sum()
    den = log_range.rolling(window, min_periods=min_periods).sum().clip(lower=eps)

    eff = num / den
    return eff.clip(lower=0.0, upper=1.0)
# try window as 24 and 168


def vol_ratio(high: pd.Series, low: pd.Series, 
              fast_window: int, slow_window: int) -> pd.Series:
    """Volatility regime: fast/slow Parkinson ratio"""
    vol_fast = parkinson_volatility(high, low, fast_window)
    vol_slow = parkinson_volatility(high, low, slow_window)
    return vol_fast / vol_slow.replace(0, np.nan)
# try fast window as 24, and slow window as 168 and 720

def oi_volatility(oi: pd.Series, span: int) -> pd.Series:
    """OI volatility: EMA of sign(roc) * diff(roc)
    
    Measures magnitude of OI changes (like Parkinson but for OI).
    """
    roc = pct_change(oi)
    volatility = np.sign(roc) * roc.diff()
    return ema(volatility, span)
# try span as 168

def oi_vol_ratio(oi: pd.Series, fast_span: int, slow_span: int) -> pd.Series:
    """OI volatility regime: fast/slow ratio"""
    vol_fast = oi_volatility(oi, fast_span)
    vol_slow = oi_volatility(oi, slow_span)
    return vol_fast / vol_slow.replace(0, np.nan)
# try fast span as 24, and slow span as 168

def cvar_var_ratio(close: pd.Series, window: int, alpha: float = 0.05) -> pd.Series:
    """Tail thickness: CVaR/VaR ratio"""
    log_returns = np.log(close / close.shift(1))
    var_val, cvar_val = var_cvar(log_returns, window, alpha)
    return cvar_val / var_val.replace(0, np.nan)
# try window as 168 and 720

def tail_skewness(close: pd.Series, window: int) -> pd.Series:
    """Tail asymmetry: (VaR95 + VaR05) / (VaR95 - VaR05)"""
    log_returns = np.log(close / close.shift(1))
    var_95 = log_returns.rolling(window).quantile(0.95)
    var_05 = log_returns.rolling(window).quantile(0.05)
    return (var_95 + var_05) / (var_95 - var_05).replace(0, np.nan)
# try window as 168 and 720

# --- Premium regime ---

def premium_vol_ratio(premium: pd.Series, 
                       fast_window: int, slow_window: int) -> pd.Series:
    """Premium volatility regime: fast/slow ratio"""
    vol_fast = rolling_std(premium, fast_window)
    vol_slow = rolling_std(premium, slow_window)
    return vol_fast / vol_slow.replace(0, np.nan)
# try fast window as 24, and slow window as 48 and 168

# --- Spot regime ---

def spot_dominance(spot_volume: pd.Series, perp_volume: pd.Series) -> pd.Series:
    """Spot as fraction of total volume (non-stationary)"""
    total = spot_volume + perp_volume
    return spot_volume / total.replace(0, np.nan)


def spot_dominance_zscore(spot_volume: pd.Series, perp_volume: pd.Series, 
                           window: int) -> pd.Series:
    """Spot dominance z-score (stationary)"""
    spot_dom = spot_dominance(spot_volume, perp_volume)
    return zscore(spot_dom, window)
# try window as 168

def spot_dom_vol_ratio(spot_volume: pd.Series, perp_volume: pd.Series,
                        fast_window: int, slow_window: int) -> pd.Series:
    """Spot dominance volatility regime"""
    spot_dom = spot_dominance(spot_volume, perp_volume)
    vol_fast = rolling_std(spot_dom, fast_window)
    vol_slow = rolling_std(spot_dom, slow_window)
    return vol_fast / vol_slow.replace(0, np.nan)


# =============================================================================
# INTERACTIONS
# =============================================================================

# --- Core physics interactions ---

def displacement_speed_product(close: pd.Series, volume: pd.Series,
                               vwap_window: int, speed_window: int) -> pd.Series:
    """x_t * speed_t: x_t is log(C/VWAP), speed is k-step log return."""
    vwap = rolling_vwap(close, volume, vwap_window)

    x = np.log(close) - np.log(vwap)                 # displacement (log space)
    speed = np.log(close) - np.log(close.shift(speed_window))  # k-step speed

    return x * speed



def range_chop_interaction(high, low, open_, close, window: int) -> pd.Series:
    pv = parkinson_volatility(high, low, window)
    eff = efficiency_avg(open_, high, low, close, window).clip(0, 1)
    return pv * (1.0 - eff)


def range_stretch_interaction(high: pd.Series, low: pd.Series,
                              close: pd.Series, volume: pd.Series,
                              vwap_window: int, vol_window: int) -> pd.Series:
    """PV * |log(C/VWAP)|: big range and big stretch from equilibrium."""
    pv = parkinson_volatility(high, low, vol_window)
    vwap = rolling_vwap(close, volume, vwap_window)

    stretch = (np.log(close) - np.log(vwap)).abs()
    return pv * stretch



# --- Utility ---

def scaled_acceleration(series: pd.Series, span: int) -> pd.Series:
    """Scaled acceleration: accel / volatility"""
    roc = pct_change(series)
    accel = roc.diff()
    volatility = np.sign(roc) * roc.diff()
    
    accel_ema = ema(accel, span)
    vol_ema = ema(volatility, span)
    
    return accel_ema / vol_ema.replace(0, np.nan)


# --- OI-Price interactions ---

def oi_price_ratio_spread(oi: pd.Series, close: pd.Series, 
                           fast_span: int, slow_span: int) -> pd.Series:
    """Difference between OI and price EMA ratios"""
    oi_ratio = ema(oi, fast_span) / ema(oi, slow_span)
    price_ratio = ema(close, fast_span) / ema(close, slow_span)
    return oi_ratio - price_ratio


# --- Spot-Price interactions ---

def spot_vol_price_corr(spot_volume: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Spot volume-price correlation"""
    spot_roc = pct_change(spot_volume)
    price_roc = pct_change(close)
    return rolling_corr(spot_roc, price_roc, window)


def oi_vol_price_corr(oi: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """OI-price correlation"""
    oi_roc = pct_change(oi)
    price_roc = pct_change(close)
    return rolling_corr(oi_roc, price_roc, window)


def spot_dom_price_corr(spot_volume: pd.Series, perp_volume: pd.Series,
                         close: pd.Series, window: int) -> pd.Series:
    """Spot dominance-price correlation"""
    spot_dom = spot_dominance(spot_volume, perp_volume)
    spot_dom_roc = pct_change(spot_dom)
    price_roc = pct_change(close)
    return rolling_corr(spot_dom_roc, price_roc, window)


def spot_dom_oi_corr(spot_volume: pd.Series, perp_volume: pd.Series,
                      oi: pd.Series, window: int) -> pd.Series:
    """Spot dominance-OI correlation"""
    spot_dom = spot_dominance(spot_volume, perp_volume)
    spot_dom_roc = pct_change(spot_dom)
    oi_roc = pct_change(oi)
    return rolling_corr(spot_dom_roc, oi_roc, window)

# --- Trade interactions ---

def trade_count_oi_corr(num_trades: pd.Series, oi: pd.Series, window: int) -> pd.Series:
    """Trade count-OI correlation"""
    tc_roc = pct_change(num_trades)
    oi_roc = pct_change(oi)
    return rolling_corr(tc_roc, oi_roc, window)


def trade_count_spot_dom_corr(num_trades: pd.Series, spot_volume: pd.Series,
                               perp_volume: pd.Series, window: int) -> pd.Series:
    """Trade count-spot dominance correlation"""
    tc_roc = pct_change(num_trades)
    spot_dom = spot_dominance(spot_volume, perp_volume)
    spot_dom_roc = pct_change(spot_dom)
    return rolling_corr(tc_roc, spot_dom_roc, window)

