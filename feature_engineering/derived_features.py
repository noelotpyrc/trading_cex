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


# Speed / SNR (signal-to-noise)
def _ema_distance_zscore(series: pd.Series, ema_span: int, zscore_window: int) -> pd.Series:
    """Internal: Distance from EMA, z-scored"""
    ema_val = ema(series, ema_span)
    ema_dist = (series - ema_val) / ema_val
    return zscore(ema_dist, zscore_window)


def price_ema_distance_zscore(close: pd.Series, ema_span: int, zscore_window: int) -> pd.Series:
    """Speed/SNR: price distance from EMA, z-scored"""
    return _ema_distance_zscore(close, ema_span, zscore_window)


def price_roc_over_volatility(close: pd.Series, high: pd.Series, low: pd.Series, 
                               window: int) -> pd.Series:
    """Speed/SNR: Price ROC divided by Parkinson volatility"""
    price_roc = pct_change(close, window)
    vol = parkinson_volatility(high, low, window)
    return price_roc / vol.replace(0, np.nan)


# =============================================================================
# MOMENTUM PERSISTENCE (ρ signals)
# =============================================================================

# --- Price-based momentum ---

def return_autocorr(close: pd.Series, window: int) -> pd.Series:
    """Lag-1 return autocorrelation (positive = trending)"""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan, 
        raw=False
    )


def run_fraction(close: pd.Series, window: int) -> pd.Series:
    """Fraction of returns with same sign as current (near 1 = streak)"""
    log_returns = np.log(close / close.shift(1))
    sign = np.sign(log_returns)
    current_sign = sign
    
    def same_sign_frac(x):
        if len(x) < 2:
            return np.nan
        current = x.iloc[-1]
        return (np.sign(x) == current).mean()
    
    return log_returns.rolling(window).apply(same_sign_frac, raw=False)


def variance_ratio(close: pd.Series, q: int, window: int) -> pd.Series:
    """Variance ratio: VR > 1 = trending, VR < 1 = mean reverting"""
    log_price = np.log(close)
    r1 = log_price.diff(1)
    rq = log_price.diff(q)
    
    var_1 = r1.rolling(window).var()
    var_q = rq.rolling(window).var()
    
    return var_q / (q * var_1).replace(0, np.nan)


# --- OI-based momentum ---

def oi_price_accel_product(oi: pd.Series, close: pd.Series, span: int) -> pd.Series:
    """Aligned OI and price accelerations (positive = momentum)"""
    oi_accel = scaled_acceleration(oi, span)
    price_accel = scaled_acceleration(close, span)
    return oi_accel * price_accel


def oi_price_momentum(oi: pd.Series, close: pd.Series, span: int) -> pd.Series:
    """Product of OI and price EMA slopes"""
    oi_ema = ema(oi, span)
    price_ema = ema(close, span)
    oi_slope = pct_change(oi_ema)
    price_slope = pct_change(price_ema)
    return oi_slope * price_slope


# --- Flow-based momentum ---

def cvd_slope(taker_buy: pd.Series, spot_volume: pd.Series, window: int) -> pd.Series:
    """Slope of cumulative volume delta (directional pressure)"""
    imbalance = taker_buy - (spot_volume - taker_buy)
    cvd = imbalance.cumsum()
    return linear_regression_slope(cvd, window)


def net_taker_volume_zscore(taker_buy: pd.Series, spot_volume: pd.Series, 
                             window: int) -> pd.Series:
    """Z-score of net taker volume (aggressive flow)"""
    imbalance = taker_buy - (spot_volume - taker_buy)
    return zscore(imbalance, window)


def relative_volume(volume: pd.Series, timestamp: pd.Series, window: int) -> pd.Series:
    """Volume relative to same hour-of-day mean"""
    ts = pd.to_datetime(timestamp)
    hour = ts.dt.hour
    df = pd.DataFrame({'volume': volume, 'hour': hour})
    vol_by_hour = df.groupby('hour')['volume'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return volume / vol_by_hour.replace(0, np.nan)


def trade_count_lead_price_corr(num_trades: pd.Series, close: pd.Series,
                                 lead_periods: int, window: int) -> pd.Series:
    """Correlation between lagged trade count and price ROC"""
    tc_shifted_roc = pct_change(num_trades.shift(lead_periods))
    price_roc = pct_change(close)
    return rolling_corr(tc_shifted_roc, price_roc, window)


# =============================================================================
# MEAN-REVERSION STRENGTH (γ signals)
# =============================================================================

# --- Price-based mean reversion ---

def pullback_slope(close: pd.Series, ema_span: int, window: int) -> pd.Series:
    """Pullback strength: cov(-d, r) / var(d) (positive = strong reversion)"""
    log_p = np.log(close)
    m = ema(log_p, ema_span)
    d = log_p - m
    r = log_p.diff(1)
    
    neg_d_lag = -d.shift(1)
    cov = neg_d_lag.rolling(window).cov(r)
    var_d = neg_d_lag.rolling(window).var()
    
    return cov / var_d.replace(0, np.nan)


def mean_cross_rate(close: pd.Series, ema_span: int, window: int) -> pd.Series:
    """Rate of mean crossings (high = choppy/reverting)"""
    log_p = np.log(close)
    m = ema(log_p, ema_span)
    d = log_p - m
    
    sign_change = (np.sign(d) != np.sign(d.shift(1))).astype(float)
    return sign_change.rolling(window).mean()


# --- OI-based reversion signals ---

def oi_zscore(oi: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """OI z-score: extreme = stretched, may revert"""
    return zscore(oi, window, min_periods)


def oi_ema_distance_zscore(oi: pd.Series, ema_span: int, zscore_window: int) -> pd.Series:
    """OI distance from EMA: extreme = stretched"""
    return _ema_distance_zscore(oi, ema_span, zscore_window)


# --- Premium-based reversion signals ---

def premium_zscore(premium: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Premium z-score: extreme = crowded, may revert"""
    return zscore(premium, window, min_periods)


# --- L/S ratio reversion signals ---

def long_short_ratio_zscore(ls_ratio: pd.Series, window: int) -> pd.Series:
    """L/S ratio z-score: extreme = crowded, may revert"""
    return zscore(ls_ratio, window)


# --- Spot reversion signals ---

def spot_vol_zscore(spot_volume: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Spot volume z-score"""
    return zscore(spot_volume, window, min_periods)


# --- Flow reversion signals ---

def imbalance_price_corr(taker_buy: pd.Series, spot_volume: pd.Series, 
                          close: pd.Series, window: int) -> pd.Series:
    """Imbalance-price correlation (exhaustion when weak)"""
    imbalance = taker_buy - (spot_volume - taker_buy)
    imb_z = zscore(imbalance, window)
    price_roc = pct_change(close)
    return rolling_corr(imb_z, price_roc, window)


def trade_size_price_corr(quote_volume: pd.Series, num_trades: pd.Series,
                           close: pd.Series, window: int) -> pd.Series:
    """Trade size-price correlation"""
    avg_size = quote_volume / num_trades.replace(0, np.nan)
    avg_size_roc = pct_change(avg_size)
    price_roc = pct_change(close)
    return rolling_corr(avg_size_roc, price_roc, window)


# =============================================================================
# REGIME INDICATORS
# =============================================================================

# --- Price-based regime ---

def bar_efficiency(open_: pd.Series, high: pd.Series, low: pd.Series, 
                   close: pd.Series) -> pd.Series:
    """Per-bar efficiency: |net| / range (1 = trend, 0 = chop)"""
    log_range = np.abs(np.log(high / low))
    log_net = np.abs(np.log(close / open_))
    return log_net / log_range.clip(lower=1e-8)


def efficiency_avg(open_: pd.Series, high: pd.Series, low: pd.Series, 
                   close: pd.Series, window: int) -> pd.Series:
    """Windowed efficiency: sum(net) / sum(range)"""
    log_range = np.abs(np.log(high / low))
    log_net = np.abs(np.log(close / open_))
    return log_net.rolling(window).sum() / log_range.rolling(window).sum().clip(lower=1e-8)


def vol_ratio(high: pd.Series, low: pd.Series, 
              fast_window: int, slow_window: int) -> pd.Series:
    """Volatility regime: fast/slow Parkinson ratio"""
    vol_fast = parkinson_volatility(high, low, fast_window)
    vol_slow = parkinson_volatility(high, low, slow_window)
    return vol_fast / vol_slow.replace(0, np.nan)


def cvar_var_ratio(close: pd.Series, window: int, alpha: float = 0.05) -> pd.Series:
    """Tail thickness: CVaR/VaR ratio"""
    log_returns = np.log(close / close.shift(1))
    var_val, cvar_val = var_cvar(log_returns, window, alpha)
    return cvar_val / var_val.replace(0, np.nan)


def tail_skewness(close: pd.Series, window: int) -> pd.Series:
    """Tail asymmetry: (VaR95 + VaR05) / (VaR95 - VaR05)"""
    log_returns = np.log(close / close.shift(1))
    var_95 = log_returns.rolling(window).quantile(0.95)
    var_05 = log_returns.rolling(window).quantile(0.05)
    return (var_95 + var_05) / (var_95 - var_05).replace(0, np.nan)


# --- Premium regime ---

def premium_vol_ratio(premium: pd.Series, 
                       fast_window: int, slow_window: int) -> pd.Series:
    """Premium volatility regime: fast/slow ratio"""
    vol_fast = rolling_std(premium, fast_window)
    vol_slow = rolling_std(premium, slow_window)
    return vol_fast / vol_slow.replace(0, np.nan)


# --- Spot regime ---

def spot_dominance(spot_volume: pd.Series, perp_volume: pd.Series) -> pd.Series:
    """Spot as fraction of total volume"""
    total = spot_volume + perp_volume
    return spot_volume / total.replace(0, np.nan)


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
                                vwap_window: int, ema_span: int) -> pd.Series:
    """d_t * v_t: distinguishes moving toward vs away from mean"""
    # Displacement from VWAP
    vwap = rolling_vwap(close, volume, vwap_window)
    d = (close - vwap) / vwap
    
    # Speed from EMA distance
    ema_val = ema(np.log(close), ema_span)
    v = np.log(close) - ema_val
    
    return d * v


def range_chop_interaction(high: pd.Series, low: pd.Series, open_: pd.Series,
                            close: pd.Series, window: int) -> pd.Series:
    """PV * (1 - Eff): high vol + low efficiency = chop"""
    pv = parkinson_volatility(high, low, window)
    eff = efficiency_avg(open_, high, low, close, window)
    return pv * (1 - eff)


def range_stretch_interaction(high: pd.Series, low: pd.Series, close: pd.Series,
                               volume: pd.Series, vwap_window: int, 
                               vol_window: int) -> pd.Series:
    """PV * |d|: volatility scaled by displacement"""
    pv = parkinson_volatility(high, low, vol_window)
    vwap = rolling_vwap(close, volume, vwap_window)
    d = np.abs((close - vwap) / vwap)
    return pv * d


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


def three_way_divergence(spot_volume: pd.Series, oi: pd.Series, premium: pd.Series,
                          window: int) -> pd.Series:
    """Divergence: spot_z - oi_z - premium_z"""
    spot_z = zscore(spot_volume, window)
    oi_z = zscore(oi, window)
    prem_z = zscore(premium, window)
    return spot_z - oi_z - prem_z


# --- Spot-Price interactions ---

def spot_vol_price_corr(spot_volume: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Spot volume-price correlation"""
    spot_roc = pct_change(spot_volume)
    price_roc = pct_change(close)
    return rolling_corr(spot_roc, price_roc, window)


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


def spot_vs_oi_price_corr(spot_volume: pd.Series, oi: pd.Series, 
                           close: pd.Series, window: int) -> pd.Series:
    """Difference: spot-price corr minus OI-price corr"""
    price_roc = pct_change(close)
    spot_roc = pct_change(spot_volume)
    oi_roc = pct_change(oi)
    
    spot_corr = rolling_corr(spot_roc, price_roc, window)
    oi_corr = rolling_corr(oi_roc, price_roc, window)
    return spot_corr - oi_corr


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


def trade_size_premium_divergence(quote_volume: pd.Series, num_trades: pd.Series,
                                   premium: pd.Series, window: int) -> pd.Series:
    """Trade size vs premium divergence"""
    avg_size = quote_volume / num_trades.replace(0, np.nan)
    size_z = zscore(avg_size, window)
    prem_z = zscore(premium, window)
    return size_z - prem_z
