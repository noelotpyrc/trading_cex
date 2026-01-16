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
    rolling_highest, rolling_lowest, rolling_median,
    parkinson_volatility, historical_volatility, ewma_volatility, true_range,
    rsi, adx, zscore,
    rolling_vwap, time_features, var_cvar, linear_regression_slope,
    kalman_filter_slope
)


# =============================================================================
# STATE VARIABLES (Price OHLC only)
# =============================================================================

# Displacement (distance from equilibrium)
def price_vwap_distance_zscore(open_: pd.Series, high: pd.Series, low: pd.Series,
                                close: pd.Series, volume: pd.Series, 
                                vwap_window: int, zscore_window: int) -> pd.Series:
    """Displacement: price distance from VWAP, z-scored"""
    vwap = rolling_vwap(open_, high, low, close, volume, vwap_window)
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
    """
    Fraction of non-zero returns in the window that have the same sign as the current.
    
    Zero returns are completely ignored (become NaN).
    
    NOTE: Not used in feature generation due to high NaN rate when close is unchanged.
    """
    r = np.log(close).diff()
    s = np.sign(r).replace(0, np.nan)  # zeros → NaN, ignored

    def same_sign_frac(x: pd.Series) -> float:
        x_valid = x.dropna()
        if len(x_valid) < 2:
            return np.nan
        cur = x_valid.iloc[-1]
        return (x_valid.iloc[:-1] == cur).mean()

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

def taker_imb_cvd_slope(taker_buy: pd.Series, spot_volume: pd.Series, window: int) -> pd.Series:
    """Slope of cumulative taker imbalance (directional pressure)"""
    imbalance = taker_buy - (spot_volume - taker_buy)  # = 2*taker_buy - spot_vol
    cvd = imbalance.cumsum()
    return (cvd - cvd.shift(window)) / window  # fast endpoint slope
# try window as 24, 168

def taker_imb_zscore(taker_buy: pd.Series, spot_volume: pd.Series, 
                     window: int) -> pd.Series:
    """Z-score of taker imbalance (aggressive flow)"""
    imbalance = taker_buy - (spot_volume - taker_buy)  # = 2*taker_buy - spot_vol
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

def pullback_slope_ema(close: pd.Series, ema_span: int, window: int) -> pd.Series:
    """Pullback slope using EMA as mean proxy."""
    y = np.log(close)
    m = ema(y, ema_span)
    d = y - m
    r = y.diff(1)

    x = (-d.shift(1))  # predictor
    cov = x.rolling(window).cov(r)
    var = x.rolling(window).var()

    return cov / var.where(var > 0)
# try ema span as 24, 168 and 720, and window as 48 and 168


def pullback_slope_vwap(open_: pd.Series, high: pd.Series, low: pd.Series,
                         close: pd.Series, volume: pd.Series, 
                         vwap_window: int, window: int) -> pd.Series:
    """Pullback slope using VWAP as mean proxy."""
    y = np.log(close)
    vwap = rolling_vwap(open_, high, low, close, volume, vwap_window)
    m = np.log(vwap)
    d = y - m
    r = y.diff(1)

    x = (-d.shift(1))  # predictor
    cov = x.rolling(window).cov(r)
    var = x.rolling(window).var()

    return cov / var.where(var > 0)
# try vwap window as 24, 168 and 720, and window as 48 and 168


def mean_cross_rate_ema(close: pd.Series, ema_span: int, window: int) -> pd.Series:
    """Mean crossing rate using EMA as mean proxy."""
    y = np.log(close)
    m = ema(y, ema_span)
    d = y - m

    s = np.sign(d).replace(0, np.nan).ffill()  # avoid 0-sign jitter
    cross = (s != s.shift(1)).astype(float)

    # exact normalization by (window - 1)
    return cross.rolling(window).sum() / (window - 1)
# try ema span as 24, 168 and 720, and window as 48 and 168


def mean_cross_rate_vwap(open_: pd.Series, high: pd.Series, low: pd.Series,
                          close: pd.Series, volume: pd.Series,
                          vwap_window: int, window: int) -> pd.Series:
    """Mean crossing rate using VWAP as mean proxy."""
    y = np.log(close)
    vwap = rolling_vwap(open_, high, low, close, volume, vwap_window)
    m = np.log(vwap)
    d = y - m

    s = np.sign(d).replace(0, np.nan).ffill()
    cross = (s != s.shift(1)).astype(float)

    return cross.rolling(window).sum() / (window - 1)
# try vwap window as 24, 168 and 720, and window as 48 and 168


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

def taker_imb_price_corr(taker_buy: pd.Series, spot_volume: pd.Series, 
                         close: pd.Series, window: int) -> pd.Series:
    """Taker imbalance-price correlation (exhaustion when weak)"""
    imbalance = taker_buy - (spot_volume - taker_buy)  # = 2*taker_buy - spot_vol
    imb_z = zscore(imbalance, window)
    price_roc = pct_change(close)
    return rolling_corr(imb_z, price_roc, window)
# try window as 168

def avg_trade_size_price_corr(quote_volume: pd.Series, num_trades: pd.Series,
                               close: pd.Series, window: int) -> pd.Series:
    """Average trade size-price correlation"""
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
# try fast window as 24, and slow window as 168

# =============================================================================
# INTERACTIONS
# =============================================================================

# --- Core physics interactions ---

def displacement_speed_product(open_: pd.Series, high: pd.Series, low: pd.Series,
                               close: pd.Series, volume: pd.Series,
                               vwap_window: int, speed_window: int) -> pd.Series:
    """x_t * speed_t: x_t is log(C/VWAP), speed is k-step log return."""
    vwap = rolling_vwap(open_, high, low, close, volume, vwap_window)

    x = np.log(close) - np.log(vwap)                 # displacement (log space)
    speed = np.log(close) - np.log(close.shift(speed_window))  # k-step speed

    return x * speed
# try vwap window as 168 and 720, and speed window as 24 and 48


def range_chop_interaction(high, low, open_, close, window: int) -> pd.Series:
    pv = parkinson_volatility(high, low, window)
    eff = efficiency_avg(open_, high, low, close, window).clip(0, 1)
    return pv * (1.0 - eff)
# try window as 24 and 168

def range_stretch_interaction(open_: pd.Series, high: pd.Series, low: pd.Series,
                              close: pd.Series, volume: pd.Series,
                              vwap_window: int, vol_window: int) -> pd.Series:
    """PV * |log(C/VWAP)|: big range and big stretch from equilibrium."""
    pv = parkinson_volatility(high, low, vol_window)
    vwap = rolling_vwap(open_, high, low, close, volume, vwap_window)

    stretch = (np.log(close) - np.log(vwap)).abs()
    return pv * stretch
# try vwap window as 168 and 720, and vol window as 24 and 168


# --- Utility ---

def scaled_acceleration(series: pd.Series, span: int) -> pd.Series:
    """Scaled acceleration: accel / volatility"""
    roc = pct_change(series)
    accel = roc.diff()
    volatility = np.sign(roc) * roc.diff()
    
    accel_ema = ema(accel, span)
    vol_ema = ema(volatility, span)
    
    return accel_ema / vol_ema.replace(0, np.nan)
# try span as 24 and 168

# --- OI-Price interactions ---

def oi_price_ratio_spread(oi: pd.Series, close: pd.Series, 
                           fast_span: int, slow_span: int) -> pd.Series:
    """Difference between OI and price EMA ratios"""
    oi_ratio = ema(oi, fast_span) / ema(oi, slow_span)
    price_ratio = ema(close, fast_span) / ema(close, slow_span)
    return oi_ratio - price_ratio
# try fast span as 24, and slow span as 168

# --- Spot-Price interactions ---

def spot_vol_price_corr(spot_volume: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Spot volume-price correlation"""
    spot_roc = pct_change(spot_volume)
    price_roc = pct_change(close)
    return rolling_corr(spot_roc, price_roc, window)
# try window as 168

def oi_vol_price_corr(oi: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """OI-price correlation"""
    oi_roc = pct_change(oi)
    price_roc = pct_change(close)
    return rolling_corr(oi_roc, price_roc, window)
# try window as 168

def spot_dom_price_corr(spot_volume: pd.Series, perp_volume: pd.Series,
                         close: pd.Series, window: int) -> pd.Series:
    """Spot dominance-price correlation"""
    spot_dom = spot_dominance(spot_volume, perp_volume)
    spot_dom_roc = pct_change(spot_dom)
    price_roc = pct_change(close)
    return rolling_corr(spot_dom_roc, price_roc, window)
# try window as 24 and 168

def spot_dom_oi_corr(spot_volume: pd.Series, perp_volume: pd.Series,
                      oi: pd.Series, window: int) -> pd.Series:
    """Spot dominance-OI correlation"""
    spot_dom = spot_dominance(spot_volume, perp_volume)
    spot_dom_roc = pct_change(spot_dom)
    oi_roc = pct_change(oi)
    return rolling_corr(spot_dom_roc, oi_roc, window)
# try window as 24 and 168

# --- Trade interactions ---

def trade_count_oi_corr(num_trades: pd.Series, oi: pd.Series, window: int) -> pd.Series:
    """Trade count-OI correlation"""
    tc_roc = pct_change(num_trades)
    oi_roc = pct_change(oi)
    return rolling_corr(tc_roc, oi_roc, window)
# try window as 168

def trade_count_spot_dom_corr(num_trades: pd.Series, spot_volume: pd.Series,
                               perp_volume: pd.Series, window: int) -> pd.Series:
    """Trade count-spot dominance correlation"""
    tc_roc = pct_change(num_trades)
    spot_dom = spot_dominance(spot_volume, perp_volume)
    spot_dom_roc = pct_change(spot_dom)
    return rolling_corr(tc_roc, spot_dom_roc, window)
# try window as 168

def amihud(close: pd.Series, volume: pd.Series, eps: float = 1e-12) -> pd.Series:
    close = close.astype(float)
    volume = volume.astype(float)
    r = np.log(close).diff().abs()
    dollar_vol = (close * volume).replace(0, np.nan)
    return r / (dollar_vol + eps)


def normalize_ratio(
    x: pd.Series,
    window: int,
    method: str = "median_ratio",
    log_transform: bool = True,
    eps: float = 1e-12,
    min_periods: int = None
) -> pd.Series:
    """
    Generic normalization helper.

    Methods:
      - "median_ratio": x / rolling_median(x)          (ratio in raw space)
      - "mean_ratio":   x / rolling_mean(x)            (ratio in raw space)
      - "demean":       log(x) - rolling_mean(log(x))  (additive, log-space if log_transform)
      - "zscore":       (log(x)-mu)/sd                 (additive, log-space if log_transform)

    If log_transform=True:
      - for ratio methods: return log(x / base)
      - for demean/zscore: operate on log(x)
    
    min_periods defaults to 60% of window (100 for 168-window).
    """
    if min_periods is None:
        min_periods = int(window * 0.6)
    
    if method in ("median_ratio", "mean_ratio"):
        if method == "median_ratio":
            base = x.rolling(window, min_periods=min_periods).median()
        else:
            base = x.rolling(window, min_periods=min_periods).mean()

        rel = x / (base + eps)
        return np.log(rel + eps) if log_transform else rel

    # additive normalizations
    z = np.log(x + eps) if log_transform else x

    if method == "demean":
        mu = z.rolling(window, min_periods=min_periods).mean()
        return z - mu

    if method == "zscore":
        mu = z.rolling(window, min_periods=min_periods).mean()
        sd = z.rolling(window, min_periods=min_periods).std()
        return (z - mu) / (sd + eps)

    raise ValueError(f"Unknown method: {method}")


def relative_amihud(
    close: pd.Series,
    volume: pd.Series,
    window: int,
    method: str = "median_ratio",
    log_transform: bool = True,
    eps: float = 1e-12
) -> pd.Series:
    a = amihud(close, volume, eps=eps)
    return normalize_ratio(a, window, method=method, log_transform=log_transform, eps=eps)

# try window as 168


def oi_volume_efficiency(oi: pd.Series, volume: pd.Series, 
                          ratio_window: int, norm_window: int,
                          method: str = "median_ratio", 
                          log_transform: bool = True) -> pd.Series:
    """
    OI change efficiency: |N-bar OI change| / (N-bar volume sum), normalized.
    
    Uses absolute OI change (like Amihud uses |return|).
    """
    oi_change = oi.diff(ratio_window).abs()
    vol_sum = volume.rolling(ratio_window).sum()
    ratio = oi_change / vol_sum.replace(0, np.nan)
    return normalize_ratio(ratio, norm_window, method=method, log_transform=log_transform)
# try ratio_window as 24, norm_window as 168


def oi_volume_efficiency_signed(oi: pd.Series, volume: pd.Series,
                                 ratio_window: int, norm_window: int,
                                 method: str = "median_ratio",
                                 log_transform: bool = True) -> tuple[pd.Series, pd.Series]:
    """
    Separate OI change efficiency for positive and negative OI changes.
    
    Returns:
        (oi_vol_eff_positive, oi_vol_eff_negative)
        
    Positive: OI building (new positions opening)
    Negative: OI shrinking (positions closing/liquidating)
    
    Uses clip pattern (zeros where sign doesn't match).
    """
    oi_change = oi.diff(ratio_window)
    vol_sum = volume.rolling(ratio_window).sum().replace(0, np.nan)
    
    ratio_pos = oi_change.clip(lower=0.0) / vol_sum
    ratio_neg = (-oi_change).clip(lower=0.0) / vol_sum
    
    eff_pos = normalize_ratio(ratio_pos, norm_window, method=method, log_transform=log_transform)
    eff_neg = normalize_ratio(ratio_neg, norm_window, method=method, log_transform=log_transform)
    
    return eff_pos, eff_neg
# try ratio_window as 48 and 168, norm_window as 168


# =============================================================================
# LONG-TERM TREND FLAGS
# =============================================================================

def is_above_daily_ema(
    timestamp: pd.Series,
    close: pd.Series,
    span_days: int,
) -> pd.Series:
    """
    Binary flag: 1 if current close is above the lagged daily EMA, 0 otherwise.
    
    Uses daily_ema_lagged() which shifts by 1 day to prevent look-ahead bias.
    At any hour of Day T, the EMA is computed from daily closes up to Day T-1.
    
    Args:
        timestamp: Timestamp series (1H frequency)
        close: Close price series (1H frequency)
        span_days: EMA span in days (e.g., 30, 180, 365)
    
    Returns:
        pd.Series: Binary flag (1 = above EMA, 0 = below or equal)
    """
    from feature_engineering.primitives import daily_ema_lagged
    
    ema_values = daily_ema_lagged(timestamp, close, span_days)
    return (close > ema_values).astype(int)
# use span_days 30, 180, 365


# =============================================================================
# LOOKBACK R-MULTIPLE FEATURES
# =============================================================================

def lookback_avg_r_multiple(
    r_multiple: pd.Series,
    window: int,
    horizon_lag: int,
    min_periods: int = None,
) -> pd.Series:
    """
    Rolling mean of historical R-multiples.
    
    This is a feature that measures "recent regime performance" based on
    what the average R-multiple outcome has been over the lookback window.
    
    The function internally shifts the R-multiple series by horizon_lag bars
    to prevent look-ahead bias. At time t, we only see R-multiples from
    trades that entered at t-horizon_lag or earlier and have concluded.
    
    Args:
        r_multiple: R-multiple target series (unshifted)
        window: Lookback window in bars
        horizon_lag: Number of bars to shift (typically 24 for 24h horizon)
        min_periods: Minimum periods for rolling calculation
    
    Returns:
        pd.Series: Rolling mean of lagged R-multiples
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    # Shift by horizon_lag to only see concluded trades
    lagged = r_multiple.shift(horizon_lag)
    return lagged.rolling(window, min_periods=min_periods).mean()
# use window 168 (1 week) or 720 (1 month), horizon_lag 24 for 24h targets


def lookback_r_multiple_std(
    r_multiple: pd.Series,
    window: int,
    horizon_lag: int,
    min_periods: int = None,
) -> pd.Series:
    """
    Rolling std of historical R-multiples.
    
    Measures volatility/consistency of recent R-multiple outcomes.
    Shifts by horizon_lag to prevent look-ahead bias.
    
    Args:
        r_multiple: R-multiple target series (unshifted)
        window: Lookback window in bars
        horizon_lag: Number of bars to shift (typically 24 for 24h horizon)
        min_periods: Minimum periods for rolling calculation
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    lagged = r_multiple.shift(horizon_lag)
    return lagged.rolling(window, min_periods=min_periods).std()
# use window 168 (1 week) or 720 (1 month), horizon_lag 24


def lookback_r_multiple_hit_rate(
    r_multiple: pd.Series,
    window: int,
    horizon_lag: int,
    threshold: float = 0.0,
    min_periods: int = None,
) -> pd.Series:
    """
    Rolling fraction of R-multiples above a threshold.
    
    Default threshold=0 gives "win rate" (positive vs negative outcomes).
    Use threshold=1.99 for "TP hit rate" (for 2R reward setups).
    Shifts by horizon_lag to prevent look-ahead bias.
    
    Args:
        r_multiple: R-multiple target series (unshifted)
        window: Lookback window in bars
        horizon_lag: Number of bars to shift (typically 24 for 24h horizon)
        threshold: R-multiple threshold (default 0 for win rate)
        min_periods: Minimum periods for rolling calculation
    
    Returns:
        pd.Series: Rolling fraction of lagged R-multiples > threshold
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    lagged = r_multiple.shift(horizon_lag)
    wins = (lagged > threshold).astype(float)
    return wins.rolling(window, min_periods=min_periods).mean()
# use window 168 or 720, horizon_lag 24, threshold 0 for win rate or 1.99 for TP rate


# =============================================================================
# VOL-NORMALIZED MOMENTUM
# =============================================================================

def vol_normalized_momentum(close: pd.Series, ret_window: int, vol_span: int,
                             scale_by_sqrt_k: bool = True,
                             eps: float = 1e-9) -> pd.Series:
    """
    Momentum in volatility units: r_t(k) / (σ_t × √k).
    
    This is a "t-stat-ish" scaling that normalizes returns by volatility
    and time horizon, making momentum comparable across different windows.
    
    Args:
        close: Close price series
        ret_window: Return lookback (k) in bars (e.g., 6, 12, 24, 48)
        vol_span: EWMA volatility span in bars (e.g., 48, 168)
        scale_by_sqrt_k: If True, divide by √k for time-scaling
        eps: Small epsilon for numerical stability
    
    Returns:
        Vol-normalized momentum signal
    """
    r_k = np.log(close / close.shift(ret_window))
    sigma = ewma_volatility(close, vol_span)
    denom = sigma * np.sqrt(ret_window) if scale_by_sqrt_k else sigma
    return r_k / (denom + eps)
# Use: ret_window ∈ {4, 12, 48}, vol_span ∈ {48, 168}


# =============================================================================
# BREAKOUT STRENGTH + CLOSE LOCATION
# =============================================================================

def breakout_distance(high: pd.Series, low: pd.Series, close: pd.Series,
                      window: int, vol_span: int, eps: float = 1e-9) -> pd.Series:
    """
    Breakout distance: (C - mid) / σ, where mid = (HH + LL) / 2.
    
    Measures how far price is from the midpoint of the N-bar range,
    normalized by volatility. Positive = near highs, negative = near lows.
    
    Args:
        high, low, close: OHLC data
        window: Lookback for HH/LL (e.g., 24, 48, 72)
        vol_span: EWMA volatility span (e.g., 168)
    """
    hh = rolling_highest(high, window)
    ll = rolling_lowest(low, window)
    mid = (hh + ll) / 2
    sigma = ewma_volatility(close, vol_span)
    dist = np.log(close / mid)
    return dist / (sigma + eps)
# Use: window ∈ {48, 2160, 4320}, vol_span = 168


def close_location_value(high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int, eps: float = 1e-9) -> pd.Series:
    """
    Close location within N-bar range: (C - LL) / (HH - LL) ∈ [0, 1].
    
    0 = close at recent lows, 1 = close at recent highs.
    
    Args:
        high, low, close: OHLC data
        window: Lookback for HH/LL (e.g., 24, 48, 72)
    """
    hh = rolling_highest(high, window)
    ll = rolling_lowest(low, window)
    clv = (close - ll) / (hh - ll + eps)
    return clv.clip(0.0, 1.0)
# Use: window ∈ {48, 2160, 4320}


def close_location_value_signed(high: pd.Series, low: pd.Series, close: pd.Series,
                                 window: int, eps: float = 1e-9) -> pd.Series:
    """
    Close location signed: 2 × CLV - 1 ∈ [-1, +1].
    
    -1 = close at recent lows, +1 = close at recent highs.
    """
    return 2 * close_location_value(high, low, close, window, eps) - 1
# Use: window ∈ {48, 720, 2160}


# =============================================================================
# RANGE EXPANSION + CLOSE LOCATION
# =============================================================================

def range_expansion(high: pd.Series, low: pd.Series, close: pd.Series,
                    window: int, eps: float = 1e-9) -> pd.Series:
    """
    Range expansion: TR / median(TR, W).
    
    Values > 1 indicate larger-than-typical range (expansion),
    values < 1 indicate smaller-than-typical range (contraction).
    
    Args:
        high, low, close: OHLC data
        window: Lookback for median TR (e.g., 48, 168)
    """
    tr = true_range(high, low, close)
    tr_med = rolling_median(tr, window).shift(1)
    return tr / (tr_med + eps)
# Use: window ∈ {6, 12, 24}


def close_in_range(high: pd.Series, low: pd.Series, close: pd.Series,
                   eps: float = 1e-9) -> pd.Series:
    """
    Within-bar close location: (C - L) / (H - L) ∈ [0, 1].
    
    0 = closed at bar low, 1 = closed at bar high.
    Single-bar feature (no window).
    """
    cir = (close - low) / (high - low + eps)
    return cir.clip(0.0, 1.0)


def close_in_range_signed(high: pd.Series, low: pd.Series, close: pd.Series,
                          eps: float = 1e-9) -> pd.Series:
    """
    Within-bar close location signed: 2 × CIR - 1 ∈ [-1, +1].
    
    -1 = closed at bar low, +1 = closed at bar high.
    """
    return 2 * close_in_range(high, low, close, eps) - 1


def impulse_signal(high: pd.Series, low: pd.Series, close: pd.Series,
                   re_window: int, eps: float = 1e-9) -> pd.Series:
    """
    Impulse signal: log(range_expansion) × close_in_range_signed.
    
    Combines "big range" with "strong close" into a single signal:
    - Big range + close near high → positive (bullish impulse)
    - Big range + close near low → negative (bearish impulse)
    - Big range + close in middle → near 0 (indecision/whipsaw)
    
    Args:
        high, low, close: OHLC data
        re_window: Lookback for range expansion median (e.g., 48, 168)
    """
    re = range_expansion(high, low, close, re_window, eps)
    cir2 = close_in_range_signed(high, low, close, eps)
    return np.log1p(re - 1.0) * cir2
# Use: re_window ∈ {6, 12, 24}


# =============================================================================
# ORDER FLOW PROXIES (OHLCV ONLY)
# =============================================================================

def close_location_value_bar(high: pd.Series, low: pd.Series, close: pd.Series,
                              eps: float = 1e-9) -> pd.Series:
    """
    CLV within bar: [(C-L) - (H-C)] / (H-L) ∈ [-1, +1].
    
    This is the standard CLV formula used in Accumulation/Distribution.
    Equivalent to 2×CIR - 1, but computed directly.
    """
    clv = ((close - low) - (high - close)) / (high - low + eps)
    return clv.clip(-1.0, 1.0)

def money_flow_volume(high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series) -> pd.Series:
    """
    Money Flow Volume: CLV × Volume.
    
    Positive when close is near high (accumulation),
    negative when close is near low (distribution).
    """
    clv = close_location_value_bar(high, low, close)
    return clv * volume


def accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series,
                              volume: pd.Series, window: int) -> pd.Series:
    """
    Accumulation/Distribution ratio: Σ(MFV, K) / Σ(V, K).
    
    Normalized flow indicator. Positive = net accumulation,
    negative = net distribution over the window.
    
    Args:
        high, low, close, volume: OHLCV data
        window: Lookback window (e.g., 6, 12, 24)
    """
    mfv = money_flow_volume(high, low, close, volume)
    mfv_sum = mfv.rolling(window, min_periods=window).sum()
    vol_sum = volume.rolling(window, min_periods=window).sum()
    ad = mfv_sum / (vol_sum + 1e-9)
    return ad.clip(-1.0, 1.0)
# Use: window ∈ {24, 48, 168}


def signed_volume_proxy(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Signed volume: sign(ΔC) × V.
    
    Simple proxy for directional volume when tick data unavailable.
    """
    return np.sign(np.log(close).diff()) * volume


def cumulative_signed_volume(close: pd.Series, volume: pd.Series,
                             window: int) -> pd.Series:
    """
    Cumulative signed volume ratio: Σ(signed_vol) / Σ(V).
    
    Net buy/sell pressure over the window, normalized to [-1, +1].
    
    Args:
        close, volume: Price and volume series
        window: Lookback window (e.g., 6, 12, 24)
    """
    signed_vol = signed_volume_proxy(close, volume)
    sv_sum = signed_vol.rolling(window, min_periods=window).sum()
    vol_sum = volume.rolling(window, min_periods=window).sum()
    csv = sv_sum / (vol_sum + 1e-9)
    return csv.clip(-1.0, 1.0)
# Use: window ∈ {48, 168, 720}


def volume_shock(volume: pd.Series, window: int, eps: float = 1e-9) -> pd.Series:
    """
    Volume shock: V / median(V, W).
    
    Values > 1 indicate higher-than-typical volume.
    Useful for gating signals (e.g., only trade on high-volume bars).
    
    Args:
        volume: Volume series
        window: Lookback for median (e.g., 48, 168)
    """
    med = rolling_median(volume, window).shift(1)
    vs = volume / (med + eps)
    return vs
# Use: window ∈ {168, 720}


# =============================================================================
# PERP ORDER FLOW FEATURES (using num_trades and taker_buy_volume)
# =============================================================================

# --- Section 1: Taker Imbalance (Core Direction Signal) ---

def taker_buy_ratio(taker_buy: pd.Series, volume: pd.Series,
                    eps: float = 1e-9) -> pd.Series:
    """
    Raw taker buy ratio: TB / V ∈ [0, 1].
    
    0.5 = balanced, >0.5 = more aggressive buying, <0.5 = more aggressive selling.
    """
    r = taker_buy / (volume + eps)
    return r.clip(0.0, 1.0)

def taker_imbalance(taker_buy: pd.Series, volume: pd.Series,
                    eps: float = 1e-9) -> pd.Series:
    """
    Signed taker imbalance: (2*TB - V) / V ∈ [-1, +1].
    
    Equivalent to (TB - TS) / V where TS = V - TB (taker sell).
    +1 = all taker buys, -1 = all taker sells.
    """
    den = volume.clip(lower=eps)
    imb = (2 * taker_buy - volume) / den
    return imb.clip(-1.0, 1.0)


def taker_imbalance_ema(taker_buy: pd.Series, volume: pd.Series,
                         span: int, eps: float = 1e-9) -> pd.Series:
    """
    Smoothed taker imbalance: EWMA of imbalance.
    
    Reduces bar-to-bar noise while preserving directional signal.
    
    Args:
        taker_buy, volume: Per-bar taker buy and total volume
        span: EWMA span (e.g., 24, 48, 168)
    """
    imb = taker_imbalance(taker_buy, volume, eps)
    return ema(imb, span)


def taker_imbalance_ema_alt(taker_buy: pd.Series, volume: pd.Series,
                         span: int, eps: float = 1e-9) -> pd.Series:
    net = (2*taker_buy - volume)
    num = net.ewm(span=span, adjust=False, min_periods=max(2, span//5)).mean()
    den = volume.ewm(span=span, adjust=False, min_periods=max(2, span//5)).mean().clip(lower=eps)
    return (num / den).clip(-1.0, 1.0)
# Use: span ∈ {24, 48, 168}

def cumulative_taker_imbalance(taker_buy: pd.Series, volume: pd.Series,
                                window: int, eps: float = 1e-9) -> pd.Series:
    """
    Cumulative taker imbalance: Σ(TB - TS) / Σ(V) over window.
    
    Net directional pressure over the lookback period.
    
    Args:
        taker_buy, volume: Per-bar taker buy and total volume
        window: Lookback window (e.g., 6, 12, 24)
    """
    taker_sell = volume - taker_buy
    net_imb = (taker_buy - taker_sell).rolling(window).sum()
    vol_sum = volume.rolling(window).sum()
    return net_imb / (vol_sum + eps)
# Use: window ∈ {24, 48, 168}


# --- Section 2: Intensity / Participation ---

def trades_per_volume(num_trades: pd.Series, volume: pd.Series,
                      eps: float = 1e-9) -> pd.Series:
    """
    Trades per volume: N / V.
    
    High = many small trades (retail), low = few large trades (institutional).
    """
    den = volume.replace(0, np.nan)
    tpv = num_trades / den
    return np.log1p(tpv)


def avg_trade_size(volume: pd.Series, num_trades: pd.Series,
                   eps: float = 1e-9) -> pd.Series:
    """
    Average trade size: V / N.
    
    Inverse of trades_per_volume. Large = whale activity.
    """
    den = num_trades.replace(0, np.nan)
    ats = volume / den
    return np.log1p(ats)


def taker_buy_ratio_zscore(taker_buy: pd.Series, volume: pd.Series,
                            window: int, eps: float = 1e-9) -> pd.Series:
    """
    Z-score of taker buy ratio: (ratio - rolling_mean) / rolling_std.
    
    Measures how unusual current taker dominance is vs recent history.
    """
    ratio = taker_buy_ratio(taker_buy, volume, eps)
    roll_mean = ratio.rolling(window, min_periods=window).mean()
    roll_std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - roll_mean) / (roll_std + eps)
    return z.clip(-5.0, 5.0)
# Use: window = 168


def trades_per_vol_zscore(num_trades: pd.Series, volume: pd.Series,
                           window: int, eps: float = 1e-9) -> pd.Series:
    """
    Z-score of trades per volume: (tpv - rolling_mean) / rolling_std.
    
    Measures unusual fragmentation (many small trades vs few large).
    """
    tpv = trades_per_volume(num_trades, volume, eps)
    roll_mean = tpv.rolling(window, min_periods=window).mean()
    roll_std = tpv.rolling(window, min_periods=window).std()
    z = (tpv - roll_mean) / (roll_std + eps)
    return z.clip(-5.0, 5.0)
# Use: window = 168


def avg_trade_size_zscore(volume: pd.Series, num_trades: pd.Series,
                           window: int, eps: float = 1e-9) -> pd.Series:
    """
    Z-score of avg trade size: (ats - rolling_mean) / rolling_std.
    
    Measures unusual trade size activity (whale vs retail).
    """
    ats = avg_trade_size(volume, num_trades, eps)
    roll_mean = ats.rolling(window, min_periods=window).mean()
    roll_std = ats.rolling(window, min_periods=window).std()
    z = (ats - roll_mean) / (roll_std + eps)
    return z.clip(-5.0, 5.0)
# Use: window = 168


def trade_count_shock(num_trades: pd.Series, window: int,
                      eps: float = 1e-9) -> pd.Series:
    """
    Trade count shock: N / median(N, W).
    
    Activity spike indicator. Values > 1 indicate more trades than typical.
    
    Args:
        num_trades: Number of trades per bar
        window: Lookback for median (e.g., 48, 168)
    """
    med = rolling_median(num_trades, window).shift(1)
    shock = num_trades / (med + eps)
    return shock
# Use: window ∈ {48, 168}


def crowdedness(num_trades: pd.Series, volume: pd.Series,
                window: int, eps: float = 1e-9) -> pd.Series:
    """
    Crowdedness proxy: log(Nshock) - log(Vshock).
    
    Positive = many trades but little volume (churny, retail-dominated).
    Negative = few trades but large volume (institutional).
    
    Args:
        num_trades, volume: Per-bar values
        window: Lookback for shock calculation (e.g., 48, 168)
    """
    n_shock = trade_count_shock(num_trades, window, eps)
    v_shock = volume_shock(volume, window, eps)
    return np.log1p(n_shock - 1.0) - np.log1p(v_shock - 1.0)
# Use: window ∈ {48, 168}


# --- Section 3: Flow × Return Alignment ---

def flow_return_alignment(taker_buy: pd.Series, volume: pd.Series,
                          close: pd.Series, eps: float = 1e-9) -> pd.Series:
    """
    Flow-return alignment: imb × r (per-bar).
    
    Positive = flow aligned with price move (buys on up, sells on down).
    Negative = flow opposing price move.
    """
    imb = taker_imbalance(taker_buy, volume, eps)
    r = np.log(close / close.shift(1))
    return imb * r


def rolling_flow_alignment(taker_buy: pd.Series, volume: pd.Series,
                           close: pd.Series, window: int,
                           eps: float = 1e-9) -> pd.Series:
    """
    Rolling flow alignment: Σ(imb × r) over window.
    
    Cumulative measure of how well flow predicts price direction.
    
    Args:
        taker_buy, volume, close: Per-bar values
        window: Lookback window (e.g., 12, 24, 48)
    """
    align = flow_return_alignment(taker_buy, volume, close, eps)
    return align.rolling(window).sum()
# Use: window ∈ {12, 24, 168}


def flow_efficiency(taker_buy: pd.Series, volume: pd.Series,
                    close: pd.Series, window: int,
                    eps: float = 1e-9) -> pd.Series:
    """
    Flow efficiency: Σ(r) / Σ(imb) over window.
    
    Price move per unit of imbalance. High = efficient flow (moves price).
    Clip denominator to avoid explosions when net imbalance is near zero.
    
    Args:
        taker_buy, volume, close: Per-bar values
        window: Lookback window (e.g., 6, 12, 24)
    """
    imb = taker_imbalance(taker_buy, volume, eps)
    r = np.log(close / close.shift(1))

    sum_r = r.rolling(window, min_periods=window).sum()
    sum_imb = imb.rolling(window, min_periods=window).sum()

    safe_den = np.sign(sum_imb) * np.maximum(sum_imb.abs(), eps)
    return sum_r / safe_den
# Use: window ∈ {24, 48, 168}


# --- Section 4: Absorption / Price Impact ---

def price_impact(taker_buy: pd.Series, volume: pd.Series,
                 close: pd.Series, eps: float = 1e-9) -> pd.Series:
    """
    Price impact proxy: |r| / |imb|.
    
    High = small imbalance moves price a lot (thin liquidity).
    Low = large imbalance moves price little (absorption/thick book).
    """
    imb = taker_imbalance(taker_buy, volume, eps)
    r = np.log(close / close.shift(1))
    impact = r.abs() / (imb.abs() + eps)
    return np.log1p(impact)

def rolling_price_impact_sum(taker_buy, volume, close, window, eps=1e-9):
    imb = taker_imbalance(taker_buy, volume, eps)
    r = np.log(close / close.shift(1))
    num = r.abs().rolling(window, min_periods=window).sum()
    den = imb.abs().rolling(window, min_periods=window).sum()
    pi = num / (den + eps)
    return np.log1p(pi)  # optional
# use window ∈ {24, 48, 168}

# --- Section 5: Churn / Reversal Risk ---

def imbalance_variance(taker_buy: pd.Series, volume: pd.Series,
                       window: int, eps: float = 1e-9) -> pd.Series:
    """
    Imbalance variance: Var(imb) over window.
    
    High variance = unstable/noisy flow, low variance = consistent direction.
    
    Args:
        taker_buy, volume: Per-bar values
        window: Lookback window (e.g., 12, 24)
    """
    imb = taker_imbalance(taker_buy, volume, eps)
    return imb.rolling(window).var()
# Use: window ∈ {48, 168, 720}


def imbalance_sign_flips(taker_buy: pd.Series, volume: pd.Series,
                         window: int, eps: float = 1e-9) -> pd.Series:
    """
    Imbalance sign flip count over window.
    
    High flips = choppy two-sided flow, low flips = sustained direction.
    Normalized by (window - 1) to get flip rate ∈ [0, 1].
    
    Args:
        taker_buy, volume: Per-bar values
        window: Lookback window (e.g., 12, 24)
    """
    imb = taker_imbalance(taker_buy, volume, eps)
    thr = 0.05  # or use a rolling percentile threshold
    s = np.where(imb > thr, 1, np.where(imb < -thr, -1, 0))
    s = pd.Series(s, index=imb.index)

    flips = ((s != s.shift(1)) & (s != 0) & (s.shift(1) != 0)).astype(float)
    return flips.rolling(window, min_periods=window).sum() / (window - 1)
# Use: window ∈ {24, 48, 168}


def churn_ratio(taker_buy: pd.Series, volume: pd.Series,
                window: int, eps: float = 1e-9) -> pd.Series:
    """
    Churn ratio: Σ|imb| / |Σ(imb)| over window.
    
    High churn (>>1) = lots of activity but nets out (two-sided chop).
    Low churn (~1) = consistent directional flow.
    
    Args:
        taker_buy, volume: Per-bar values
        window: Lookback window (e.g., 12, 24)
    """
    imb = taker_imbalance(taker_buy, volume, eps)
    sum_abs_imb = imb.abs().rolling(window).sum()
    abs_sum_imb = imb.rolling(window).sum().abs()
    churn = sum_abs_imb / (abs_sum_imb + eps)
    return np.log1p(churn - 1.0)  # equals log(churn) when churn>0, nicer near 1
# Use: window ∈ {24, 48, 168, 720}
