import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import timedelta

def blend_mar(mar_1, mar_3, mar_5, w1=0.2, w3=0.3, w5=0.5):
    mar_1 = pd.to_numeric(mar_1, errors="coerce")
    mar_3 = pd.to_numeric(mar_3, errors="coerce")
    mar_5 = pd.to_numeric(mar_5, errors="coerce")
    return (w1*mar_1 + w3*mar_3 + w5*mar_5)

def sigma_W_from_mar(mar_blend, W, k=np.sqrt(np.pi/2), safety=1.0):
    mar_blend = pd.to_numeric(mar_blend, errors="coerce").clip(lower=0)
    sigma_1m = k * mar_blend
    sigma_W = safety * sigma_1m * np.sqrt(W)
    return sigma_W

def prob_yes_normal(S_t, K, sigma_W, mu=0.0):
    """
    P(S_{t+W} > K) under lognormal with log-return ~ N(mu, sigma_W^2).
    mu=0 is a common short-horizon default. Supports scalar and vectorized inputs.
    """
    S_t = np.asarray(S_t, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma_W = np.asarray(sigma_W, dtype=float)
    x = (np.log(K / S_t) - mu) / sigma_W
    return 1.0 - norm.cdf(x)

def terminal_range_normal(S_t, sigma_W, coverage=0.98, mu=0.0):
    z = norm.ppf((1.0 + coverage) / 2.0)
    lower = S_t * np.exp(mu - z * sigma_W)
    upper = S_t * np.exp(mu + z * sigma_W)
    return lower, upper

def assign_group_strike(df, n_minutes, dt_col='datetime_utc',
                        close_col='close', open_col='open'):
    """
    Assign strike price K for close-based betting groups.

    K = close of the K-setter bar (the bar with ttl = n_minutes).
    The K-setter bar's close becomes K for the NEXT n_minutes bars.

    Exception: the very first group (before the first K-setter bar)
    uses open of the first bar in the dataset as K.

    Args:
        df: DataFrame with datetime, close, and open columns
        n_minutes: window size in minutes (e.g. 15)
        dt_col: name of the datetime column
        close_col: name of the close price column
        open_col: name of the open price column

    Returns:
        Series aligned to df index with the group strike price (K)
    """
    ttl = time_to_strike(df, n_minutes, dt_col)

    # K-setter bars have ttl == n_minutes. Their close becomes K
    # for the following bars until the next K-setter.
    k_series = pd.Series(np.nan, index=df.index)
    k_series.loc[ttl == n_minutes] = df.loc[ttl == n_minutes, close_col].values
    k_series = k_series.ffill()

    # Exception: first group (before any K-setter bar) uses open of first bar
    first_setter_idx = (ttl == n_minutes).idxmax() if (ttl == n_minutes).any() else len(df)
    k_series.loc[:first_setter_idx - 1] = df[open_col].iloc[0]

    return k_series

def time_to_strike(df, n_minutes, dt_col='datetime_utc'):
    """
    Close-based time-to-strike for N-minute betting groups.

    At each bar's close, ttl = number of close-to-close intervals
    remaining until the group resolves.

    Pattern for n_minutes=15:
      bar 00:00 → 14, 00:01 → 13, ..., 00:13 → 1,
      bar 00:14 → 15 (K-setter, starts new group)

    Args:
        df: DataFrame with datetime column
        n_minutes: window size in minutes
        dt_col: name of the datetime column

    Returns:
        Series (int) aligned to df index
    """
    group_start = df[dt_col].dt.floor(f'{n_minutes}min')
    elapsed = ((df[dt_col] - group_start).dt.total_seconds() / 60).astype(int)
    return ((n_minutes - 2 - elapsed) % n_minutes) + 1

def build_z_pool(df, score_date, group_minutes=15, lookback_days=7):
    """
    Build z-score pool for a given scoring day (walk-forward, no look-ahead).

    Uses pre-computed columns: ttl, sigma_W_ttl.
    Each bar's z = target_fwd_ret_{ttl} / sigma_W_ttl.

    Window: [score_date - lookback_days, score_date - group_minutes min)
    The cutoff ensures every bar's forward return is fully resolved.

    Args:
        df: DataFrame with datetime_utc, ttl, sigma_W_ttl, target_fwd_ret_{1..N}
        score_date: pd.Timestamp (midnight of the scoring day)
        group_minutes: group window size (max ttl)
        lookback_days: number of days of history to use

    Returns:
        1D numpy array of z-scores (NaN/inf removed)
    """
    train_start = score_date - timedelta(days=lookback_days)
    train_end = score_date - timedelta(minutes=group_minutes)

    mask = (df['datetime_utc'] >= train_start) & (df['datetime_utc'] < train_end)
    window = df.loc[mask]

    if len(window) == 0:
        return np.array([])

    # Pick the correct fwd_ret column for each bar's ttl
    # fwd_ret starts from close[t], consistent with prob_yes using close as S_t
    fwd_ret = pd.Series(np.nan, index=window.index)
    for t in range(1, group_minutes + 1):
        t_mask = window['ttl'] == t
        col = f'target_fwd_ret_{t}'
        if col in window.columns and t_mask.any():
            fwd_ret.loc[t_mask] = window.loc[t_mask, col]

    z = (fwd_ret / window['sigma_W_ttl']).replace([np.inf, -np.inf], np.nan).dropna().values
    return z

def ecdf_prob_greater(z_samples, threshold):
    """
    z_samples: 1D array of historical standardized returns Z (out-of-sample / training)
    threshold: scalar or array
    Returns P(Z > threshold) using empirical CDF.
    """
    z = np.asarray(z_samples)
    z = z[np.isfinite(z)]
    z_sorted = np.sort(z)

    thr = np.asarray(threshold)
    # P(Z <= thr) = searchsorted / n
    cdf = np.searchsorted(z_sorted, thr, side="right") / z_sorted.size
    return 1.0 - cdf

def prob_yes_empirical(S_t, K, sigma_W, z_samples, mu=0.0):
    """
    Uses empirical tail of standardized returns instead of Normal.
    """
    S_t = np.asarray(S_t, dtype=float)
    sigma_W = np.asarray(sigma_W, dtype=float)
    thr = (np.log(K / S_t) - mu) / sigma_W
    return ecdf_prob_greater(z_samples, thr)
