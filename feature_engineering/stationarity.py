"""
Stationarity Analysis Functions

This module provides functions to assess the stationarity of time series data,
useful for ranking features or price series by their mean-reverting properties.

Functions:
    - get_hurst_exponent: Estimates Hurst exponent using variance-time method
    - get_half_life: Estimates mean-reversion half-life using discrete AR(1)
    - stationarity_score: Combined metric for ranking (lower = more stationary)
    - rank_stationarity: Rank multiple series by stationarity
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional, Union


def get_hurst_exponent(
    x: Union[np.ndarray, pd.Series],
    max_lag: int = 20,
    min_lag: int = 2
) -> float:
    """
    Variance-time style estimate using Std(x[t+lag] - x[t]) ~ lag^H.

    Interpretation (on price levels):
        - H < 0.5: Mean-reverting / anti-persistent
        - H ≈ 0.5: Random walk (no memory)
        - H > 0.5: Trending / persistent

    Note: For white noise (returns), H ≈ 0. For a random walk (prices), H ≈ 0.5.

    Parameters
    ----------
    x : array-like
        Time series data (typically price levels, not returns)
    max_lag : int
        Maximum lag to consider (default: 20)
    min_lag : int
        Minimum lag to consider (default: 2)

    Returns
    -------
    float
        Hurst exponent, or np.nan if computation fails
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < (max_lag + 2):
        return np.nan

    lags = np.arange(min_lag, max_lag + 1)
    tau = np.empty(len(lags), dtype=float)

    for i, lag in enumerate(lags):
        diff = x[lag:] - x[:-lag]
        if len(diff) < 2:
            tau[i] = np.nan
            continue
        tau[i] = np.std(diff, ddof=1)

    # Avoid log(0) and invalid values
    mask = (tau > 0) & np.isfinite(tau)
    if mask.sum() < 2:
        return np.nan

    slope, _ = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)
    return float(slope)


def get_half_life(x: Union[np.ndarray, pd.Series]) -> float:
    """
    Estimates mean-reversion half-life using discrete-time AR(1) model.

    Model:
        Δx_t = α + β·x_{t-1} + ε_t

    The discrete-time autoregressive coefficient φ = 1 + β determines
    the half-life:
        half_life = -ln(2) / ln(φ)  when 0 < φ < 1

    Mean reversion requires 0 < φ < 1 (equivalently, -1 < β < 0).

    Parameters
    ----------
    x : array-like
        Time series data (typically price levels or spread)

    Returns
    -------
    float
        Half-life in number of periods, or np.inf if series is not mean-reverting
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 3:
        return np.nan

    ts = pd.Series(x)
    lag = ts.shift(1)
    delta = ts - lag
    df = pd.DataFrame({"delta": delta, "lag": lag}).dropna()

    if len(df) < 3:
        return np.nan

    X = sm.add_constant(df["lag"].values)
    y = df["delta"].values

    try:
        res = sm.OLS(y, X).fit()
        beta = res.params[1]  # coefficient on lag
    except Exception:
        return np.nan

    phi = 1.0 + beta

    # Mean reversion requires 0 < phi < 1
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.inf

    return float(-np.log(2.0) / np.log(phi))


def stationarity_score(
    hurst: float,
    half_life: float,
    hl_weight: float = 0.5,
    hl_scale: float = 100.0
) -> float:
    """
    Combined stationarity score: lower = more stationary.

    Combines Hurst exponent and half-life into a single metric for ranking.
    The half-life is log-transformed and normalized to prevent it from
    dominating the score.

    Parameters
    ----------
    hurst : float
        Hurst exponent (0 to 1, lower = more mean-reverting)
    half_life : float
        Mean-reversion half-life in periods
    hl_weight : float
        Weight for normalized half-life component (default: 0.5)
    hl_scale : float
        Normalization scale for half-life (default: 100.0)
        When half_life = hl_scale, the normalized value ≈ 1.0

    Returns
    -------
    float
        Stationarity score (lower = more stationary)

    Examples
    --------
    >>> stationarity_score(0.3, 5)   # Mean-reverting: ~0.38
    >>> stationarity_score(0.5, 100) # Random walk-ish: ~1.0
    >>> stationarity_score(0.7, 500) # Trending: ~1.37
    """
    # Handle invalid inputs
    if not np.isfinite(hurst):
        hurst = 1.0  # Assume worst case (trending)
    if not np.isfinite(half_life) or half_life <= 0:
        half_life = 1e9  # Very large = no mean reversion

    # Normalize half_life to similar scale as Hurst (roughly 0-2)
    normalized_hl = np.log1p(half_life) / np.log1p(hl_scale)

    return hurst + hl_weight * normalized_hl


def rank_stationarity(
    series_dict: Dict[str, Union[np.ndarray, pd.Series]],
    max_lag: int = 20,
    hl_weight: float = 0.5
) -> pd.DataFrame:
    """
    Rank multiple time series by their stationarity properties.

    Computes Hurst exponent and half-life for each series, then ranks
    them by combined stationarity score (lower rank = more stationary).

    Parameters
    ----------
    series_dict : dict
        Dictionary mapping series names to array-like data
    max_lag : int
        Maximum lag for Hurst exponent calculation
    hl_weight : float
        Weight for half-life in combined score

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: name, hurst, half_life, score, rank
        Sorted by rank (most stationary first)
    """
    results = []
    for name, ts in series_dict.items():
        h = get_hurst_exponent(ts, max_lag=max_lag)
        hl = get_half_life(ts)
        score = stationarity_score(h, hl, hl_weight=hl_weight)
        results.append({
            'name': name,
            'hurst': h,
            'half_life': hl,
            'score': score
        })

    df = pd.DataFrame(results)
    df['rank'] = df['score'].rank(method='min')
    return df.sort_values('rank').reset_index(drop=True)


# --- Example / Test ---
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000

    # 1. Mean-Reverting Series (Ornstein-Uhlenbeck)
    mu, theta, sigma = 100, 0.1, 2.0
    ou_prices = np.zeros(n)
    ou_prices[0] = 100
    for t in range(1, n):
        dx = theta * (mu - ou_prices[t - 1]) + sigma * np.random.normal()
        ou_prices[t] = ou_prices[t - 1] + dx

    # 2. Random Walk
    rw_prices = 100 + np.cumsum(np.random.normal(0, 1, n))

    # 3. Trending Series (momentum)
    trend = np.linspace(0, 50, n)
    trending_prices = 100 + trend + np.cumsum(np.random.normal(0, 0.5, n))

    # 4. White Noise (stationary returns)
    white_noise = np.random.normal(0, 1, n)

    # Calculate metrics
    print("=" * 60)
    print("STATIONARITY ANALYSIS")
    print("=" * 60)

    series = {
        "OU (Mean-Reverting)": ou_prices,
        "Random Walk": rw_prices,
        "Trending": trending_prices,
        "White Noise": white_noise,
    }

    for name, data in series.items():
        h = get_hurst_exponent(data)
        hl = get_half_life(data)
        score = stationarity_score(h, hl)
        print(f"\n{name}:")
        print(f"  Hurst Exponent: {h:.4f}")
        print(f"  Half-Life:      {hl:.2f} periods" if np.isfinite(hl) else f"  Half-Life:      ∞ (non-reverting)")
        print(f"  Score:          {score:.4f}")

    print("\n" + "=" * 60)
    print("RANKING (lower = more stationary)")
    print("=" * 60)
    ranking = rank_stationarity(series)
    print(ranking.to_string(index=False))

    # Theoretical half-life for OU process
    print(f"\n[Theoretical OU Half-Life: {-np.log(2) / np.log(1 - theta):.2f} periods]")
