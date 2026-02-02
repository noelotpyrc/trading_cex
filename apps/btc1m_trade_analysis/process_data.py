"""
VWAP Session Analysis - Data Processing

Processes 1-minute BTCUSDT data, calculates VWAP 15/30/60, and assigns session IDs.
Also pre-computes features for momentum strategy filters:
- Parkinson volatility (30/1440 bars)
- Volume SMA and ratios
- Intrabar returns and VWAP gaps
- Cross direction detection

Outputs processed CSV with session columns for use in Streamlit app and strategy backtests.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from feature_engineering.primitives import rolling_vwap, parkinson_volatility
from feature_engineering.derived_features import taker_imbalance_ema_alt, avg_trade_size, avg_trade_size_zscore

# =============================================================================
# CONFIG
# =============================================================================

INPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/binance_btcusdt_perp_1m/BTCUSDT-1m-merged.csv")
# OUTPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/binance_btcusdt_perp_1m/BTCUSDT-1m-vwap-sessions.csv")
OUTPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/binance_btcusdt_perp_1m/BTCUSDT-1m-features-lite.csv")
FILTER_DATE = "2022-01-01"

# =============================================================================
# SESSION ASSIGNMENT
# =============================================================================

def assign_sessions(
    close: pd.Series,
    vwap: pd.Series,
    eps: float = 0.0,  # set to small value like 1e-8 if you want tolerance
) -> tuple[pd.Series, pd.Series]:
    """
    Assign session IDs based on close vs VWAP.
    A session is a continuous period where close stays above or below VWAP.
    Equality (within eps) is treated as 'no change' (carry forward).
    """
    if not close.index.equals(vwap.index):
        raise ValueError("close and vwap must have the same index")

    valid = close.notna() & vwap.notna()

    # raw side: 1 above, 0 below, NaN for "equal/unknown"
    side = pd.Series(np.nan, index=close.index, dtype="float")
    side[valid & (close > vwap + eps)] = 1.0
    side[valid & (close < vwap - eps)] = 0.0

    # equality/eps-band -> carry forward last known side
    side = side.ffill()

    # session id: new session whenever side changes (only when side is known)
    change = side.ne(side.shift()) & side.notna()
    session_id = change.cumsum()

    # keep NaN where side never became known
    session_id = session_id.where(side.notna())

    session_type = side.map({1.0: "above_vwap", 0.0: "below_vwap"}).astype("object")

    return session_id, session_type


def rolling_vwap_hlc(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, window: int) -> pd.Series:
    """
    Rolling VWAP using HLC/3 typical price (legacy formula).
    This matches the original vwap_momentum_parkinson.py calculation.
    """
    typical_price = (high + low + close) / 3
    tpv = typical_price * volume
    return tpv.rolling(window=window).sum() / volume.rolling(window=window).sum()


def detect_cross_direction(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """
    Detect VWAP cross direction at each bar.

    Returns:
        Series with 'above'/'below'/None:
        - 'above': price crossed from below to above VWAP at this bar
        - 'below': price crossed from above to below VWAP at this bar
        - None: no cross at this bar
    """
    prev_above = (close.shift(1) > vwap.shift(1))
    curr_above = (close > vwap)

    cross_direction = pd.Series(None, index=close.index, dtype="object")
    cross_direction[~prev_above & curr_above] = 'above'
    cross_direction[prev_above & ~curr_above] = 'below'

    return cross_direction


def intrabar_noise_ratio(high: pd.Series, low: pd.Series, open_: pd.Series,
                          close: pd.Series, window: int, eps: float = 1e-9) -> pd.Series:
    """
    Intrabar noise ratio: Σ(H-L) / Σ|C-O| over window.
    
    High values = lots of wicks relative to body (choppy/indecisive)
    Low values = clean moves with small wicks (trending)
    
    Args:
        high, low, open_, close: OHLC data
        window: Lookback window
        eps: Small value to avoid division by zero
    
    Returns:
        pd.Series: Noise ratio (typically > 1)
    """
    range_sum = (high - low).rolling(window, min_periods=1).sum()
    body_sum = (close - open_).abs().rolling(window, min_periods=1).sum()
    return range_sum / (body_sum + eps)


def sum_zscore_gt_threshold(sum_zscore: pd.Series, window: int, 
                             filter_zscore: pd.Series = None, threshold: float = 1.0) -> pd.Series:
    """
    Sum of z-scores over a rolling window, filtering by threshold on a separate z-score.
    
    Args:
        sum_zscore: Z-score to sum (e.g., avg_trade_size_zscore_1440)
        window: Lookback window in bars
        filter_zscore: Z-score to use for threshold check (default: same as sum_zscore)
                       Typically use zscore_60 for filtering, zscore_1440 for summing
        threshold: Only sum where filter_zscore > threshold (default 1.0)
    
    Returns:
        pd.Series: Rolling sum of sum_zscore where filter_zscore > threshold
    """
    if filter_zscore is None:
        filter_zscore = sum_zscore
    zscore_clipped = sum_zscore.where(filter_zscore > threshold, 0)
    return zscore_clipped.rolling(window, min_periods=1).sum()


def sum_signed_zscore_gt_threshold(sum_zscore: pd.Series, intrabar_return: pd.Series,
                                    window: int, filter_zscore: pd.Series = None, 
                                    threshold: float = 1.0) -> pd.Series:
    """
    Sum of signed z-scores (zscore × sign(intrabar_return)), filtering by threshold.
    
    Args:
        sum_zscore: Z-score to sum (e.g., avg_trade_size_zscore_1440)
        intrabar_return: Intrabar return for directional signing
        window: Lookback window in bars
        filter_zscore: Z-score to use for threshold check (default: same as sum_zscore)
                       Typically use zscore_60 for filtering, zscore_1440 for summing
        threshold: Only sum where filter_zscore > threshold (default 1.0)
    
    Returns:
        pd.Series: Rolling sum of signed sum_zscore where filter_zscore > threshold
    """
    if filter_zscore is None:
        filter_zscore = sum_zscore
    sign = np.sign(intrabar_return)
    signed_zscore = sum_zscore * sign
    signed_clipped = signed_zscore.where(filter_zscore > threshold, 0)
    return signed_clipped.rolling(window, min_periods=1).sum()


def vwap_gap_normalized(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """
    VWAP gap normalized by close price: (close - vwap) / close.
    
    Args:
        close: Close price series
        vwap: VWAP series
    
    Returns:
        pd.Series: Normalized gap (positive = close above vwap)
    """
    return (close - vwap) / close

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr_time(tr: pd.Series, window: str = "30min") -> pd.Series:
    # time-based rolling mean
    return tr.rolling(window, min_periods=1).mean()

def atr_ref_quantile(
    atr_fast: pd.Series,
    ref_window: str = "63D",   # ~3 months calendar; or use '90D'
    q: float = 0.90,
    shift: int = 1
) -> pd.Series:
    # rolling quantile of atr_fast; shift to avoid lookahead at decision time
    ref = atr_fast.rolling(ref_window, min_periods=100).quantile(q)
    return ref.shift(shift)

def ref_quantile_bars(
    x: pd.Series,
    lookback_bars: int,
    q: float = 0.75,
    min_periods: int | None = None,
    shift_bars: int = 1,
) -> pd.Series:
    """
    Rolling reference quantile over the past `lookback_bars` bars.
    Uses pandas bar-count rolling window (consistent with other functions).

    shift_bars=1 prevents lookahead (today's bar uses ref built up to prior bar).
    """
    if min_periods is None:
        # require at least some data; tune as needed
        min_periods = max(100, int(0.50 * lookback_bars))  # heuristic

    ref = x.rolling(window=int(lookback_bars), min_periods=min_periods).quantile(q)
    return ref.shift(shift_bars)

# Use example when we already have parkinson30
# move_scale = df["close"] * df["parkinson_30"]  # price units
# parkinson_ref = ref_quantile_bars(move_scale, lookback_bars=63*1440, q=0.75, shift_bars=1)

# Normalize features
# gap_abs = (df["close"] - vwap) / parkinson_ref
# breakout_abs = (df["close"] - rolling_low) / parkinson_ref


def compute_mfe_mae(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute forward MFE (max favorable excursion) and MAE (max adverse excursion).
    
    MFE = max(high over next `window` bars) / close - 1 (positive = upside)
    MAE = min(low over next `window` bars) / close - 1 (negative = downside)
    
    These are shifted backward so the value at bar t represents the excursion
    that WILL happen over the next `window` bars (forward-looking).
    """
    # Rolling max/min shifted to align with entry bar
    fwd_high = high.rolling(window, min_periods=1).max().shift(-window)
    fwd_low = low.rolling(window, min_periods=1).min().shift(-window)
    
    mfe = fwd_high / close - 1
    mae = fwd_low / close - 1
    return mfe, mae


def compute_fwd_return(close: pd.Series, window: int) -> pd.Series:
    """
    Compute forward return: close[t+window] / close[t] - 1.
    """
    return close.shift(-window) / close - 1


def compute_fwd_parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Compute forward-looking Parkinson volatility over next `window` bars.
    
    This is the volatility that WILL happen over the next `window` bars (forward-looking).
    Uses the same Parkinson formula: sqrt(sum(ln(H/L)^2) / (4*ln(2)*n))
    
    Args:
        high: High price series
        low: Low price series
        window: Number of forward bars to calculate volatility over
    
    Returns:
        pd.Series: Forward Parkinson volatility (shifted back to align with entry bar)
    """
    ln_hl = np.log(high / low)
    ln_hl_sq = ln_hl ** 2
    
    # Rolling sum shifted to get forward window
    fwd_sum = ln_hl_sq.rolling(window, min_periods=1).sum().shift(-window)
    fwd_parkinson = np.sqrt(fwd_sum / (4 * np.log(2) * window))
    
    return fwd_parkinson


def lagged_proxy(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fwd_window: int,
    shift_bars: int,
    direction: str = "long",
    parkinson_floor: pd.Series = None,
) -> pd.Series:
    """
    Compute lagged long/short proxy to avoid lookahead.
    
    The proxy is computed as:
    - Long proxy = fwd_return / abs(mae)  
    - Short proxy = -fwd_return / abs(mfe)
    
    To avoid lookahead:
    1. Compute proxy for each bar (which uses forward data)
    2. Shift by `shift_bars` (should be >= fwd_window to ensure no future info)
    
    Args:
        high, low, close: OHLC price series
        fwd_window: Forward window for MFE/MAE/return calculation (e.g., 90, 120, 180 bars)
        shift_bars: How many bars to shift back (should be >= fwd_window)
        direction: "long" or "short"
        parkinson_floor: Optional floor for abs(mae/mfe) to avoid extreme values
    
    Returns:
        pd.Series: Lagged proxy values (shifted to avoid lookahead)
    """
    # Compute forward-looking values
    fwd_ret = compute_fwd_return(close, fwd_window)
    mfe, mae = compute_mfe_mae(high, low, close, fwd_window)
    
    # Apply Parkinson floor to avoid extreme values from tiny MAE/MFE
    if parkinson_floor is not None:
        mae_denom = mae.abs().clip(lower=parkinson_floor)
        mfe_denom = mfe.abs().clip(lower=parkinson_floor)
    else:
        mae_denom = mae.abs()
        mfe_denom = mfe.abs()
    
    # Compute proxy
    if direction == "long":
        proxy = fwd_ret / mae_denom
    else:  # short
        proxy = -fwd_ret / mfe_denom
    
    # Shift to avoid lookahead (proxy at bar t now represents bar t-shift_bars)
    return proxy.shift(shift_bars)


def compute_r_multiple(
    fwd_return: pd.Series,
    mfe: pd.Series,
    mae: pd.Series,
    stop_vol: pd.Series,
    direction: str = "long",
    stop_multiplier: float = 4.0,
    stop_cutoff: float = 0.005,  # 50 bps max stop
) -> pd.Series:
    """
    Compute R-Multiple using volatility as dynamic stop loss.
    
    R-Multiple = fwd_return / effective_stop if survived, else -1 (stopped out)
    
    effective_stop = min(stop_vol * multiplier, stop_cutoff)
    
    For longs: stopped if abs(mae) >= effective_stop
    For shorts: stopped if mfe >= effective_stop
    
    Args:
        fwd_return: Forward return series
        mfe: Max favorable excursion (positive for upward move)
        mae: Max adverse excursion (negative for downward move)
        stop_vol: Volatility series to use as dynamic stop (e.g., parkinson_30)
        direction: "long" or "short"
        stop_multiplier: Multiplier for stop_vol (default=4.0, i.e., stop at 4x parkinson)
        stop_cutoff: Maximum stop loss (default=0.005 = 50bps). If vol*multiplier > cutoff, use cutoff.
    
    Returns:
        pd.Series: R-multiple values (-1 for stopped, else return/effective_stop)
    """
    # Cap the stop at cutoff if volatility-based stop is too large
    effective_stop = (stop_vol * stop_multiplier).clip(upper=stop_cutoff)
    
    if direction == "long":
        # Long: stopped if price dropped >= effective_stop
        stopped = mae.abs() >= effective_stop
        r_mult = np.where(
            stopped,
            -1.0,
            fwd_return / effective_stop
        )
    else:  # short
        # Short: stopped if price rose >= effective_stop
        stopped = mfe >= effective_stop
        r_mult = np.where(
            stopped,
            -1.0,
            -fwd_return / effective_stop
        )
    
    return pd.Series(r_mult, index=fwd_return.index)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_data():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows")

    # Parse datetime
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df = df.sort_values('datetime_utc').reset_index(drop=True)

    # =========================================================================
    # VWAP CALCULATIONS (on full data for warmup)
    # =========================================================================
    print("Calculating VWAP (OHLC/4 formula)...")
    df['vwap_15'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=15
    )
    df['vwap_30'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=30
    )
    df['vwap_60'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=60
    )
    df['vwap_240'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=240
    )
    df['vwap_120'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=120
    )
    df['vwap_180'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=180
    )

    # Legacy VWAP using HLC/3 (for compatibility with old strategy)
    print("Calculating VWAP (HLC/3 legacy formula)...")
    df['vwap_15_hlc'] = rolling_vwap_hlc(
        df['high'], df['low'], df['close'], df['volume'], window=15
    )

    # =========================================================================
    # PARKINSON VOLATILITY (needs 1440 bars warmup)
    # =========================================================================
    print("Calculating Parkinson volatility (30/1440 bars)...")
    df['parkinson_30'] = parkinson_volatility(df['high'], df['low'], window=30)
    # Extra parkinson windows (COMMENTED OUT - only using parkinson_30 with 4x multiplier now)
    # for p_window in [60, 90, 120, 180]:
    #     df[f'parkinson_{p_window}'] = parkinson_volatility(df['high'], df['low'], window=p_window)
    df['parkinson_1440'] = parkinson_volatility(df['high'], df['low'], window=1440)
    df['parkinson_ratio'] = df['parkinson_30'] / df['parkinson_1440']

    # Normalized parkinson ratio: percentile rank of log ratio within 7-day window
    print("Calculating normalized parkinson ratio (percentile rank over 7 days)...")
    # parkinson_lr = np.log(df['parkinson_30'].clip(lower=1e-12)) - np.log(df['parkinson_1440'].clip(lower=1e-12))
    # window_7d = 60 * 24 * 7  # 7 days in minutes

    def rolling_percentile_rank(s, window):
        def pct_rank(x):
            return (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5
        return s.rolling(window, min_periods=window // 2).apply(pct_rank, raw=True)

    # df['parkinson_ratio_pct_7d'] = rolling_percentile_rank(parkinson_lr, window_7d)

    # =========================================================================
    # VOLUME SMA AND RATIOS
    # =========================================================================
    print("Calculating volume SMA and ratios...")
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_sma_30'] = df['volume'].rolling(window=30).mean()
    df['volume_sma_60'] = df['volume'].rolling(window=60).mean()
    df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
    df['volume_ratio_30'] = df['volume'] / df['volume_sma_30']
    df['volume_ratio_60'] = df['volume'] / df['volume_sma_60']

    # =========================================================================
    # TAKER IMBALANCE EMA (ALT) - smoothed directional flow
    # =========================================================================
    print("Calculating taker imbalance EMA (alt)...")
    df['taker_imb_ema_15'] = taker_imbalance_ema_alt(
        df['taker_buy_base_asset_volume'], df['volume'], span=15
    )
    df['taker_imb_ema_30'] = taker_imbalance_ema_alt(
        df['taker_buy_base_asset_volume'], df['volume'], span=30
    )

    # =========================================================================
    # AVERAGE TRADE SIZE AND Z-SCORES
    # =========================================================================
    print("Calculating average trade size and z-scores...")
    df['avg_trade_size'] = avg_trade_size(df['volume'], df['number_of_trades'])
    df['avg_trade_size_zscore_60'] = avg_trade_size_zscore(
        df['volume'], df['number_of_trades'], window=60
    )
    df['avg_trade_size_zscore_240'] = avg_trade_size_zscore(
        df['volume'], df['number_of_trades'], window=240
    )
    df['avg_trade_size_zscore_1440'] = avg_trade_size_zscore(
        df['volume'], df['number_of_trades'], window=1440
    )

    # =========================================================================
    # SUM ZSCORE FEATURES (60m window = 60 bars)
    # =========================================================================
    print("Calculating sum zscore features (1440 zscore, 60m window)...")
    # Intrabar return for signing
    df['intrabar_return'] = (df['close'] - df['open']) / df['open']

    # Sum of zscore_1440 over 60 bars (filter by zscore_60 > 1.0)
    df['sum_zscore_1440_60m'] = sum_zscore_gt_threshold(
        df['avg_trade_size_zscore_1440'], window=60,
        filter_zscore=df['avg_trade_size_zscore_60'], threshold=1.0
    )

    # Sum of signed zscore_1440 over 60 bars (filter by zscore_60 > 1.0)
    df['sum_signed_zscore_1440_60m'] = sum_signed_zscore_gt_threshold(
        df['avg_trade_size_zscore_1440'], df['intrabar_return'], window=60,
        filter_zscore=df['avg_trade_size_zscore_60'], threshold=1.0
    )

    # =========================================================================
    # PER-BAR FEATURES (no warmup needed)
    # =========================================================================
    print("Calculating per-bar features...")
    # Note: intrabar_return already calculated above for sum zscore features

    # Intrabar return z-score (rolling window)
    print("Calculating intrabar return z-score (1440 bars)...")
    intrabar_mean = df['intrabar_return'].rolling(window=1440, min_periods=100).mean()
    intrabar_std = df['intrabar_return'].rolling(window=1440, min_periods=100).std()
    df['intrabar_return_zscore_1440'] = (df['intrabar_return'] - intrabar_mean) / intrabar_std

    # Multi-bar cumulative returns and z-scores
    # These match the strategy's cumulative move: from bar 1 open to bar N close
    # cum_return_Nbar = current close / (N-1 bars ago open) - 1
    print("Calculating multi-bar cumulative returns (2-5 bars)...")
    for n_bars_span in [2, 3, 4, 5]:
        shift = n_bars_span - 1
        col_name = f'cum_return_{n_bars_span}bar'
        df[col_name] = df['close'] / df['open'].shift(shift) - 1

        # Z-score with 1440-bar rolling window
        zscore_col = f'{col_name}_zscore_1440'
        col_mean = df[col_name].rolling(window=1440, min_periods=100).mean()
        col_std = df[col_name].rolling(window=1440, min_periods=100).std()
        df[zscore_col] = (df[col_name] - col_mean) / col_std

    # Normalized cumulative returns (by parkinson_30)
    print("Calculating normalized cumulative returns (by parkinson_30)...")
    for n_bars_span in [3, 4, 5]:
        cum_col = f'cum_return_{n_bars_span}bar'
        norm_col = f'cum_return_{n_bars_span}bar_norm_p30'
        df[norm_col] = df[cum_col] / df['parkinson_30'].clip(lower=1e-9)

    # =========================================================================
    # EFFICIENCY RATIO (path efficiency)
    # =========================================================================
    print("Calculating efficiency ratio (60/240/360 bars)...")
    eps = 1e-9
    for window in [60, 240, 360]:
        net_return = df['close'] - df['close'].shift(window)
        abs_path = df['close'].diff().abs().rolling(window, min_periods=window).sum()
        raw_efficiency = net_return / (abs_path + eps)
        df[f'efficiency_ratio_{window}'] = raw_efficiency.abs().clip(0.0, 1.0)

    # VWAP gap normalized with percentile rank (dependencies for vwap_gap_norm_*_pct)
    print("Calculating VWAP gap normalized percentiles...")
    df['vwap_90'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=90
    )
    df['vwap_gap_norm_90'] = vwap_gap_normalized(df['close'], df['vwap_90'])
    df['vwap_gap_norm_120'] = vwap_gap_normalized(df['close'], df['vwap_120'])
    df['vwap_gap_norm_180'] = vwap_gap_normalized(df['close'], df['vwap_180'])
    window_30d = 60 * 24 * 30
    for gap_w in [90, 120, 180]:
        gap_col = f'vwap_gap_norm_{gap_w}'
        df[f'{gap_col}_pct_7d'] = rolling_percentile_rank(df[gap_col], window_7d)
        df[f'{gap_col}_pct_30d'] = rolling_percentile_rank(df[gap_col], window_30d)

    # =========================================================================
    # CLOSE LOCATION VALUE (COMMENTED OUT - not in output)
    # =========================================================================
    # print("Calculating close location value (60/240/1440)...")
    # for window in [60, 240, 1440]:
    #     hh = df['high'].rolling(window, min_periods=1).max()
    #     ll = df['low'].rolling(window, min_periods=1).min()
    #     df[f'close_location_value_{window}'] = ((df['close'] - ll) / (hh - ll + 1e-9)).clip(0, 1)

    # =========================================================================
    # INTRABAR NOISE RATIO (COMMENTED OUT - not in output)
    # =========================================================================
    # print("Calculating intrabar noise ratio (30/60/90)...")
    # for window in [30, 60, 90]:
    #     df[f'intrabar_noise_ratio_{window}'] = intrabar_noise_ratio(
    #         df['high'], df['low'], df['open'], df['close'], window
    #     )

    # =========================================================================
    # BREAKOUT DISTANCE FROM LOW (COMMENTED OUT - not in output)
    # =========================================================================
    # print("Calculating breakout distance from low (60/240/1440)...")
    # for window in [60, 240, 1440]:
    #     rolling_low = df['low'].rolling(window, min_periods=1).min()
    #     df[f'breakout_dist_from_low_{window}'] = np.log(df['close'] / rolling_low)
    # df['breakout_dist_from_low_240_pct_7d'] = rolling_percentile_rank(
    #     df['breakout_dist_from_low_240'], window_7d
    # )
    # df['breakout_dist_from_low_240_pct_30d'] = rolling_percentile_rank(
    #     df['breakout_dist_from_low_240'], window_30d
    # )

    # =========================================================================
    # FORWARD TARGETS (fwd_return, mfe, mae, long_proxy, short_proxy)
    # These use forward-looking data and should NOT be used as features!
    # =========================================================================
    print("Calculating forward targets (fwd_return, mfe, mae, proxy) for 90/120/180...")
    
    for w in [90, 120, 180]:
        # Forward return
        df[f'fwd_return_{w}'] = compute_fwd_return(df['close'], w)
        
        # Forward Parkinson volatility (volatility over next w bars)
        df[f'fwd_parkinson_{w}'] = compute_fwd_parkinson(df['high'], df['low'], w)
        
        # MFE and MAE
        mfe, mae = compute_mfe_mae(df['high'], df['low'], df['close'], w)
        df[f'mfe_{w}'] = mfe
        df[f'mae_{w}'] = mae
        
        # OLD PROXY LOGIC (commented out - replaced by R-multiple)
        # mae_floored = df[f'mae_{w}'].abs().clip(lower=df['parkinson_30'])
        # df[f'long_proxy_{w}'] = df[f'fwd_return_{w}'] / mae_floored
        # mfe_floored = df[f'mfe_{w}'].abs().clip(lower=df['parkinson_30'])
        # df[f'short_proxy_{w}'] = -df[f'fwd_return_{w}'] / mfe_floored
        
        # R-Multiple with parkinson_30 as dynamic stop (use for long_proxy/short_proxy for app compatibility)
        df[f'long_proxy_{w}'] = compute_r_multiple(
            df[f'fwd_return_{w}'], df[f'mfe_{w}'], df[f'mae_{w}'],
            df['parkinson_30'], direction="long"
        )
        df[f'short_proxy_{w}'] = compute_r_multiple(
            df[f'fwd_return_{w}'], df[f'mfe_{w}'], df[f'mae_{w}'],
            df['parkinson_30'], direction="short"
        )
        
        # R-Multiple with various parkinson windows (COMMENTED OUT - need data analysis first)
        # for p_window in [30, 60, 90, 120, 180]:
        #     df[f'r_mult_long_{w}_p{p_window}'] = compute_r_multiple(
        #         df[f'fwd_return_{w}'], df[f'mfe_{w}'], df[f'mae_{w}'],
        #         df[f'parkinson_{p_window}'], direction="long"
        #     )
        #     df[f'r_mult_short_{w}_p{p_window}'] = compute_r_multiple(
        #         df[f'fwd_return_{w}'], df[f'mfe_{w}'], df[f'mae_{w}'],
        #         df[f'parkinson_{p_window}'], direction="short"
        #     )



    # =========================================================================
    # LAGGED PROXY (COMMENTED OUT - using R-multiple now, need to redesign)
    # =========================================================================
    # print("Calculating lagged proxy from filtered subset (180-min lookback)...")
    # 
    # # Create filtered subset
    # subset_mask = (
    #     (df['parkinson_ratio'] >= 1.5) & 
    #     (df['volume_ratio_30'] >= 1.0) &
    #     (df['avg_trade_size_zscore_60'] >= 1.0)
    # )
    # 
    # # Select columns needed for proxy calculation
    # proxy_input_cols = ['datetime_utc', 'parkinson_30'] + [
    #     f'fwd_return_{w}' for w in [90, 120, 180]
    # ] + [
    #     f'mfe_{w}' for w in [90, 120, 180]
    # ] + [
    #     f'mae_{w}' for w in [90, 120, 180]
    # ]
    # df_subset = df[subset_mask][proxy_input_cols].copy()
    # print(f"  Subset size: {len(df_subset):,} rows ({len(df_subset)/len(df)*100:.1f}%)")
    # 
    # # Compute proxy on subset (with parkinson floor)
    # for w in [90, 120, 180]:
    #     mae_floored = df_subset[f'mae_{w}'].abs().clip(lower=df_subset['parkinson_30'])
    #     mfe_floored = df_subset[f'mfe_{w}'].abs().clip(lower=df_subset['parkinson_30'])
    #     df_subset[f'long_proxy_{w}'] = df_subset[f'fwd_return_{w}'] / mae_floored
    #     df_subset[f'short_proxy_{w}'] = -df_subset[f'fwd_return_{w}'] / mfe_floored
    # 
    # # Sort by datetime for merge_asof
    # df_subset = df_subset.sort_values('datetime_utc').reset_index(drop=True)
    # 
    # # Create lookup table with proxy values
    # proxy_cols = [f'long_proxy_{w}' for w in [90, 120, 180]] + [f'short_proxy_{w}' for w in [90, 120, 180]]
    # df_lookup = df_subset[['datetime_utc'] + proxy_cols].copy()
    # 
    # # Create a version of the subset with shifted lookup time (T + 180 min)
    # # So when we merge at time T, we find bars where original_time <= T - 180 min
    # df_lookup['lookup_time'] = df_lookup['datetime_utc'] + pd.Timedelta(minutes=180)
    # 
    # # Use merge_asof to find for each bar, the most recent bar at least 180 min earlier
    # df_subset_sorted = df_subset[['datetime_utc']].copy()
    # df_result = pd.merge_asof(
    #     df_subset_sorted,
    #     df_lookup[['lookup_time'] + proxy_cols].rename(columns={
    #         **{c: f'lagged_{c}' for c in proxy_cols}
    #     }),
    #     left_on='datetime_utc',
    #     right_on='lookup_time',
    #     direction='backward'
    # )
    # 
    # # Rename back to lagged_long_proxy_* and lagged_short_proxy_*
    # lagged_cols = [f'lagged_long_proxy_{w}' for w in [90, 120, 180]] + [f'lagged_short_proxy_{w}' for w in [90, 120, 180]]
    # 
    # # Set datetime as index for rolling operations
    # df_result = df_result.set_index('datetime_utc').sort_index()
    # 
    # # Compute 30D rolling median on lagged proxy
    # median_cols = {}
    # for col in lagged_cols:
    #     median_col = f'{col}_median_30d'
    #     df_result[median_col] = df_result[col].rolling('30D', min_periods=10).median()
    #     median_cols[col] = median_col
    # 
    # # Reset index and prepare for join
    # output_cols_final = ['datetime_utc'] + lagged_cols + list(median_cols.values())
    # df_result = df_result.reset_index()[output_cols_final]
    # 
    # # Join back to full dataframe
    # df = df.merge(df_result, on='datetime_utc', how='left')
    # 
    # non_null_count = df[lagged_cols[0]].notna().sum()
    # print(f"  Rows with lagged proxy: {non_null_count:,} ({non_null_count/len(df)*100:.1f}%)")

    # =========================================================================
    # FILTER TO 2022+ (after all warmup-dependent calculations)
    # =========================================================================
    print(f"Filtering to data after {FILTER_DATE}...")
    df = df[df['datetime_utc'] >= FILTER_DATE].reset_index(drop=True)
    print(f"Filtered to {len(df):,} rows")

    # =========================================================================
    # CROSS DIRECTION DETECTION (needed for VWAP cross rate)
    # =========================================================================
    print("Detecting cross directions...")
    df['cross_direction_15'] = detect_cross_direction(df['close'], df['vwap_15'])
    # df['cross_direction_15_hlc'] = detect_cross_direction(df['close'], df['vwap_15_hlc'])  # COMMENTED OUT - vwap_15_hlc not needed
    df['cross_direction_30'] = detect_cross_direction(df['close'], df['vwap_30'])
    df['cross_direction_60'] = detect_cross_direction(df['close'], df['vwap_60'])
    # df['cross_direction_90'] = detect_cross_direction(df['close'], df['vwap_90'])  # COMMENTED OUT - vwap_90 not computed

    # =========================================================================
    # VWAP CROSS RATE (rate windows: 30/60/90/180)
    # =========================================================================
    print("Calculating VWAP cross rate (15/30/60 x 30/60/90/180)...")
    for vwap_w in [15, 30, 60]:
        cross_col = f'cross_direction_{vwap_w}'
        if cross_col in df.columns:
            numeric = df[cross_col].map({'above': 1, 'below': -1}).fillna(0)
            crosses = (numeric != numeric.shift(1)).astype(int)
            for rate_w in [30, 60, 90, 180]:
                df[f'vwap_{vwap_w}_cross_rate_{rate_w}'] = crosses.rolling(rate_w, min_periods=1).sum()

    # =========================================================================
    # SESSION ASSIGNMENT (COMMENTED OUT - not in output)
    # =========================================================================
    # print("Assigning sessions for VWAP 15...")
    # df['session_id_vwap15'], df['session_type_vwap15'] = assign_sessions(
    #     df['close'], df['vwap_15']
    # )

    # print("Assigning sessions for VWAP 30...")
    # df['session_id_vwap30'], df['session_type_vwap30'] = assign_sessions(
    #     df['close'], df['vwap_30']
    # )

    # =========================================================================
    # SELECT AND ORDER OUTPUT COLUMNS
    # =========================================================================
    output_cols = [
        # Core OHLCV
        'datetime_utc', 'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        # VWAP (OHLC/4 formula) - COMMENTED OUT
        # 'vwap_15', 'vwap_30', 'vwap_60', 'vwap_90', 'vwap_120', 'vwap_180', 'vwap_240',
        # VWAP (HLC/3 legacy formula)
        # 'vwap_15_hlc',
        # Parkinson volatility (only 30 and 1440 now, others commented out)
        'parkinson_30',
        # 'parkinson_60', 'parkinson_90', 'parkinson_120', 'parkinson_180',
        'parkinson_1440', 'parkinson_ratio', # 'parkinson_ratio_pct_7d',
        # Volume features
        # 'volume_sma_20', 'volume_sma_30', 'volume_sma_60',
        'volume_ratio_20', 'volume_ratio_30', 'volume_ratio_60',
        # Taker imbalance EMA (alt) - COMMENTED OUT
        'taker_imb_ema_15', 'taker_imb_ema_30',
        # Average trade size
        'avg_trade_size', 'avg_trade_size_zscore_60', # 'avg_trade_size_zscore_240', 'avg_trade_size_zscore_1440',
        # Sum zscore features (60m window) - COMMENTED OUT
        # 'sum_zscore_1440_60m', 'sum_signed_zscore_1440_60m',
        # Per-bar features
        'intrabar_return', 'intrabar_return_zscore_1440',
        # Multi-bar cumulative returns
        'cum_return_2bar', 'cum_return_2bar_zscore_1440',
        'cum_return_3bar', 'cum_return_3bar_zscore_1440', 'cum_return_3bar_norm_p30',
        'cum_return_4bar', 'cum_return_4bar_zscore_1440', 'cum_return_4bar_norm_p30',
        'cum_return_5bar', 'cum_return_5bar_zscore_1440', 'cum_return_5bar_norm_p30',
        # Efficiency ratio - COMMENTED OUT
        'efficiency_ratio_60', 'efficiency_ratio_240', 'efficiency_ratio_360',
        # VWAP gaps - COMMENTED OUT
        # 'vwap_gap_15', 'vwap_gap_15_hlc', 'vwap_gap_30', 'vwap_gap_60',
        # 'vwap_gap_90', 'vwap_gap_120', 'vwap_gap_180',
        # VWAP gap normalized - COMMENTED OUT
        # 'vwap_gap_norm_90', 'vwap_gap_norm_120', 'vwap_gap_norm_180',
        'vwap_gap_norm_90_pct_7d', 'vwap_gap_norm_90_pct_30d',
        'vwap_gap_norm_120_pct_7d', 'vwap_gap_norm_120_pct_30d',
        'vwap_gap_norm_180_pct_7d', 'vwap_gap_norm_180_pct_30d',
        # Close location value - COMMENTED OUT
        # 'close_location_value_60', 'close_location_value_240', 'close_location_value_1440',
        # Intrabar noise ratio - COMMENTED OUT
        # 'intrabar_noise_ratio_30', 'intrabar_noise_ratio_60', 'intrabar_noise_ratio_90',
        # Breakout distance from low - COMMENTED OUT
        # 'breakout_dist_from_low_60', 'breakout_dist_from_low_240', 'breakout_dist_from_low_1440',
        # 'breakout_dist_from_low_240_pct_7d', 'breakout_dist_from_low_240_pct_30d',
        # Cross detection - COMMENTED OUT
        # 'cross_direction_15', 'cross_direction_15_hlc', 'cross_direction_30', 'cross_direction_60', 'cross_direction_90',
        # VWAP cross rate
        # 'vwap_15_cross_rate_30', 'vwap_15_cross_rate_60', 'vwap_15_cross_rate_90', 'vwap_15_cross_rate_180',
        #'vwap_30_cross_rate_30', 'vwap_30_cross_rate_60', 'vwap_30_cross_rate_90', 'vwap_30_cross_rate_180',
        'vwap_60_cross_rate_30', 'vwap_60_cross_rate_60', 'vwap_60_cross_rate_90', 'vwap_60_cross_rate_180',
        # Forward targets (FORWARD-LOOKING - do NOT use as features!)
        'fwd_return_90', 'fwd_return_120', 'fwd_return_180',
        'fwd_parkinson_90', 'fwd_parkinson_120', 'fwd_parkinson_180',
        'mfe_90', 'mfe_120', 'mfe_180',
        'mae_90', 'mae_120', 'mae_180',
        'long_proxy_90', 'long_proxy_120', 'long_proxy_180',
        'short_proxy_90', 'short_proxy_120', 'short_proxy_180',
        # R-Multiple with variable parkinson windows (COMMENTED OUT - need data analysis first)
        # 'r_mult_long_90_p30', 'r_mult_long_90_p60', 'r_mult_long_90_p90', 'r_mult_long_90_p120', 'r_mult_long_90_p180',
        # 'r_mult_long_120_p30', 'r_mult_long_120_p60', 'r_mult_long_120_p90', 'r_mult_long_120_p120', 'r_mult_long_120_p180',
        # 'r_mult_long_180_p30', 'r_mult_long_180_p60', 'r_mult_long_180_p90', 'r_mult_long_180_p120', 'r_mult_long_180_p180',
        # 'r_mult_short_90_p30', 'r_mult_short_90_p60', 'r_mult_short_90_p90', 'r_mult_short_90_p120', 'r_mult_short_90_p180',
        # 'r_mult_short_120_p30', 'r_mult_short_120_p60', 'r_mult_short_120_p90', 'r_mult_short_120_p120', 'r_mult_short_120_p180',
        # 'r_mult_short_180_p30', 'r_mult_short_180_p60', 'r_mult_short_180_p90', 'r_mult_short_180_p120', 'r_mult_short_180_p180',
        # Lagged proxy (COMMENTED OUT - disabled)
        # 'lagged_long_proxy_90', 'lagged_long_proxy_120', 'lagged_long_proxy_180',
        # 'lagged_short_proxy_90', 'lagged_short_proxy_120', 'lagged_short_proxy_180',
        # Lagged proxy 30D rolling median (COMMENTED OUT - disabled)
        # 'lagged_long_proxy_90_median_30d', 'lagged_long_proxy_120_median_30d', 'lagged_long_proxy_180_median_30d',
        # 'lagged_short_proxy_90_median_30d', 'lagged_short_proxy_120_median_30d', 'lagged_short_proxy_180_median_30d',
        # Sessions - COMMENTED OUT
        # 'session_id_vwap15', 'session_type_vwap15',
        # 'session_id_vwap30', 'session_type_vwap30'
    ]
    df = df[output_cols]

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================
    print(f"Saving to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows with {len(output_cols)} columns")

    # =========================================================================
    # SUMMARY STATS
    # =========================================================================
    print("\n=== Summary ===")
    print(f"Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
    # Feature-specific stats - COMMENTED OUT
    # print(f"Parkinson ratio - mean: {df['parkinson_ratio'].mean():.3f}, "
    #       f"median: {df['parkinson_ratio'].median():.3f}")
    # print(f"Parkinson ratio >= 1.5: {(df['parkinson_ratio'] >= 1.5).mean() * 100:.1f}%")
    # print(f"Volume ratio 20 - mean: {df['volume_ratio_20'].mean():.3f}")
    # print(f"Avg trade size - mean: {df['avg_trade_size'].mean():.2f}, "
    #       f"median: {df['avg_trade_size'].median():.2f}")

    return df


if __name__ == "__main__":
    process_data()
