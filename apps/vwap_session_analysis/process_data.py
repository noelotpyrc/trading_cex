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
OUTPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/binance_btcusdt_perp_1m/BTCUSDT-1m-vwap-sessions.csv")
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
    df['parkinson_1440'] = parkinson_volatility(df['high'], df['low'], window=1440)
    df['parkinson_ratio'] = df['parkinson_30'] / df['parkinson_1440']

    # Normalized parkinson ratio: percentile rank of log ratio within 7-day window
    print("Calculating normalized parkinson ratio (percentile rank over 7 days)...")
    parkinson_lr = np.log(df['parkinson_30'].clip(lower=1e-12)) - np.log(df['parkinson_1440'].clip(lower=1e-12))
    window_7d = 60 * 24 * 7  # 7 days in minutes

    def rolling_percentile_rank(s, window):
        def pct_rank(x):
            return (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5
        return s.rolling(window, min_periods=window // 2).apply(pct_rank, raw=True)

    df['parkinson_ratio_pct_7d'] = rolling_percentile_rank(parkinson_lr, window_7d)

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

    # =========================================================================
    # EFFICIENCY RATIO (path efficiency)
    # =========================================================================
    # Return efficiency: abs(net return / sum of absolute bar-to-bar changes)
    # 1 = straight path, 0 = choppy with no net change
    print("Calculating efficiency ratio (60/240/360 bars)...")
    eps = 1e-9
    for window in [60, 240, 360]:
        net_return = df['close'] - df['close'].shift(window)
        abs_path = df['close'].diff().abs().rolling(window, min_periods=window).sum()
        raw_efficiency = net_return / (abs_path + eps)
        df[f'efficiency_ratio_{window}'] = raw_efficiency.abs().clip(0.0, 1.0)

    # VWAP gap: close - vwap_15
    df['vwap_gap_15'] = df['close'] - df['vwap_15']

    # Legacy VWAP gap (HLC/3)
    df['vwap_gap_15_hlc'] = df['close'] - df['vwap_15_hlc']

    # VWAP 30 gap
    df['vwap_gap_30'] = df['close'] - df['vwap_30']

    # VWAP 60, 90, 120, 180 gaps
    df['vwap_gap_60'] = df['close'] - df['vwap_60']
    df['vwap_90'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=90
    )
    df['vwap_gap_90'] = df['close'] - df['vwap_90']
    df['vwap_gap_120'] = df['close'] - df['vwap_120']
    df['vwap_gap_180'] = df['close'] - df['vwap_180']

    # VWAP gap normalized (close - vwap) / close
    df['vwap_gap_norm_90'] = vwap_gap_normalized(df['close'], df['vwap_90'])
    df['vwap_gap_norm_120'] = vwap_gap_normalized(df['close'], df['vwap_120'])
    df['vwap_gap_norm_180'] = vwap_gap_normalized(df['close'], df['vwap_180'])

    # =========================================================================
    # CLOSE LOCATION VALUE (windows: 60/240/1440)
    # =========================================================================
    print("Calculating close location value (60/240/1440)...")
    for window in [60, 240, 1440]:
        hh = df['high'].rolling(window, min_periods=1).max()
        ll = df['low'].rolling(window, min_periods=1).min()
        df[f'close_location_value_{window}'] = ((df['close'] - ll) / (hh - ll + 1e-9)).clip(0, 1)

    # =========================================================================
    # INTRABAR NOISE RATIO (windows: 30/60/90)
    # =========================================================================
    print("Calculating intrabar noise ratio (30/60/90)...")
    for window in [30, 60, 90]:
        df[f'intrabar_noise_ratio_{window}'] = intrabar_noise_ratio(
            df['high'], df['low'], df['open'], df['close'], window
        )

    # =========================================================================
    # BREAKOUT DISTANCE FROM LOW (windows: 60/240/1440)
    # =========================================================================
    print("Calculating breakout distance from low (60/240/1440)...")
    for window in [60, 240, 1440]:
        rolling_low = df['low'].rolling(window, min_periods=1).min()
        df[f'breakout_dist_from_low_{window}'] = np.log(df['close'] / rolling_low)

    # =========================================================================
    # FILTER TO 2022+ (after all warmup-dependent calculations)
    # =========================================================================
    print(f"Filtering to data after {FILTER_DATE}...")
    df = df[df['datetime_utc'] >= FILTER_DATE].reset_index(drop=True)
    print(f"Filtered to {len(df):,} rows")

    # =========================================================================
    # CROSS DIRECTION DETECTION (after filtering, uses only prev bar)
    # =========================================================================
    print("Detecting cross directions...")
    df['cross_direction_15'] = detect_cross_direction(df['close'], df['vwap_15'])
    df['cross_direction_15_hlc'] = detect_cross_direction(df['close'], df['vwap_15_hlc'])
    df['cross_direction_30'] = detect_cross_direction(df['close'], df['vwap_30'])
    df['cross_direction_60'] = detect_cross_direction(df['close'], df['vwap_60'])
    df['cross_direction_90'] = detect_cross_direction(df['close'], df['vwap_90'])

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
    # SESSION ASSIGNMENT
    # =========================================================================
    print("Assigning sessions for VWAP 15...")
    df['session_id_vwap15'], df['session_type_vwap15'] = assign_sessions(
        df['close'], df['vwap_15']
    )

    print("Assigning sessions for VWAP 30...")
    df['session_id_vwap30'], df['session_type_vwap30'] = assign_sessions(
        df['close'], df['vwap_30']
    )

    # =========================================================================
    # SELECT AND ORDER OUTPUT COLUMNS
    # =========================================================================
    output_cols = [
        # Core OHLCV
        'datetime_utc', 'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        # VWAP (OHLC/4 formula)
        'vwap_15', 'vwap_30', 'vwap_60', 'vwap_90', 'vwap_120', 'vwap_180', 'vwap_240',
        # VWAP (HLC/3 legacy formula)
        'vwap_15_hlc',
        # Parkinson volatility
        'parkinson_30', 'parkinson_1440', 'parkinson_ratio', 'parkinson_ratio_pct_7d',
        # Volume features
        'volume_sma_20', 'volume_sma_30', 'volume_sma_60',
        'volume_ratio_20', 'volume_ratio_30', 'volume_ratio_60',
        # Taker imbalance EMA (alt)
        'taker_imb_ema_15', 'taker_imb_ema_30',
        # Average trade size
        'avg_trade_size', 'avg_trade_size_zscore_60', 'avg_trade_size_zscore_240', 'avg_trade_size_zscore_1440',
        # Sum zscore features (60m window)
        'sum_zscore_1440_60m', 'sum_signed_zscore_1440_60m',
        # Per-bar features
        'intrabar_return', 'intrabar_return_zscore_1440',
        # Multi-bar cumulative returns
        'cum_return_2bar', 'cum_return_2bar_zscore_1440',
        'cum_return_3bar', 'cum_return_3bar_zscore_1440',
        'cum_return_4bar', 'cum_return_4bar_zscore_1440',
        'cum_return_5bar', 'cum_return_5bar_zscore_1440',
        # Efficiency ratio
        'efficiency_ratio_60', 'efficiency_ratio_240', 'efficiency_ratio_360',
        # VWAP gaps
        'vwap_gap_15', 'vwap_gap_15_hlc', 'vwap_gap_30', 'vwap_gap_60',
        'vwap_gap_90', 'vwap_gap_120', 'vwap_gap_180',
        # VWAP gap normalized
        'vwap_gap_norm_90', 'vwap_gap_norm_120', 'vwap_gap_norm_180',
        # Close location value
        'close_location_value_60', 'close_location_value_240', 'close_location_value_1440',
        # Intrabar noise ratio
        'intrabar_noise_ratio_30', 'intrabar_noise_ratio_60', 'intrabar_noise_ratio_90',
        # Breakout distance from low
        'breakout_dist_from_low_60', 'breakout_dist_from_low_240', 'breakout_dist_from_low_1440',
        # Cross detection
        'cross_direction_15', 'cross_direction_15_hlc', 'cross_direction_30', 'cross_direction_60', 'cross_direction_90',
        # VWAP cross rate
        'vwap_15_cross_rate_30', 'vwap_15_cross_rate_60', 'vwap_15_cross_rate_90', 'vwap_15_cross_rate_180',
        'vwap_30_cross_rate_30', 'vwap_30_cross_rate_60', 'vwap_30_cross_rate_90', 'vwap_30_cross_rate_180',
        'vwap_60_cross_rate_30', 'vwap_60_cross_rate_60', 'vwap_60_cross_rate_90', 'vwap_60_cross_rate_180',
        # Sessions
        'session_id_vwap15', 'session_type_vwap15',
        'session_id_vwap30', 'session_type_vwap30'
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
    print(f"\nVWAP 15 Sessions: {int(df['session_id_vwap15'].max())} total")
    print(df['session_type_vwap15'].value_counts())
    print(f"\nVWAP 30 Sessions: {int(df['session_id_vwap30'].max())} total")
    print(df['session_type_vwap30'].value_counts())

    # New feature stats
    print(f"\n=== Precomputed Feature Stats ===")
    print(f"Parkinson ratio - mean: {df['parkinson_ratio'].mean():.3f}, "
          f"median: {df['parkinson_ratio'].median():.3f}")
    print(f"Parkinson ratio >= 1.5: {(df['parkinson_ratio'] >= 1.5).mean() * 100:.1f}%")
    print(f"Parkinson ratio pct 7d - mean: {df['parkinson_ratio_pct_7d'].mean():.3f}, "
          f"p90: {df['parkinson_ratio_pct_7d'].quantile(0.9):.3f}")
    print(f"Volume ratio 20 - mean: {df['volume_ratio_20'].mean():.3f}")
    print(f"Cross events (VWAP 15 OHLC/4): {df['cross_direction_15'].notna().sum():,}")
    print(f"Cross events (VWAP 15 HLC/3): {df['cross_direction_15_hlc'].notna().sum():,}")
    print(f"Taker imb EMA 15 - mean: {df['taker_imb_ema_15'].mean():.4f}, "
          f"std: {df['taker_imb_ema_15'].std():.4f}")
    print(f"Taker imb EMA 30 - mean: {df['taker_imb_ema_30'].mean():.4f}, "
          f"std: {df['taker_imb_ema_30'].std():.4f}")
    print(f"Avg trade size - mean: {df['avg_trade_size'].mean():.2f}, "
          f"median: {df['avg_trade_size'].median():.2f}")
    print(f"Avg trade size zscore 60 - std: {df['avg_trade_size_zscore_60'].std():.2f}")
    print(f"Avg trade size zscore 240 - std: {df['avg_trade_size_zscore_240'].std():.2f}")
    print(f"Avg trade size zscore 1440 - std: {df['avg_trade_size_zscore_1440'].std():.2f}")
    print(f"Intrabar return zscore 1440 - std: {df['intrabar_return_zscore_1440'].std():.2f}, "
          f">1: {(df['intrabar_return_zscore_1440'] > 1).mean() * 100:.1f}%, "
          f"<-1: {(df['intrabar_return_zscore_1440'] < -1).mean() * 100:.1f}%")

    return df


if __name__ == "__main__":
    process_data()
