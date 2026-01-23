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
    df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
    df['volume_ratio_30'] = df['volume'] / df['volume_sma_30']

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
    # PER-BAR FEATURES (no warmup needed)
    # =========================================================================
    print("Calculating per-bar features...")
    # Intrabar return: (close - open) / open
    df['intrabar_return'] = (df['close'] - df['open']) / df['open']

    # VWAP gap: close - vwap_15
    df['vwap_gap_15'] = df['close'] - df['vwap_15']

    # Legacy VWAP gap (HLC/3)
    df['vwap_gap_15_hlc'] = df['close'] - df['vwap_15_hlc']

    # VWAP 30 gap
    df['vwap_gap_30'] = df['close'] - df['vwap_30']

    # VWAP 60 and 90 gaps
    df['vwap_gap_60'] = df['close'] - df['vwap_60']
    df['vwap_90'] = rolling_vwap(
        df['open'], df['high'], df['low'], df['close'], df['volume'], window=90
    )
    df['vwap_gap_90'] = df['close'] - df['vwap_90']

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
        'vwap_15', 'vwap_30', 'vwap_60', 'vwap_90', 'vwap_240',
        # VWAP (HLC/3 legacy formula)
        'vwap_15_hlc',
        # Parkinson volatility
        'parkinson_30', 'parkinson_1440', 'parkinson_ratio', 'parkinson_ratio_pct_7d',
        # Volume features
        'volume_sma_20', 'volume_sma_30', 'volume_ratio_20', 'volume_ratio_30',
        # Taker imbalance EMA (alt)
        'taker_imb_ema_15', 'taker_imb_ema_30',
        # Average trade size
        'avg_trade_size', 'avg_trade_size_zscore_60', 'avg_trade_size_zscore_240', 'avg_trade_size_zscore_1440',
        # Per-bar features
        'intrabar_return', 'vwap_gap_15', 'vwap_gap_15_hlc', 'vwap_gap_30', 'vwap_gap_60', 'vwap_gap_90',
        # Cross detection
        'cross_direction_15', 'cross_direction_15_hlc', 'cross_direction_30', 'cross_direction_60', 'cross_direction_90',
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

    return df


if __name__ == "__main__":
    process_data()
