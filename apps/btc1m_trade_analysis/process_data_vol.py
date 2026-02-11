"""
Volatility Feature Processing

Processes 1-minute BTCUSDT data and computes volatility features and targets:

Features:
- Parkinson volatility (10/15/30/45/60/1440 bars)
- Parkinson ratio (p30/p1440, p15/p1440)

Targets (forward-looking, N=1..15):
- target_rv_nextN_close: realized variance from close-to-close returns
- target_mar_nextN_close: mean absolute return

Same input as process_data.py, minimal output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from feature_engineering.primitives import parkinson_volatility, ewma_variance
# from feature_engineering.derived_features import (
#     garman_klass_var, rogers_satchell_var,
#     garman_klass_vol, rogers_satchell_vol,
# )
from feature_engineering.new_features import forward_sum

# =============================================================================
# CONFIG
# =============================================================================

INPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/binance_btcusdt_perp_1m/BTCUSDT-1m-merged.csv")
OUTPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/binance_btcusdt_perp_1m/BTCUSDT-1m-features-vol.csv")
FILTER_DATE = "2022-01-01"

VOL_WINDOWS = [10, 15, 30, 45, 60, 1440]
# EWMA_HALFLIFES = [15, 30, 60]
TARGET_HORIZONS = list(range(1, 16))  # N = 1..15


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
    # PARKINSON VOLATILITY
    # =========================================================================
    print("Calculating Parkinson volatility...")
    for w in VOL_WINDOWS:
        df[f'parkinson_{w}'] = parkinson_volatility(df['high'], df['low'], window=w)

    # =========================================================================
    # PARKINSON RATIOS
    # =========================================================================
    print("Calculating Parkinson ratios...")
    df['parkinson_ratio_30_1440'] = df['parkinson_30'] / df['parkinson_1440'].replace(0, np.nan)
    df['parkinson_ratio_15_1440'] = df['parkinson_15'] / df['parkinson_1440'].replace(0, np.nan)

    # =========================================================================
    # GARMAN-KLASS VOLATILITY (rolling window) — DISABLED
    # =========================================================================
    # print("Calculating Garman-Klass volatility (rolling)...")
    # for w in VOL_WINDOWS:
    #     df[f'gk_vol_{w}'] = garman_klass_vol(
    #         df['open'], df['high'], df['low'], df['close'], window=w
    #     )

    # =========================================================================
    # ROGERS-SATCHELL VOLATILITY (rolling window) — DISABLED
    # =========================================================================
    # print("Calculating Rogers-Satchell volatility (rolling)...")
    # for w in VOL_WINDOWS:
    #     df[f'rs_vol_{w}'] = rogers_satchell_vol(
    #         df['open'], df['high'], df['low'], df['close'], window=w
    #     )

    # =========================================================================
    # EWMA VOLATILITY — DISABLED
    # =========================================================================
    # print("Computing per-bar variances for EWMA...")
    # var_gk = garman_klass_var(df['open'], df['high'], df['low'], df['close'])
    # var_rs = rogers_satchell_var(df['open'], df['high'], df['low'], df['close'])
    #
    # print("Calculating EWMA volatility (GK + RS)...")
    # for hl in EWMA_HALFLIFES:
    #     df[f'gk_ewma_vol_{hl}'] = np.sqrt(ewma_variance(var_gk, halflife=hl))
    #     df[f'rs_ewma_vol_{hl}'] = np.sqrt(ewma_variance(var_rs, halflife=hl))

    # =========================================================================
    # FORWARD TARGETS (N=1..15)
    # Must be computed BEFORE date filter (forward-looking)
    # =========================================================================
    print(f"Calculating forward targets (N={TARGET_HORIZONS[0]}..{TARGET_HORIZONS[-1]})...")

    log_ret = np.log(df['close'] / df['close'].shift(1))
    log_ret_sq = log_ret.pow(2)
    log_ret_abs = log_ret.abs()

    for n in TARGET_HORIZONS:
        # (1) Realized variance (close-to-close)
        df[f'target_rv_next{n}_close'] = forward_sum(log_ret_sq, n=n, offset=1)
        # (2) Integrated variance (GK) — DISABLED
        # df[f'target_iv_next{n}_gk'] = forward_sum(var_gk, n=n, offset=1)
        # (3) Integrated variance (RS) — DISABLED
        # df[f'target_iv_next{n}_rs'] = forward_sum(var_rs, n=n, offset=1)
        # (4) Mean absolute return
        df[f'target_mar_next{n}_close'] = forward_sum(log_ret_abs, n=n, offset=1) / float(n)

    # =========================================================================
    # FILTER TO 2022+
    # =========================================================================
    print(f"Filtering to data after {FILTER_DATE}...")
    df = df[df['datetime_utc'] >= FILTER_DATE].reset_index(drop=True)
    print(f"Filtered to {len(df):,} rows")

    # =========================================================================
    # SELECT OUTPUT COLUMNS
    # =========================================================================
    output_cols = [
        # Core OHLCV
        'datetime_utc', 'open', 'high', 'low', 'close', 'volume',
        # Parkinson volatility
        *[f'parkinson_{w}' for w in VOL_WINDOWS],
        # Parkinson ratios
        'parkinson_ratio_30_1440', 'parkinson_ratio_15_1440',
        # Garman-Klass (rolling) — DISABLED
        # *[f'gk_vol_{w}' for w in VOL_WINDOWS],
        # Rogers-Satchell (rolling) — DISABLED
        # *[f'rs_vol_{w}' for w in VOL_WINDOWS],
        # Garman-Klass (EWMA) — DISABLED
        # *[f'gk_ewma_vol_{hl}' for hl in EWMA_HALFLIFES],
        # Rogers-Satchell (EWMA) — DISABLED
        # *[f'rs_ewma_vol_{hl}' for hl in EWMA_HALFLIFES],
        # Forward targets
        *[f'target_rv_next{n}_close' for n in TARGET_HORIZONS],
        # *[f'target_iv_next{n}_gk' for n in TARGET_HORIZONS],
        # *[f'target_iv_next{n}_rs' for n in TARGET_HORIZONS],
        *[f'target_mar_next{n}_close' for n in TARGET_HORIZONS],
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
    print(f"\nVolatility feature stats (median):")
    vals = [f"{df[f'parkinson_{w}'].median():.6f}" for w in VOL_WINDOWS]
    print(f"  {'parkinson':16s}  {' / '.join(f'{w}={v}' for w, v in zip(VOL_WINDOWS, vals))}")
    print(f"  parkinson_ratio   30/1440={df['parkinson_ratio_30_1440'].median():.4f} / 15/1440={df['parkinson_ratio_15_1440'].median():.4f}")
    print(f"\nTarget stats (median, N=1/5/10/15):")
    for tgt in ['target_rv_next{}_close', 'target_mar_next{}_close']:
        sample_ns = [1, 5, 10, 15]
        vals = [f"{df[tgt.format(n)].median():.8f}" for n in sample_ns]
        label = tgt.format('N')
        print(f"  {label:28s}  {' / '.join(f'{n}={v}' for n, v in zip(sample_ns, vals))}")

    return df


if __name__ == "__main__":
    process_data()
