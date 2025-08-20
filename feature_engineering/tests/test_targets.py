"""
Print-based tests for feature_engineering/targets.py (consistent with project style).
"""

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

# Add parent directory to path to import targets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from targets import (
    compute_forward_return,
    compute_mfe_mae,
    compute_barrier_outcomes,
    TargetGenerationConfig,
    generate_targets_for_row,
    extract_forward_window,
)


def make_forward_ohlcv(periods: int) -> pd.DataFrame:
    # Simple ascending series around base=100 to create predictable highs/lows
    base = 100.0
    idx = pd.RangeIndex(start=1, stop=periods + 1)
    # Create bars with 0.5 range around a linear trend
    close = base + np.arange(1, periods + 1, dtype=float)
    open_s = close - 0.2
    high = np.maximum(open_s, close) + 0.5
    low = np.minimum(open_s, close) - 0.5
    df = pd.DataFrame({
        'open': open_s,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.arange(1, periods + 1, dtype=float) * 10.0,
    }, index=idx)
    return df


def test_compute_forward_return():
    print("\n" + "=" * 50)
    print("Testing compute_forward_return...")

    entry_price = 100.0
    forward = make_forward_ohlcv(3)  # closes: 101, 102, 103

    res_log = compute_forward_return(entry_price, forward['close'], 2, log=True)
    expected_log = float(np.log(102.0) - np.log(100.0))
    print("Log result:", res_log, "expected:", expected_log)
    print("Test passed:", abs(res_log['y_logret_2b'] - expected_log) < 1e-12)

    res_simple = compute_forward_return(entry_price, forward['close'], 3, log=False)
    expected_simple = 103.0 / 100.0 - 1.0
    print("Simple result:", res_simple, "expected:", expected_simple)
    print("Test passed:", abs(res_simple['y_ret_3b'] - expected_simple) < 1e-12)

    # Insufficient bars
    res_insuff = compute_forward_return(entry_price, forward['close'], 5, log=True)
    print("Insufficient bars -> NaN:", res_insuff)
    print("Test passed:", np.isnan(res_insuff['y_logret_5b']))


def test_compute_mfe_mae():
    print("\n" + "=" * 50)
    print("Testing compute_mfe_mae...")

    entry_price = 100.0
    # Construct highs/lows such that max_high=103, min_low=98 within window
    forward_high = pd.Series([101.0, 103.0, 102.0])
    forward_low = pd.Series([99.5, 100.0, 98.0])
    res = compute_mfe_mae(forward_high, forward_low, entry_price, 3)
    expected_mfe = 103.0 / 100.0 - 1.0
    expected_mae = 98.0 / 100.0 - 1.0
    print("Result:", res, "expected MFE:", expected_mfe, "expected MAE:", expected_mae)
    ok = abs(res['y_mfe_3b'] - expected_mfe) < 1e-12 and abs(res['y_mae_3b'] - expected_mae) < 1e-12
    print("Test passed:", ok)

    # Insufficient bars
    res_insuff = compute_mfe_mae(forward_high, forward_low, entry_price, 5)
    print("Insufficient bars -> NaNs:", res_insuff)
    print("Test passed:", np.isnan(res_insuff['y_mfe_5b']) and np.isnan(res_insuff['y_mae_5b']))


def test_compute_barrier_outcomes():
    print("\n" + "=" * 50)
    print("Testing compute_barrier_outcomes...")

    entry_price = 100.0
    up, down = 0.02, 0.01  # +2% / -1%

    # Case 1: Upper hit first (ternary=+1, binary=1)
    fh1 = pd.Series([100.5, 102.2, 101.5])
    fl1 = pd.Series([99.8, 100.4, 100.0])
    r1t = compute_barrier_outcomes(fh1, fl1, entry_price, up, down, 3, mode='ternary')
    r1b = compute_barrier_outcomes(fh1, fl1, entry_price, up, down, 3, mode='binary')
    print("Upper first ternary:", r1t, "binary:", r1b)
    print("Test passed:", r1t['y_tb_label_u0.02_d0.01_3b'] == 1.0 and r1b['y_tp_before_sl_u0.02_d0.01_3b'] == 1.0)

    # Case 2: Lower hit first (ternary=-1, binary=0)
    fh2 = pd.Series([100.5, 100.8, 100.9])
    fl2 = pd.Series([98.9, 99.2, 99.4])  # 98.9 <= 99.0 triggers SL on first bar
    r2 = compute_barrier_outcomes(fh2, fl2, entry_price, up, down, 3, mode='both')
    print("Lower first both:", r2)
    ok2 = r2['y_tb_label_u0.02_d0.01_3b'] == -1.0 and r2['y_tp_before_sl_u0.02_d0.01_3b'] == 0.0
    print("Test passed:", ok2)

    # Case 3: Tie in same bar -> conservative policy (0)
    fh3 = pd.Series([102.5])  # >= 102 (TP)
    fl3 = pd.Series([98.5])   # <= 99 (SL)
    r3 = compute_barrier_outcomes(fh3, fl3, entry_price, up, down, 1, mode='ternary', tie_policy='conservative')
    print("Tie conservative:", r3)
    print("Test passed:", r3['y_tb_label_u0.02_d0.01_1b'] == 0.0)

    # Case 4: Tie in same bar -> proximity_to_open
    open3 = pd.Series([100.1])
    # distances: upper |102 - 100.1| = 1.9; lower |100.1 - 99| = 1.1 -> choose lower => -1
    r4 = compute_barrier_outcomes(fh3, fl3, entry_price, up, down, 1, mode='ternary', tie_policy='proximity_to_open', forward_open=open3)
    print("Tie proximity_to_open:", r4)
    print("Test passed:", r4['y_tb_label_u0.02_d0.01_1b'] == -1.0)

    # Insufficient bars -> NaN
    r5 = compute_barrier_outcomes(fh1, fl1, entry_price, up, down, 5, mode='both')
    print("Insufficient bars (both):", r5)
    print("Test passed:", np.isnan(r5['y_tb_label_u0.02_d0.01_5b']) and np.isnan(r5['y_tp_before_sl_u0.02_d0.01_5b']))


def test_generate_targets_for_row_and_extract():
    print("\n" + "=" * 50)
    print("Testing generate_targets_for_row and extract_forward_window...")

    # Create a dummy OHLCV dataset of 10 rows
    data = pd.DataFrame({
        'open': np.linspace(100, 109, 10),
        'high': np.linspace(100.5, 109.5, 10),
        'low': np.linspace(99.5, 108.5, 10),
        'close': np.linspace(100.2, 109.2, 10),
        'volume': np.linspace(1000, 1900, 10),
    })

    current_idx = 5
    fw = extract_forward_window(data, current_idx, 3)  # rows 6..8
    print("Forward window shape:", fw.shape)
    print("Test passed:", len(fw) == 3)

    entry_price = float(data['close'].iloc[current_idx])
    cfg = TargetGenerationConfig(
        horizons_bars=[1, 3],
        barrier_pairs=[(0.02, 0.01)],
        include_returns=True,
        include_mfe_mae=True,
        include_barriers=True,
        log_returns=True,
    )
    res = generate_targets_for_row(fw, entry_price, cfg)
    print("Generated keys:", sorted(res.keys()))
    # Check presence of expected keys (labels use '1b' and '3b' horizon suffixes)
    expected_keys = [
        'y_logret_1b', 'y_logret_3b',
        'y_mfe_1b', 'y_mae_1b', 'y_mfe_3b', 'y_mae_3b',
        'y_tb_label_u0.02_d0.01_1b', 'y_tp_before_sl_u0.02_d0.01_1b',
        'y_tb_label_u0.02_d0.01_3b', 'y_tp_before_sl_u0.02_d0.01_3b',
    ]
    has_all = all(k in res for k in expected_keys)
    print("Test passed:", has_all)


if __name__ == '__main__':
    test_compute_forward_return()
    test_compute_mfe_mae()
    test_compute_barrier_outcomes()
    test_generate_targets_for_row_and_extract()


