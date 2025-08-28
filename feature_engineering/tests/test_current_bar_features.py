"""
Smoke tests for feature_engineering/current_bar_features.py

Validates shapes, column names (with timeframe suffix), and math for a small
OHLCV sample using only t and t−1 information.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path so 'feature_engineering' package is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feature_engineering.current_bar_features import compute_current_bar_features, with_current_bar_features


def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    if (a is None) or (b is None):
        return False
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= tol


def run_tests():
    print("\n== Testing current_bar_features (vectorized) ==")

    idx = pd.to_datetime([
        "2025-01-01 00:00:00",
        "2025-01-01 01:00:00",
        "2025-01-01 02:00:00",
    ])
    df = pd.DataFrame(
        {
            "open":   [100.0, 101.0, 102.0],
            "high":   [102.0, 104.0, 105.0],
            "low":    [ 99.0, 100.0, 101.0],
            "close":  [101.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0,  900.0],
        },
        index=idx,
    )

    tf = "1H"
    feats = compute_current_bar_features(df, timeframe_suffix=tf, include_original=False)

    expected_cols = {
        f"close_logret_current_{tf}",
        f"high_low_range_pct_current_{tf}",
        f"close_open_pct_current_{tf}",
        f"log_volume_{tf}",
        f"log_volume_delta_current_{tf}",
        f"sign_close_logret_current_{tf}",
        f"close_logret_current_x_log_volume_{tf}",
        f"close_logret_current_x_log_volume_delta_current_{tf}",
        f"sign_close_logret_current_x_log_volume_{tf}",
        f"high_low_range_pct_current_x_log_volume_{tf}",
        f"close_open_pct_current_x_log_volume_{tf}",
    }
    print("Columns:", list(feats.columns))
    print("All expected columns present:", expected_cols.issubset(set(feats.columns)))

    # Compute expectations for last row (t = 2025-01-01 02:00)
    C2, C1 = 104.0, 103.0
    O2, H2, L2 = 102.0, 105.0, 101.0
    V2, V1 = 900.0, 1100.0

    close_logret_exp = np.log(C2 / C1)
    hl_range_pct_exp = (H2 - L2) / O2
    co_pct_exp = (C2 - O2) / O2
    log_vol_exp = np.log1p(V2)
    log_vol_delta_exp = np.log1p(V2) - np.log1p(V1)
    sign_close_logret_exp = np.sign(close_logret_exp)

    row = feats.iloc[-1]
    assert _approx_equal(row[f"close_logret_current_{tf}"], close_logret_exp)
    assert _approx_equal(row[f"high_low_range_pct_current_{tf}"], hl_range_pct_exp)
    assert _approx_equal(row[f"close_open_pct_current_{tf}"], co_pct_exp)
    assert _approx_equal(row[f"log_volume_{tf}"], log_vol_exp)
    assert _approx_equal(row[f"log_volume_delta_current_{tf}"], log_vol_delta_exp)
    assert _approx_equal(row[f"sign_close_logret_current_{tf}"], sign_close_logret_exp)
    assert _approx_equal(row[f"close_logret_current_x_log_volume_{tf}"], close_logret_exp * log_vol_exp)
    assert _approx_equal(row[f"close_logret_current_x_log_volume_delta_current_{tf}"], close_logret_exp * log_vol_delta_exp)
    assert _approx_equal(row[f"sign_close_logret_current_x_log_volume_{tf}"], sign_close_logret_exp * log_vol_exp)
    assert _approx_equal(row[f"high_low_range_pct_current_x_log_volume_{tf}"], hl_range_pct_exp * log_vol_exp)
    assert _approx_equal(row[f"close_open_pct_current_x_log_volume_{tf}"], co_pct_exp * log_vol_exp)

    # First row checks (t0): features requiring t−1 should be NaN
    row0 = feats.iloc[0]
    assert np.isnan(row0[f"close_logret_current_{tf}"])
    assert np.isnan(row0[f"log_volume_delta_current_{tf}"])
    # Range/close-open percentages are current-only and should be finite
    assert np.isfinite(row0[f"high_low_range_pct_current_{tf}"])
    assert np.isfinite(row0[f"close_open_pct_current_{tf}"])

    # Wrapper: original + features
    df_plus = with_current_bar_features(df, timeframe_suffix=tf)
    for base_col in ["open", "high", "low", "close", "volume"]:
        assert base_col in df_plus.columns
    for col in expected_cols:
        assert col in df_plus.columns

    print("All assertions passed.")


if __name__ == "__main__":
    run_tests()


