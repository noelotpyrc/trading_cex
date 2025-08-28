"""
Tests for compute_current_and_lag_features_from_lookback.
"""

import os
import sys
import numpy as np
import pandas as pd
from pprint import pprint

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feature_engineering.current_bar_features import (
    compute_current_and_lag_features_from_lookback,
)


def _approx(a, b, tol=1e-9):
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= tol


def run_tests():
    print("\n== Testing compute_current_and_lag_features_from_lookback ==")
    # Build a small 4H-like lookback (3 rows)
    idx = pd.to_datetime([
        "2025-01-01 04:00:00",
        "2025-01-01 08:00:00",
        "2025-01-01 12:00:00",
    ])
    lb = pd.DataFrame(
        {
            "open":   [100.0, 102.0, 103.0],
            "high":   [104.0, 105.0, 106.0],
            "low":    [ 99.0, 101.0, 102.0],
            "close":  [103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0,  900.0],
        },
        index=idx,
    )

    tf = "4H"
    out = compute_current_and_lag_features_from_lookback(lb, timeframe_suffix=tf, max_lag=2)
    print("\nOutput (current + lags):")
    pprint(out)

    # Current at 12:00 (t)
    C2, C1 = 105.0, 104.0
    O2, H2, L2 = 103.0, 106.0, 102.0
    V2, V1 = 900.0, 1100.0
    close_logret_t = np.log(C2 / C1)
    hl_pct_t = (H2 - L2) / O2
    co_pct_t = (C2 - O2) / O2
    lv_t = np.log1p(V2)
    lvd_t = np.log1p(V2) - np.log1p(V1)

    def g(k):
        return out.get(k)

    # Current assertions
    assert _approx(g(f"close_logret_current_{tf}"), close_logret_t)
    assert _approx(g(f"high_low_range_pct_current_{tf}"), hl_pct_t)
    assert _approx(g(f"close_open_pct_current_{tf}"), co_pct_t)
    assert _approx(g(f"log_volume_{tf}"), lv_t)
    assert _approx(g(f"log_volume_delta_current_{tf}"), lvd_t)

    # Lag-1 should match values at 08:00
    C1p, C0 = 104.0, 103.0
    O1, H1, L1 = 102.0, 105.0, 101.0
    V1p, V0 = 1100.0, 1000.0
    close_logret_l1 = np.log(C1p / C0)
    hl_pct_l1 = (H1 - L1) / O1
    co_pct_l1 = (C1p - O1) / O1
    lv_l1 = np.log1p(V1p)
    lvd_l1 = np.log1p(V1p) - np.log1p(V0)

    assert _approx(g(f"close_logret_lag_1_{tf}"), close_logret_l1)
    assert _approx(g(f"high_low_range_pct_lag_1_{tf}"), hl_pct_l1)
    assert _approx(g(f"close_open_pct_lag_1_{tf}"), co_pct_l1)
    assert _approx(g(f"log_volume_lag_1_{tf}"), lv_l1)
    assert _approx(g(f"log_volume_delta_lag_1_{tf}"), lvd_l1)

    # Lag-2 should match values at 04:00
    C0p, Cm1 = 103.0, np.nan
    O0, H0, L0 = 100.0, 104.0, 99.0
    V0p = 1000.0
    # close_logret_lag_2 cannot be computed without prior close; expect NaN
    assert np.isnan(g(f"close_logret_lag_2_{tf}"))
    assert _approx(g(f"high_low_range_pct_lag_2_{tf}"), (H0 - L0) / O0)
    assert _approx(g(f"close_open_pct_lag_2_{tf}"), (C0p - O0) / O0)
    assert _approx(g(f"log_volume_lag_2_{tf}"), np.log1p(V0p))
    # delta at lag-2 also NaN (no V_{-1})
    assert np.isnan(g(f"log_volume_delta_lag_2_{tf}"))

    print("All assertions passed for current+lag features from lookback.")


if __name__ == "__main__":
    run_tests()


