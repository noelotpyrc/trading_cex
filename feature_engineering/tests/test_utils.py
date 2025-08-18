"""
Simple test script for feature_engineering/utils.py
Runs basic checks with prints instead of a full test framework.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    validate_ohlcv_data,
    safe_divide,
    get_lookback_window,
    resample_ohlcv_right_closed,
    generate_timeframe_lookbacks,
)


def make_ohlcv(start: str, periods: int, freq: str = '1H') -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    base = np.arange(periods, dtype=float)
    df = pd.DataFrame({
        'open': base + 100,
        'high': base + 100.5,
        'low': base + 99.5,
        'close': base + 100.2,
        'volume': (base + 1) * 10,
    }, index=idx)
    return df


def test_validate_ohlcv_data():
    print("\n" + "="*50)
    print("Testing validate_ohlcv_data...")
    df = make_ohlcv('2024-01-01 00:00', 5)
    print("Valid DF:", validate_ohlcv_data(df))
    bad = df.drop(columns=['volume'])
    print("Missing column:", validate_ohlcv_data(bad))
    empty = df.iloc[0:0]
    print("Empty:", validate_ohlcv_data(empty))
    print("None:", validate_ohlcv_data(None))


def test_safe_divide():
    print("\n" + "="*50)
    print("Testing safe_divide...")
    print("6/3:", safe_divide(6.0, 3.0))
    print("denom 0 default NaN -> is NaN:", np.isnan(safe_divide(1.0, 0.0)))
    print("denom 0 default -1:", safe_divide(1.0, 0.0, default=-1.0))
    print("nan numerator:", np.isnan(safe_divide(np.nan, 2.0)))
    print("nan denominator:", np.isnan(safe_divide(2.0, np.nan)))


def test_get_lookback_window():
    print("\n" + "="*50)
    print("Testing get_lookback_window...")
    df = make_ohlcv('2024-01-01 00:00', 10)
    current_idx = 7
    lb = get_lookback_window(df, current_idx, 5)
    print(lb)
    print("Slice 5 ending at 7 -> rows:", lb.index[0], "to", lb.index[-1])
    print("Len == 5:", len(lb) == 5)
    # near start
    lb2 = get_lookback_window(df, 2, 5)
    print("Slice near start -> len 3:", len(lb2) == 3)


def test_resample_ohlcv_right_closed():
    print("\n" + "="*50)
    print("Testing resample_ohlcv_right_closed...")
    # 8 hourly bars 00:00..07:00
    df = make_ohlcv('2024-01-01 00:00', 8)
    out4 = resample_ohlcv_right_closed(df, '4H')
    print(out4)
    print("Resample 4H labels:", list(out4.index))
    # Expect only 04:00 (08:00 would be > last ts 07:00)
    print("Last label <= last ts:", out4.index[-1] <= df.index[-1])
    # extend to 09 points 00..08 so 08:00 aligns
    df2 = make_ohlcv('2024-01-01 00:00', 9)
    out4_2 = resample_ohlcv_right_closed(df2, '4H')
    print("Resample 4H with aligned end labels:", list(out4_2.index))
    print("Ends at 08:00:", str(out4_2.index[-1]) == '2024-01-01 08:00:00')


def test_generate_timeframe_lookbacks():
    print("\n" + "="*50)
    print("Testing generate_timeframe_lookbacks...")
    df = make_ohlcv('2024-01-01 00:00', 24)
    current_idx = 9
    res = generate_timeframe_lookbacks(df, current_idx, 10, ("1H", "4H", "12H", "1D"))
    print("Keys:", list(res.keys()))
    print("1H equals raw lb:", res['1H'].equals(get_lookback_window(df, current_idx, 10)))
    print("4H last label <= current ts:", list(res['4H'].index)[-1] <= df.index[current_idx])


if __name__ == '__main__':
    test_validate_ohlcv_data()
    test_safe_divide()
    test_get_lookback_window()
    test_resample_ohlcv_right_closed()
    test_generate_timeframe_lookbacks()


