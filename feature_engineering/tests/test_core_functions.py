"""
Simple test script for core feature calculation functions.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import core_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_functions import get_lags, calculate_price_differences, calculate_log_transforms, calculate_percentage_changes, calculate_cumulative_returns, calculate_zscore, calculate_sma, calculate_ema, calculate_wma, calculate_ma_crossovers, calculate_ma_distance, calculate_macd, calculate_volume_ma


def test_get_lags():
    """Simple test for get_lags function"""
    print("Testing get_lags...")
    
    # Test data: [10, 20, 30, 40, 50]
    # Current value (last): 50
    # lag_1 should be 40 (previous)
    # lag_2 should be 30 (two back)
    test_data = pd.Series([10, 20, 30, 40, 50])
    
    # Test basic lags with default column name
    result = get_lags(test_data, [1, 2])
    expected = {'close_lag_1': 40, 'close_lag_2': 30}
    
    print(f"Input data: {test_data.values}")
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Test passed: {result == expected}")
    
    # Test with custom column name
    result2 = get_lags(test_data, [1, 2], "close")
    expected2 = {'close_lag_1': 40, 'close_lag_2': 30}
    print(f"\nCustom column name 'close': {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {result2 == expected2}")
    
    # Test with volume column name
    result3 = get_lags(test_data, [1, 2], "volume")
    expected3 = {'volume_lag_1': 40, 'volume_lag_2': 30}
    print(f"Custom column name 'volume': {result3}")
    print(f"Expected: {expected3}")
    print(f"Test passed: {result3 == expected3}")
    
    # Test edge case: insufficient data
    result4 = get_lags(test_data, [6], "close")
    expected4 = {'close_lag_6': np.nan}
    print(f"\nEdge case - lag 6: {result4}")
    print(f"Expected: {expected4}")
    print(f"Test passed: {result4 == expected4}")


def test_price_differences():
    """Simple test for calculate_price_differences function"""
    print("\n" + "="*50)
    print("Testing calculate_price_differences...")
    
    # Test data: OHLCV DataFrame with more realistic price movements
    test_data = pd.DataFrame({
        'open': [100, 102, 98, 105, 103],
        'high': [103, 104, 100, 107, 105],
        'low': [99, 101, 97, 103, 101],
        'close': [102, 98, 105, 103, 101],
        'volume': [1000, 1200, 800, 1500, 1100]
    })
    
    print(f"Test data:\n{test_data}")
    
    # Test default behavior (lag 0 = current row)
    result1 = calculate_price_differences(test_data)
    expected1 = {
        'close_open_diff_current': -2,  # 101 - 103 = -2
        'high_low_range_current': 4,   # 105 - 101 = 4
        'close_change_current': -2      # 101 - 103 = -2 (current to previous)
    }
    print(f"\nDefault result (lag 0 = current): {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {result1 == expected1}")
    
    # Test lag 1 (previous row)
    result2 = calculate_price_differences(test_data, 1)
    expected2 = {
        'close_open_diff_lag_1': -2,   # 103 - 105 = -2
        'high_low_range_lag_1': 4,     # 107 - 103 = 4
        'close_change_lag_1': -2       # 103 - 105 = -2 (row 3 to row 2)
    }
    print(f"Lag 1 (previous row): {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {result2 == expected2}")
    
    # Test lag 2 (two rows back)
    result3 = calculate_price_differences(test_data, 2)
    expected3 = {
        'close_open_diff_lag_2': 7,    # 105 - 98 = 7
        'high_low_range_lag_2': 3,     # 100 - 97 = 3
        'close_change_lag_2': 7        # 105 - 98 = 7 (row 2 to row 1)
    }
    print(f"Lag 2 (two rows back): {result3}")
    print(f"Expected: {expected3}")
    print(f"Test passed: {result3 == expected3}")
    
    # Test edge case: lag beyond data length
    result4 = calculate_price_differences(test_data, 5)
    expected4 = {
        'close_open_diff': np.nan,
        'high_low_range': np.nan,
        'price_change': np.nan
    }
    print(f"Edge case - lag 5 (beyond data): {result4}")
    print(f"Expected: {expected4}")
    print(f"Test passed: {result4 == expected4}")


def test_log_transforms():
    """Simple test for calculate_log_transforms function"""
    print("\n" + "="*50)
    print("Testing calculate_log_transforms...")
    
    # Test data: OHLCV DataFrame
    test_data = pd.DataFrame({
        'open': [100, 102, 98, 105, 103],
        'high': [103, 104, 100, 107, 105],
        'low': [99, 101, 97, 103, 101],
        'close': [102, 98, 105, 103, 101],
        'volume': [1000, 1200, 800, 1500, 1100]
    })
    
    print(f"Test data:\n{test_data}")
    
    # Test default behavior (all columns)
    result1 = calculate_log_transforms(test_data)
    print(f"\nDefault result (all columns): {result1}")
    
    # Test with specific columns
    result2 = calculate_log_transforms(test_data, ['close', 'volume'])
    print(f"Specific columns ['close', 'volume']: {result2}")
    
    # Test with single column
    result3 = calculate_log_transforms(test_data, ['high'])
    print(f"Single column ['high']: {result3}")
    
    # Test with non-existent column
    result4 = calculate_log_transforms(test_data, ['close', 'nonexistent'])
    print(f"Non-existent column: {result4}")


def test_percentage_changes():
    """Simple test for calculate_percentage_changes function"""
    print("\n" + "="*50)
    print("Testing calculate_percentage_changes...")
    
    # Test data: Price series with realistic movements
    test_data = pd.Series([100, 102, 98, 105, 103])
    
    print(f"Test data: {test_data.values}")
    
    # Test default behavior (current row)
    result1 = calculate_percentage_changes(test_data)
    expected1 = {'close_pct_change_current': -1.9047619047619049}  # (103 - 105) / 105 * 100
    print(f"\nDefault result (current row): {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['close_pct_change_current'] - expected1['close_pct_change_current']) < 0.001}")
    
    # Test lag 1 (previous row)
    result2 = calculate_percentage_changes(test_data, 1)
    expected2 = {'close_pct_change_lag_1': 7.142857142857142}  # (105 - 98) / 98 * 100
    print(f"Lag 1 (previous row): {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2['close_pct_change_lag_1'] - expected2['close_pct_change_lag_1']) < 0.001}")
    
    # Test lag 2 (two rows back)
    result3 = calculate_percentage_changes(test_data, 2)
    expected3 = {'close_pct_change_lag_2': -3.9215686274509802}  # (98 - 102) / 102 * 100
    print(f"Lag 2 (two rows back): {result3}")
    print(f"Expected: {expected3}")
    print(f"Test passed: {abs(result3['close_pct_change_lag_2'] - expected3['close_pct_change_lag_2']) < 0.001}")
    
    # Test with custom column name
    result4 = calculate_percentage_changes(test_data, 0, "volume")
    expected4 = {'volume_pct_change_current': -1.9047619047619049}  # (103 - 105) / 105 * 100
    print(f"Custom column name 'volume': {result4}")
    print(f"Expected: {expected4}")
    print(f"Test passed: {abs(result4['volume_pct_change_current'] - expected4['volume_pct_change_current']) < 0.001}")
    
    # Test edge case: lag beyond data length
    result5 = calculate_percentage_changes(test_data, 5)
    expected5 = {'close_pct_change': np.nan}
    print(f"Edge case - lag 5 (beyond data): {result5}")
    print(f"Expected: {expected5}")
    print(f"Test passed: {result5 == expected5}")


def test_cumulative_returns():
    """Simple test for calculate_cumulative_returns function"""
    print("\n" + "="*50)
    print("Testing calculate_cumulative_returns...")
    
    # Test data: Price series with realistic movements
    test_data = pd.Series([100, 102, 98, 105, 103])
    
    print(f"Test data: {test_data.values}")
    
    # Test with different window sizes
    result1 = calculate_cumulative_returns(test_data, [2, 3, 4])
    
    # Calculate expected values using the math formula
    log_returns = np.log(test_data / test_data.shift(1)).dropna()
    expected1 = {
        'cum_return_2': log_returns.tail(2).sum(),
        'cum_return_3': log_returns.tail(3).sum(),
        'cum_return_4': log_returns.tail(4).sum()
    }
    print(f"\nWindows [2, 3, 4]: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {all(abs(result1[f'cum_return_{w}'] - expected1[f'cum_return_{w}']) < 0.001 for w in [2, 3, 4])}")
    
    # Test with single window
    result2 = calculate_cumulative_returns(test_data, [2])
    expected2 = {'cum_return_2': log_returns.tail(2).sum()}
    print(f"Single window [2]: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2['cum_return_2'] - expected2['cum_return_2']) < 0.001}")
    
    # Test edge case: insufficient data
    short_data = pd.Series([100, 101])
    result3 = calculate_cumulative_returns(short_data, [3])
    expected3 = {'cum_return_3': np.nan}
    print(f"Edge case - insufficient data: {result3}")
    print(f"Expected: {expected3}")
    print(f"Test passed: {result3 == expected3}")
    
    # Test edge case: single data point
    single_data = pd.Series([100])
    result4 = calculate_cumulative_returns(single_data, [2])
    expected4 = {'cum_return_2': np.nan}
    print(f"Edge case - single data point: {result4}")
    print(f"Expected: {expected4}")
    print(f"Test passed: {result4 == expected4}")


def test_zscore():
    """Simple test for calculate_zscore function"""
    print("\n" + "="*50)
    print("Testing calculate_zscore...")
    
    # Test data: Price series with realistic movements
    test_data = pd.Series([100, 102, 98, 105, 103, 107, 101, 109])
    
    print(f"Test data: {test_data.values}")
    
    # Test with window size 3
    result1 = calculate_zscore(test_data, 3)
    
    # Calculate expected value using the math formula
    # Z-score = (current_price - rolling_mean) / rolling_std
    # Current price (last): 109
    # Rolling mean of last 3: (107 + 101 + 109) / 3 = 105.67
    # Rolling std of last 3: sqrt(sum((x - mean)^2) / (n-1))
    rolling_data = test_data.tail(3)  # [107, 101, 109]
    mean = rolling_data.mean()  # 105.67
    std = rolling_data.std()    # 4.16
    expected_zscore = (109 - mean) / std  # (109 - 105.67) / 4.16 ≈ 0.80
    
    print(f"\nWindow size 3: {result1}")
    print(f"Expected: {expected_zscore}")
    print(f"Test passed: {abs(result1 - expected_zscore) < 0.01}")
    
    # Test with window size 5
    result2 = calculate_zscore(test_data, 5)
    
    # Calculate expected value
    rolling_data2 = test_data.tail(5)  # [105, 103, 107, 101, 109]
    mean2 = rolling_data2.mean()  # 105.0
    std2 = rolling_data2.std()    # 3.16
    expected_zscore2 = (109 - mean2) / std2  # (109 - 105.0) / 3.16 ≈ 1.26
    
    print(f"Window size 5: {result2}")
    print(f"Expected: {expected_zscore2}")
    print(f"Test passed: {abs(result2 - expected_zscore2) < 0.01}")
    
    # Test edge case: insufficient data
    short_data = pd.Series([100, 101])
    result3 = calculate_zscore(short_data, 3)
    expected3 = np.nan
    print(f"Edge case - insufficient data: {result3}")
    print(f"Expected: {expected3}")
    print(f"Test passed: {pd.isna(result3) and pd.isna(expected3)}")
    
    # Test edge case: zero standard deviation (all values same)
    same_data = pd.Series([100, 100, 100, 100])
    result4 = calculate_zscore(same_data, 3)
    expected4 = np.nan
    print(f"Edge case - zero std dev: {result4}")
    print(f"Expected: {expected4}")
    print(f"Test passed: {pd.isna(result3) and pd.isna(expected3)}")


def test_sma():
    """Simple test for calculate_sma function"""
    print("\n" + "="*50)
    print("Testing calculate_sma...")

    # Test data
    test_data = pd.Series([100, 102, 98, 105, 103])
    print(f"Test data: {test_data.values}")

    # Window 3
    result1 = calculate_sma(test_data, 3)
    expected1 = test_data.rolling(window=3).mean().iloc[-1]
    print(f"Window 3 SMA: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1 - expected1) < 1e-9}")

    # Window 5
    result2 = calculate_sma(test_data, 5)
    expected2 = test_data.rolling(window=5).mean().iloc[-1]
    print(f"Window 5 SMA: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2 - expected2) < 1e-9}")

    # Insufficient window
    result3 = calculate_sma(test_data, 6)
    print(f"Insufficient window (6): {result3}")
    print(f"Expected: {np.nan}")
    print(f"Test passed: {pd.isna(result3)}")


def test_ema():
    """Simple test for calculate_ema function"""
    print("\n" + "="*50)
    print("Testing calculate_ema...")

    test_data = pd.Series([100, 102, 98, 105, 103])
    print(f"Test data: {test_data.values}")

    # span=3
    result1 = calculate_ema(test_data, 3)
    expected1 = test_data.ewm(span=3).mean().iloc[-1]
    print(f"EMA span 3: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1 - expected1) < 1e-9}")

    # span=5
    result2 = calculate_ema(test_data, 5)
    expected2 = test_data.ewm(span=5).mean().iloc[-1]
    print(f"EMA span 5: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2 - expected2) < 1e-9}")

    # insufficient
    result3 = calculate_ema(pd.Series([100, 101]), 5)
    print(f"Insufficient data: {result3}")
    print(f"Expected: {np.nan}")
    print(f"Test passed: {pd.isna(result3)}")


def test_wma():
    """Simple test for calculate_wma function"""
    print("\n" + "="*50)
    print("Testing calculate_wma...")

    test_data = pd.Series([100, 102, 98, 105, 103])
    print(f"Test data: {test_data.values}")

    # Window 3
    window = 3
    result1 = calculate_wma(test_data, window)
    values1 = test_data.tail(window).values  # oldest->newest
    weights1 = np.arange(1, window + 1)
    expected1 = np.sum(weights1 * values1) / np.sum(weights1)
    print(f"WMA window 3: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1 - expected1) < 1e-9}")

    # Window 5
    window = 5
    result2 = calculate_wma(test_data, window)
    values2 = test_data.tail(window).values
    weights2 = np.arange(1, window + 1)
    expected2 = np.sum(weights2 * values2) / np.sum(weights2)
    print(f"WMA window 5: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2 - expected2) < 1e-9}")

    # Insufficient window
    result3 = calculate_wma(test_data, 6)
    print(f"Insufficient window (6): {result3}")
    print(f"Expected: {np.nan}")
    print(f"Test passed: {pd.isna(result3)}")


def test_ma_crossovers():
    """Simple test for calculate_ma_crossovers function"""
    print("\n" + "="*50)
    print("Testing calculate_ma_crossovers...")

    test_data = pd.Series([100, 102, 98, 105, 103])
    print(f"Test data: {test_data.values}")

    fast_window, slow_window = 2, 3
    result1 = calculate_ma_crossovers(test_data, fast_window, slow_window)

    ma_fast = test_data.rolling(window=fast_window).mean().iloc[-1]
    ma_slow = test_data.rolling(window=slow_window).mean().iloc[-1]
    expected1 = {
        'ma_cross_diff': ma_fast - ma_slow,
        'ma_cross_ratio': ma_fast / ma_slow if ma_slow != 0 else np.nan,
        'ma_cross_signal': 1 if ma_fast > ma_slow else 0
    }

    print(f"fast={fast_window}, slow={slow_window}: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['ma_cross_diff'] - expected1['ma_cross_diff']) < 1e-9 and abs(result1['ma_cross_ratio'] - expected1['ma_cross_ratio']) < 1e-9 and result1['ma_cross_signal'] == expected1['ma_cross_signal']}")

    # Insufficient data case
    short_data = pd.Series([100, 101])
    result2 = calculate_ma_crossovers(short_data, 3, 4)
    expected2 = {
        'ma_cross_diff': np.nan,
        'ma_cross_ratio': np.nan,
        'ma_cross_signal': np.nan
    }
    print(f"Insufficient data: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {all(pd.isna(result2[k]) for k in expected2.keys())}")


def test_ma_distance():
    """Simple test for calculate_ma_distance function"""
    print("\n" + "="*50)
    print("Testing calculate_ma_distance...")

    price = 105.0
    ma_val = 100.0
    result1 = calculate_ma_distance(price, ma_val)
    expected1 = {
        'ma_distance': price - ma_val,
        'ma_distance_pct': ((price - ma_val) / ma_val) * 100
    }
    print(f"price={price}, ma={ma_val}: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['ma_distance'] - expected1['ma_distance']) < 1e-9 and abs(result1['ma_distance_pct'] - expected1['ma_distance_pct']) < 1e-9}")

    # ma_value zero -> pct NaN
    price2 = 100.0
    ma_val2 = 0.0
    result2 = calculate_ma_distance(price2, ma_val2)
    print(f"ma=0 case: {result2}")
    print(f"Expected: {{'ma_distance': nan, 'ma_distance_pct': nan}}")
    print(f"Test passed: {pd.isna(result2['ma_distance']) and pd.isna(result2['ma_distance_pct'])}")

    # ma_value NaN
    price3 = 100.0
    ma_val3 = np.nan
    result3 = calculate_ma_distance(price3, ma_val3)
    print(f"ma=NaN case: {result3}")
    print(f"Expected: {{'ma_distance': nan, 'ma_distance_pct': nan}}")
    print(f"Test passed: {pd.isna(result3['ma_distance']) and pd.isna(result3['ma_distance_pct'])}")


def test_macd():
    """Simple test for calculate_macd function"""
    print("\n" + "="*50)
    print("Testing calculate_macd...")

    # Use small spans so short series works
    test_data = pd.Series([100, 102, 98, 105, 103])
    fast, slow, signal = 3, 5, 2
    result1 = calculate_macd(test_data, fast, slow, signal)

    ema_fast = test_data.ewm(span=fast).mean()
    ema_slow = test_data.ewm(span=slow).mean()
    macd_series = ema_fast - ema_slow
    macd_line_exp = macd_series.iloc[-1]
    macd_signal_exp = macd_series.ewm(span=signal).mean().iloc[-1]
    macd_hist_exp = macd_line_exp - macd_signal_exp

    print(f"fast={fast}, slow={slow}, signal={signal}: {result1}")
    print(f"Expected line/signal/hist: {macd_line_exp}, {macd_signal_exp}, {macd_hist_exp}")
    print(f"Test passed: {abs(result1['macd_line'] - macd_line_exp) < 1e-9 and abs(result1['macd_signal'] - macd_signal_exp) < 1e-9 and abs(result1['macd_histogram'] - macd_hist_exp) < 1e-9}")

    # Insufficient data for given spans
    short = pd.Series([100, 101])
    res2 = calculate_macd(short, 12, 26, 9)
    print(f"Insufficient data defaults: {res2}")
    print(f"Expected: {{'macd_line': np.nan, 'macd_signal': np.nan, 'macd_histogram': np.nan}}")
    print(f"Test passed: {pd.isna(res2['macd_line']) and pd.isna(res2['macd_signal']) and pd.isna(res2['macd_histogram'])}")


def test_volume_ma():
    """Simple test for calculate_volume_ma function (wraps SMA)"""
    print("\n" + "="*50)
    print("Testing calculate_volume_ma...")

    vol = pd.Series([1000, 1200, 800, 1500, 1100])
    window = 3
    res = calculate_volume_ma(vol, window)
    exp = vol.rolling(window=window).mean().iloc[-1]
    print(f"Volume MA window {window}: {res}")
    print(f"Expected: {exp}")
    print(f"Test passed: {abs(res - exp) < 1e-9}")


if __name__ == '__main__':
    test_get_lags()
    test_price_differences()
    test_log_transforms()
    test_percentage_changes()
    test_cumulative_returns()
    test_zscore()
    test_sma()
    test_ema()
    test_wma()
    test_ma_crossovers()
    test_ma_distance()
    test_macd()
    test_volume_ma()
