"""
Simple test script for core feature calculation functions.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import core_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_functions import get_lags, calculate_price_differences, calculate_log_transform, calculate_percentage_changes, calculate_cumulative_returns, calculate_zscore, calculate_sma, calculate_ema, calculate_wma, calculate_ma_crossovers, calculate_ma_distance, calculate_macd, calculate_volume_ma, calculate_rsi, calculate_stochastic, calculate_cci, calculate_roc, calculate_williams_r, calculate_ultimate_oscillator, calculate_mfi, calculate_historical_volatility, calculate_atr, calculate_bollinger_bands, calculate_volatility_ratio, calculate_parkinson_volatility, calculate_garman_klass_volatility


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
    result1 = calculate_price_differences(test_data['open'], test_data['high'], test_data['low'], test_data['close'])
    expected1 = {
        'close_open_diff_current': -2,  # 101 - 103 = -2
        'high_low_range_current': 4,   # 105 - 101 = 4
        'close_change_current': -2      # 101 - 103 = -2 (current to previous)
    }
    print(f"\nDefault result (lag 0 = current): {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {result1 == expected1}")
    
    # Test lag 1 (previous row)
    result2 = calculate_price_differences(test_data['open'], test_data['high'], test_data['low'], test_data['close'], 1)
    expected2 = {
        'close_open_diff_lag_1': -2,   # 103 - 105 = -2
        'high_low_range_lag_1': 4,     # 107 - 103 = 4
        'close_change_lag_1': -2       # 103 - 105 = -2 (row 3 to row 2)
    }
    print(f"Lag 1 (previous row): {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {result2 == expected2}")
    
    # Test lag 2 (two rows back)
    result3 = calculate_price_differences(test_data['open'], test_data['high'], test_data['low'], test_data['close'], 2)
    expected3 = {
        'close_open_diff_lag_2': 7,    # 105 - 98 = 7
        'high_low_range_lag_2': 3,     # 100 - 97 = 3
        'close_change_lag_2': 7        # 105 - 98 = 7 (row 2 to row 1)
    }
    print(f"Lag 2 (two rows back): {result3}")
    print(f"Expected: {expected3}")
    print(f"Test passed: {result3 == expected3}")
    
    # Test edge case: lag beyond data length
    result4 = calculate_price_differences(test_data['open'], test_data['high'], test_data['low'], test_data['close'], 5)
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
    
    # Test default behavior (single column close)
    result1 = calculate_log_transform(test_data['close'], 'close')
    print(f"\nDefault result (all columns): {result1}")
    
    # Test with specific columns separately
    result2_close = calculate_log_transform(test_data['close'], 'close')
    result2_volume = calculate_log_transform(test_data['volume'], 'volume')
    print(f"Specific columns ['close', 'volume']: {result2_close}, {result2_volume}")
    
    # Test with single column
    result3 = calculate_log_transform(test_data['high'], 'high')
    print(f"Single column ['high']: {result3}")
    
    # Non-applicable scenario: negative or zero
    test_data2 = pd.Series([0, -1, 5])
    result4 = calculate_log_transform(test_data2, 'synthetic')
    print(f"Non-positive series value: {result4}")


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
    result1 = calculate_cumulative_returns(test_data, [2, 3, 4], 'close')
    
    # Calculate expected values using the math formula
    log_returns = np.log(test_data / test_data.shift(1)).dropna()
    expected1 = {
        'close_cum_return_2': log_returns.tail(2).sum(),
        'close_cum_return_3': log_returns.tail(3).sum(),
        'close_cum_return_4': log_returns.tail(4).sum()
    }
    print(f"\nWindows [2, 3, 4]: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {all(abs(result1[f'close_cum_return_{w}'] - expected1[f'close_cum_return_{w}']) < 0.001 for w in [2, 3, 4])}")
    
    # Test with single window
    result2 = calculate_cumulative_returns(test_data, [2], 'close')
    expected2 = {'close_cum_return_2': log_returns.tail(2).sum()}
    print(f"Single window [2]: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2['close_cum_return_2'] - expected2['close_cum_return_2']) < 0.001}")
    
    # Test edge case: insufficient data
    short_data = pd.Series([100, 101])
    result3 = calculate_cumulative_returns(short_data, [3], 'close')
    expected3 = {'close_cum_return_3': np.nan}
    print(f"Edge case - insufficient data: {result3}")
    print(f"Expected: {expected3}")
    print(f"Test passed: {result3 == expected3}")
    
    # Test edge case: single data point
    single_data = pd.Series([100])
    result4 = calculate_cumulative_returns(single_data, [2], 'close')
    expected4 = {'close_cum_return_2': np.nan}
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
    result1 = calculate_zscore(test_data, 3, 'close')
    
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
    print(f"Test passed: {abs(result1['close_zscore_3'] - expected_zscore) < 0.01}")
    
    # Test with window size 5
    result2 = calculate_zscore(test_data, 5, 'close')
    
    # Calculate expected value
    rolling_data2 = test_data.tail(5)  # [105, 103, 107, 101, 109]
    mean2 = rolling_data2.mean()  # 105.0
    std2 = rolling_data2.std()    # 3.16
    expected_zscore2 = (109 - mean2) / std2  # (109 - 105.0) / 3.16 ≈ 1.26
    
    print(f"Window size 5: {result2}")
    print(f"Expected: {expected_zscore2}")
    print(f"Test passed: {abs(result2['close_zscore_5'] - expected_zscore2) < 0.01}")
    
    # Test edge case: insufficient data
    short_data = pd.Series([100, 101])
    result3 = calculate_zscore(short_data, 3, 'close')
    expected3 = np.nan
    print(f"Edge case - insufficient data: {result3}")
    print(f"Expected: {expected3}")
    print(f"Test passed: {pd.isna(result3['close_zscore_3']) and pd.isna(expected3)}")
    
    # Test edge case: zero standard deviation (all values same)
    same_data = pd.Series([100, 100, 100, 100])
    result4 = calculate_zscore(same_data, 3, 'close')
    expected4 = np.nan
    print(f"Edge case - zero std dev: {result4}")
    print(f"Expected: {expected4}")
    print(f"Test passed: {pd.isna(result4['close_zscore_3']) and pd.isna(expected4)}")


def test_sma():
    """Simple test for calculate_sma function"""
    print("\n" + "="*50)
    print("Testing calculate_sma...")

    # Test data
    test_data = pd.Series([100, 102, 98, 105, 103])
    print(f"Test data: {test_data.values}")

    # Window 3
    result1 = calculate_sma(test_data, 3, 'close')
    expected1 = test_data.rolling(window=3).mean().iloc[-1]
    print(f"Window 3 SMA: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['close_sma_3'] - expected1) < 1e-9}")

    # Window 5
    result2 = calculate_sma(test_data, 5, 'close')
    expected2 = test_data.rolling(window=5).mean().iloc[-1]
    print(f"Window 5 SMA: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2['close_sma_5'] - expected2) < 1e-9}")

    # Insufficient window
    result3 = calculate_sma(test_data, 6, 'close')
    print(f"Insufficient window (6): {result3}")
    print(f"Expected: {np.nan}")
    print(f"Test passed: {pd.isna(result3['close_sma_6'])}")


def test_ema():
    """Simple test for calculate_ema function"""
    print("\n" + "="*50)
    print("Testing calculate_ema...")

    test_data = pd.Series([100, 102, 98, 105, 103])
    print(f"Test data: {test_data.values}")

    # span=3
    result1 = calculate_ema(test_data, 3, 'close')
    expected1 = test_data.ewm(span=3).mean().iloc[-1]
    print(f"EMA span 3: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['close_ema_3'] - expected1) < 1e-9}")

    # span=5
    result2 = calculate_ema(test_data, 5, 'close')
    expected2 = test_data.ewm(span=5).mean().iloc[-1]
    print(f"EMA span 5: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2['close_ema_5'] - expected2) < 1e-9}")

    # insufficient
    result3 = calculate_ema(pd.Series([100, 101]), 5, 'close')
    print(f"Insufficient data: {result3}")
    print(f"Expected: {np.nan}")
    print(f"Test passed: {pd.isna(result3['close_ema_5'])}")


def test_wma():
    """Simple test for calculate_wma function"""
    print("\n" + "="*50)
    print("Testing calculate_wma...")

    test_data = pd.Series([100, 102, 98, 105, 103])
    print(f"Test data: {test_data.values}")

    # Window 3
    window = 3
    result1 = calculate_wma(test_data, window, 'close')
    values1 = test_data.tail(window).values  # oldest->newest
    weights1 = np.arange(1, window + 1)
    expected1 = np.sum(weights1 * values1) / np.sum(weights1)
    print(f"WMA window 3: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['close_wma_3'] - expected1) < 1e-9}")

    # Window 5
    window = 5
    result2 = calculate_wma(test_data, window, 'close')
    values2 = test_data.tail(window).values
    weights2 = np.arange(1, window + 1)
    expected2 = np.sum(weights2 * values2) / np.sum(weights2)
    print(f"WMA window 5: {result2}")
    print(f"Expected: {expected2}")
    print(f"Test passed: {abs(result2['close_wma_5'] - expected2) < 1e-9}")

    # Insufficient window
    result3 = calculate_wma(test_data, 6, 'close')
    print(f"Insufficient window (6): {result3}")
    print(f"Expected: {np.nan}")
    print(f"Test passed: {pd.isna(result3['close_wma_6'])}")


def test_ma_crossovers():
    """Simple test for calculate_ma_crossovers function"""
    print("\n" + "="*50)
    print("Testing calculate_ma_crossovers...")

    test_data = pd.Series([100, 102, 98, 105, 103])
    print(f"Test data: {test_data.values}")

    fast_window, slow_window = 2, 3
    result1 = calculate_ma_crossovers(test_data, fast_window, slow_window, 'close')

    ma_fast = test_data.rolling(window=fast_window).mean().iloc[-1]
    ma_slow = test_data.rolling(window=slow_window).mean().iloc[-1]
    expected1 = {
        'close_ma_cross_diff_2_3': ma_fast - ma_slow,
        'close_ma_cross_ratio_2_3': ma_fast / ma_slow if ma_slow != 0 else np.nan,
        'close_ma_cross_signal_2_3': 1 if ma_fast > ma_slow else 0
    }

    print(f"fast={fast_window}, slow={slow_window}: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['close_ma_cross_diff_2_3'] - expected1['close_ma_cross_diff_2_3']) < 1e-9 and abs(result1['close_ma_cross_ratio_2_3'] - expected1['close_ma_cross_ratio_2_3']) < 1e-9 and result1['close_ma_cross_signal_2_3'] == expected1['close_ma_cross_signal_2_3']}")

    # Insufficient data case
    short_data = pd.Series([100, 101])
    result2 = calculate_ma_crossovers(short_data, 3, 4, 'close')
    expected2 = {
        'close_ma_cross_diff_3_4': np.nan,
        'close_ma_cross_ratio_3_4': np.nan,
        'close_ma_cross_signal_3_4': np.nan
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
    result1 = calculate_ma_distance(price, ma_val, 'close', 'sma20')
    expected1 = {
        'close_ma_distance_sma20': price - ma_val,
        'close_ma_distance_pct_sma20': ((price - ma_val) / ma_val) * 100
    }
    print(f"price={price}, ma={ma_val}: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['close_ma_distance_sma20'] - expected1['close_ma_distance_sma20']) < 1e-9 and abs(result1['close_ma_distance_pct_sma20'] - expected1['close_ma_distance_pct_sma20']) < 1e-9}")

    # ma_value zero -> pct NaN
    price2 = 100.0
    ma_val2 = 0.0
    result2 = calculate_ma_distance(price2, ma_val2, 'close', 'sma20')
    print(f"ma=0 case: {result2}")
    print(f"Expected: {{'ma_distance': nan, 'ma_distance_pct': nan}}")
    print(f"Test passed: {pd.isna(result2['close_ma_distance_sma20']) and pd.isna(result2['close_ma_distance_pct_sma20'])}")

    # ma_value NaN
    price3 = 100.0
    ma_val3 = np.nan
    result3 = calculate_ma_distance(price3, ma_val3, 'close', 'sma20')
    print(f"ma=NaN case: {result3}")
    print(f"Expected: {{'ma_distance': nan, 'ma_distance_pct': nan}}")
    print(f"Test passed: {pd.isna(result3['close_ma_distance_sma20']) and pd.isna(result3['close_ma_distance_pct_sma20'])}")


def test_macd():
    """Simple test for calculate_macd function"""
    print("\n" + "="*50)
    print("Testing calculate_macd...")

    # Use small spans so short series works
    test_data = pd.Series([100, 102, 98, 105, 103])
    fast, slow, signal = 3, 5, 2
    result1 = calculate_macd(test_data, fast, slow, signal, 'close')

    ema_fast = test_data.ewm(span=fast).mean()
    ema_slow = test_data.ewm(span=slow).mean()
    macd_series = ema_fast - ema_slow
    macd_line_exp = macd_series.iloc[-1]
    macd_signal_exp = macd_series.ewm(span=signal).mean().iloc[-1]
    macd_hist_exp = macd_line_exp - macd_signal_exp

    print(f"fast={fast}, slow={slow}, signal={signal}: {result1}")
    print(f"Expected line/signal/hist: {macd_line_exp}, {macd_signal_exp}, {macd_hist_exp}")
    print(f"Test passed: {abs(result1['close_macd_line_3_5'] - macd_line_exp) < 1e-9 and abs(result1['close_macd_signal_2'] - macd_signal_exp) < 1e-9 and abs(result1['close_macd_histogram_3_5_2'] - macd_hist_exp) < 1e-9}")

    # Insufficient data for given spans
    short = pd.Series([100, 101])
    res2 = calculate_macd(short, 12, 26, 9, 'close')
    print(f"Insufficient data defaults: {res2}")
    print(f"Expected: nan for all keys")
    print(f"Test passed: {pd.isna(res2['close_macd_line_12_26']) and pd.isna(res2['close_macd_signal_9']) and pd.isna(res2['close_macd_histogram_12_26_9'])}")


def test_volume_ma():
    """Simple test for calculate_volume_ma function (wraps SMA)"""
    print("\n" + "="*50)
    print("Testing calculate_volume_ma...")

    vol = pd.Series([1000, 1200, 800, 1500, 1100])
    window = 3
    res = calculate_volume_ma(vol, window, 'volume')
    exp = vol.rolling(window=window).mean().iloc[-1]
    print(f"Volume MA window {window}: {res}")
    print(f"Expected: {exp}")
    print(f"Test passed: {abs(res['volume_sma_3'] - exp) < 1e-9}")


def test_rsi():
    """Simple test for calculate_rsi function"""
    print("\n" + "="*50)
    print("Testing calculate_rsi...")

    # General case with mixed moves (Wilder's method)
    data = pd.Series([100, 102, 98, 105, 103, 107, 101])
    window = 3
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain_series = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss_series = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    last_avg_gain = avg_gain_series.iloc[-1]
    last_avg_loss = avg_loss_series.iloc[-1]
    if np.isnan(last_avg_loss) or last_avg_loss == 0:
        expected1 = 100.0
    else:
        rs = last_avg_gain / last_avg_loss
        expected1 = 100 - (100 / (1 + rs))
    result1 = calculate_rsi(data, window, 'close')
    print(f"RSI window {window}: {result1}")
    print(f"Expected: {expected1}")
    print(f"Test passed: {abs(result1['close_rsi_3'] - expected1) < 1e-9}")

    # Strictly increasing -> losses zero => RSI 100
    inc = pd.Series([100, 101, 102, 103, 104])
    result2 = calculate_rsi(inc, 3, 'close')
    print(f"Increasing series RSI: {result2}")
    print(f"Expected: 100.0")
    print(f"Test passed: {abs(result2['close_rsi_3'] - 100.0) < 1e-9}")

    # Strictly decreasing -> gains zero; depending on window and smoothing, result tends toward 0
    dec = pd.Series([104, 103, 102, 101, 100])
    delta_d = dec.diff()
    gain_d = delta_d.where(delta_d > 0, 0.0)
    loss_d = -delta_d.where(delta_d < 0, 0.0)
    avg_gain_d = gain_d.ewm(alpha=1/3, adjust=False, min_periods=3).mean().iloc[-1]
    avg_loss_d = loss_d.ewm(alpha=1/3, adjust=False, min_periods=3).mean().iloc[-1]
    if np.isnan(avg_loss_d) or avg_loss_d == 0:
        expected_dec = 100.0
    else:
        rs_d = avg_gain_d / avg_loss_d
        expected_dec = 100 - (100 / (1 + rs_d))
    result3 = calculate_rsi(dec, 3, 'close')
    print(f"Decreasing series RSI: {result3}")
    print(f"Expected: {expected_dec}")
    print(f"Test passed: {abs(result3['close_rsi_3'] - expected_dec) < 1e-9}")

    # Insufficient data
    short = pd.Series([100, 101, 102])
    result4 = calculate_rsi(short, 5, 'close')
    print(f"Insufficient data RSI: {result4}")
    print(f"Expected: {np.nan}")
    print(f"Test passed: {pd.isna(result4['close_rsi_5'])}")


def test_stochastic():
    """Simple test for calculate_stochastic function"""
    print("\n" + "="*50)
    print("Testing calculate_stochastic...")

    high = pd.Series([105, 106, 107, 110, 111, 109, 112])
    low = pd.Series([100, 101, 102, 104, 105, 103, 106])
    close = pd.Series([102, 104, 103, 109, 106, 108, 107])
    k_window, d_window = 5, 3

    # Expected %K for last point
    ll = low.tail(k_window).min()
    hh = high.tail(k_window).max()
    if hh == ll:
        expected_k = 50.0
    else:
        expected_k = 100 * (close.iloc[-1] - ll) / (hh - ll)

    # Expected %D: average of last d_window K values computed over rolling k_window windows
    expected_k_values = []
    for i in range(d_window):
        idx = len(close) - 1 - i
        if idx < k_window - 1:
            break
        ll_i = low.iloc[idx - k_window + 1: idx + 1].min()
        hh_i = high.iloc[idx - k_window + 1: idx + 1].max()
        if hh_i == ll_i:
            k_i = 50.0
        else:
            k_i = 100 * (close.iloc[idx] - ll_i) / (hh_i - ll_i)
        expected_k_values.append(k_i)
    expected_d = np.mean(expected_k_values) if len(expected_k_values) == d_window else np.nan

    result = calculate_stochastic(high, low, close, k_window, d_window, 'close')
    print(f"Result: {result}")
    print(f"Expected K: {expected_k}, Expected D: {expected_d}")
    k_ok = abs(result['close_stoch_k_5'] - expected_k) < 1e-9 if not pd.isna(expected_k) else pd.isna(result['close_stoch_k_5'])
    d_ok = abs(result['close_stoch_d_5_3'] - expected_d) < 1e-9 if not pd.isna(expected_d) else pd.isna(result['close_stoch_d_5_3'])
    print(f"Test passed: {k_ok and d_ok}")

    # Edge case: insufficient data for k_window
    high_short = pd.Series([105, 106, 107])
    low_short = pd.Series([100, 101, 102])
    close_short = pd.Series([102, 103, 104])
    res_short = calculate_stochastic(high_short, low_short, close_short, 5, 3, 'close')
    print(f"Insufficient data (k_window=5): {res_short}")
    print("Expected: {'close_stoch_k_5': nan, 'close_stoch_d_5_3': nan}")
    print(f"Test passed: {pd.isna(res_short['close_stoch_k_5']) and pd.isna(res_short['close_stoch_d_5_3'])}")

    # Edge case: flat window (hh == ll) -> K=50; if enough data for D, D is mean of 50's
    high_flat = pd.Series([100, 100, 100, 100, 100, 100])
    low_flat = pd.Series([100, 100, 100, 100, 100, 100])
    close_flat = pd.Series([100, 100, 100, 100, 100, 100])
    res_flat = calculate_stochastic(high_flat, low_flat, close_flat, 5, 3, 'close')
    print(f"Flat window: {res_flat}")
    # With k_window=5, d_window=3 and only 6 points, there's not enough history to compute %D per implementation
    print("Expected: {'close_stoch_k_5': 50.0, 'close_stoch_d_5_3': nan}")
    print(f"Test passed: {abs(res_flat['close_stoch_k_5'] - 50.0) < 1e-9 and pd.isna(res_flat['close_stoch_d_5_3'])}")


def test_cci():
    """Simple test for calculate_cci function"""
    print("\n" + "="*50)
    print("Testing calculate_cci...")

    # Construct small series and use window=5 for computability
    high = pd.Series([10, 11, 12, 13, 14, 15])
    low = pd.Series([ 8,  9, 10, 11, 12, 13])
    close = pd.Series([ 9, 10, 11, 12, 13, 14])
    window = 5

    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=window).mean().iloc[-1]
    rolling_tp = tp.tail(window)
    mean_dev = np.mean(np.abs(rolling_tp - rolling_tp.mean()))
    if mean_dev == 0:
        expected = np.nan
    else:
        expected = (tp.iloc[-1] - sma_tp) / (0.015 * mean_dev)

    result = calculate_cci(high, low, close, window, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    ok = (pd.isna(result['close_cci_5']) and pd.isna(expected)) or (not pd.isna(result['close_cci_5']) and abs(result['close_cci_5'] - expected) < 1e-9)
    print(f"Test passed: {ok}")

    # Insufficient data
    res_insuff = calculate_cci(high.head(3), low.head(3), close.head(3), window, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_cci_5'])}")

    # Flat window -> mean deviation zero -> NaN
    hf = pd.Series([10, 10, 10, 10, 10])
    lf = pd.Series([10, 10, 10, 10, 10])
    cf = pd.Series([10, 10, 10, 10, 10])
    res_flat = calculate_cci(hf, lf, cf, 5, 'close')
    print(f"Flat window CCI: {res_flat}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_flat['close_cci_5'])}")


def test_roc():
    """Simple test for calculate_roc function"""
    print("\n" + "="*50)
    print("Testing calculate_roc...")

    data = pd.Series([100, 105, 102, 110, 115])

    # period = 1
    res1 = calculate_roc(data, 1, 'close')
    exp1 = (data.iloc[-1] - data.iloc[-2]) / data.iloc[-2] * 100
    print(f"ROC period 1: {res1}")
    print(f"Expected: {exp1}")
    print(f"Test passed: {abs(res1['close_roc_1'] - exp1) < 1e-9}")

    # period = 3
    res2 = calculate_roc(data, 3, 'close')
    past2 = data.iloc[-1-3]
    exp2 = (data.iloc[-1] - past2) / past2 * 100
    print(f"ROC period 3: {res2}")
    print(f"Expected: {exp2}")
    print(f"Test passed: {abs(res2['close_roc_3'] - exp2) < 1e-9}")

    # insufficient data
    res3 = calculate_roc(data, 10, 'close')
    print(f"Insufficient data (period=10): {res3}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res3['close_roc_10'])}")

    # past value zero -> NaN
    data_zero = pd.Series([0.0, 5.0, 10.0])
    res4 = calculate_roc(data_zero, 2, 'close')
    print(f"Past zero case: {res4}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res4['close_roc_2'])}")


def test_williams_r():
    """Simple test for calculate_williams_r function"""
    print("\n" + "="*50)
    print("Testing calculate_williams_r...")

    high = pd.Series([10, 12, 11, 13, 15, 14])
    low = pd.Series([ 8,  9, 10, 11, 12, 13])
    close = pd.Series([ 9, 11, 10, 12, 14, 13])
    window = 5

    highest_high = high.rolling(window=window).max().iloc[-1]
    lowest_low = low.rolling(window=window).min().iloc[-1]
    if highest_high == lowest_low:
        expected = -50.0
    else:
        expected = -100 * (highest_high - close.iloc[-1]) / (highest_high - lowest_low)

    result = calculate_williams_r(high, low, close, window, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    ok = (abs(result['close_williams_r_5'] - expected) < 1e-9)
    print(f"Test passed: {ok}")

    # Insufficient data -> NaN
    res_insuff = calculate_williams_r(high.head(3), low.head(3), close.head(3), window, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_williams_r_5'])}")

    # Flat window -> division by zero avoided -> implementation returns -50.0
    hf = pd.Series([10, 10, 10, 10, 10, 10])
    lf = pd.Series([10, 10, 10, 10, 10, 10])
    cf = pd.Series([10, 10, 10, 10, 10, 10])
    res_flat = calculate_williams_r(hf, lf, cf, 5, 'close')
    print(f"Flat window Williams %R: {res_flat}")
    print("Expected: -50.0")
    print(f"Test passed: {abs(res_flat['close_williams_r_5'] + 50.0) < 1e-9}")


def test_ultimate_oscillator():
    """Simple test for calculate_ultimate_oscillator function"""
    print("\n" + "="*50)
    print("Testing calculate_ultimate_oscillator...")

    # Construct a small realistic series and use small periods
    high = pd.Series([10, 12, 11, 13, 15, 14, 16, 17])
    low = pd.Series([ 8,  9, 10, 11, 12, 13, 14, 15])
    close = pd.Series([ 9, 11, 10, 12, 14, 13, 15, 16])

    periods = [3, 4, 5]
    weights = [4, 2, 1]

    # Expected calculation per implementation
    prev_close = close.shift(1)
    bp = close - np.minimum(low, prev_close)
    tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)

    avgs = []
    valid = True
    for i, p in enumerate(periods):
        if len(close) < p:
            valid = False
            break
        sum_bp = bp.rolling(window=p).sum().iloc[-1]
        sum_tr = tr.rolling(window=p).sum().iloc[-1]
        if sum_tr == 0 or pd.isna(sum_tr):
            valid = False
            break
        avgs.append((sum_bp / sum_tr) * weights[i])

    if valid:
        expected = 100 * sum(avgs) / sum(weights)
    else:
        expected = np.nan

    result = calculate_ultimate_oscillator(high, low, close, periods, weights, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    ok = (pd.isna(result['close_uo_3_4_5']) and pd.isna(expected)) or (not pd.isna(result['close_uo_3_4_5']) and abs(result['close_uo_3_4_5'] - expected) < 1e-9)
    print(f"Test passed: {ok}")

    # Insufficient data (max period > len)
    res_insuff = calculate_ultimate_oscillator(high.head(4), low.head(4), close.head(4), periods, weights, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_uo_3_4_5'])}")

    # Flat window where TR sums to 0 -> expect NaN
    hf = pd.Series([10, 10, 10, 10, 10, 10])
    lf = pd.Series([10, 10, 10, 10, 10, 10])
    cf = pd.Series([10, 10, 10, 10, 10, 10])
    res_flat = calculate_ultimate_oscillator(hf, lf, cf, [3, 4, 5], [4, 2, 1], 'close')
    print(f"Flat window UO: {res_flat}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_flat['close_uo_3_4_5'])}")


def test_mfi():
    """Simple test for calculate_mfi function"""
    print("\n" + "="*50)
    print("Testing calculate_mfi...")

    high = pd.Series([10, 12, 11, 13, 15, 14, 16])
    low = pd.Series([ 8,  9, 10, 11, 12, 13, 14])
    close = pd.Series([ 9, 11, 10, 12, 14, 13, 15])
    volume = pd.Series([1000, 1100, 900, 1200, 1300, 1250, 1400])
    window = 5

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    price_change = typical_price.diff()
    positive_flow = money_flow.where(price_change > 0, 0)
    negative_flow = money_flow.where(price_change < 0, 0)

    pos_sum = positive_flow.rolling(window=window).sum().iloc[-1]
    neg_sum = negative_flow.rolling(window=window).sum().iloc[-1]

    if neg_sum == 0:
        expected = 100.0
    else:
        mfr = pos_sum / neg_sum
        expected = 100 - (100 / (1 + mfr))

    result = calculate_mfi(high, low, close, volume, window, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    ok = (abs(result['close_mfi_5'] - expected) < 1e-9)
    print(f"Test passed: {ok}")

    # Insufficient data
    res_insuff = calculate_mfi(high.head(4), low.head(4), close.head(4), volume.head(4), window, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_mfi_5'])}")

    # All up moves -> negative_mf == 0 => 100
    high_up = pd.Series([10, 11, 12, 13, 14, 15])
    low_up = pd.Series([ 9, 10, 11, 12, 13, 14])
    close_up = pd.Series([ 9.5, 10.5, 11.5, 12.5, 13.5, 14.5])
    vol_up = pd.Series([1000, 1000, 1000, 1000, 1000, 1000])
    res_up = calculate_mfi(high_up, low_up, close_up, vol_up, 5, 'close')
    print(f"All up moves MFI: {res_up}")
    print("Expected: 100.0")
    print(f"Test passed: {abs(res_up['close_mfi_5'] - 100.0) < 1e-9}")

    # Flat typical price -> both flows zero -> implementation returns 100.0 (neg flow == 0)
    hf = pd.Series([10, 10, 10, 10, 10, 10])
    lf = pd.Series([10, 10, 10, 10, 10, 10])
    cf = pd.Series([10, 10, 10, 10, 10, 10])
    vf = pd.Series([1000, 1000, 1000, 1000, 1000, 1000])
    res_flat = calculate_mfi(hf, lf, cf, vf, 5, 'close')
    print(f"Flat MFI: {res_flat}")
    print("Expected: 100.0")
    print(f"Test passed: {abs(res_flat['close_mfi_5'] - 100.0) < 1e-9}")


def test_historical_volatility():
    """Simple test for calculate_historical_volatility function"""
    print("\n" + "="*50)
    print("Testing calculate_historical_volatility...")

    # Price series with varying log returns
    prices = pd.Series([100, 102, 101, 105, 104, 108, 110, 109])
    window = 5
    returns = np.log(prices / prices.shift(1)).dropna()
    expected = returns.tail(window).std() * np.sqrt(365)

    result = calculate_historical_volatility(prices, window, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Test passed: {abs(result['close_hv_5'] - expected) < 1e-9}")

    # Insufficient data
    res_insuff = calculate_historical_volatility(prices.head(3), window, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_hv_5'])}")

    # Nearly constant returns -> very low HV (could be exactly zero if perfectly constant)
    prices_const = pd.Series([100, 101, 102.01, 103.0301, 104.060401, 105.10100501])
    # This approximates a constant 1% compounding per step
    res_const = calculate_historical_volatility(prices_const, 5, 'close')
    print(f"Near-constant returns HV: {res_const}")
    print("Expected: approximately 0 (very small)")
    print(f"Test passed: {res_const['close_hv_5'] >= 0}")


def test_atr():
    """Simple test for calculate_atr function (SMA ATR)"""
    print("\n" + "="*50)
    print("Testing calculate_atr...")

    high = pd.Series([10, 12, 11, 13, 15, 14, 16])
    low = pd.Series([ 8,  9, 10, 11, 12, 13, 14])
    close = pd.Series([ 9, 11, 10, 12, 14, 13, 15])
    window = 5

    tr_list = []
    for i in range(1, len(close)):
        h = high.iloc[i]
        l = low.iloc[i]
        c_prev = close.iloc[i-1]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    expected = np.mean(tr_list[-window:])

    result = calculate_atr(high, low, close, window, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Test passed: {abs(result['close_atr_5'] - expected) < 1e-9}")

    # Insufficient data (needs at least window+1 closes)
    res_insuff = calculate_atr(high.head(4), low.head(4), close.head(4), window, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_atr_5'])}")

    # Flat series -> ATR = 0
    hf = pd.Series([10, 10, 10, 10, 10, 10])
    lf = pd.Series([10, 10, 10, 10, 10, 10])
    cf = pd.Series([10, 10, 10, 10, 10, 10])
    res_flat = calculate_atr(hf, lf, cf, 3, 'close')
    print(f"Flat ATR: {res_flat}")
    print("Expected: 0.0")
    print(f"Test passed: {abs(res_flat['close_atr_3'] - 0.0) < 1e-9}")


def test_bollinger_bands():
    """Simple test for calculate_bollinger_bands function"""
    print("\n" + "="*50)
    print("Testing calculate_bollinger_bands...")

    prices = pd.Series([100, 102, 101, 105, 104, 108, 110, 109])
    window = 5
    num_std = 2.0

    rolling = prices.tail(window)
    middle = rolling.mean()
    std = rolling.std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    current = prices.iloc[-1]
    bb_percent = 0.5 if width == 0 else (current - lower) / width

    expected = {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width,
        'percent': bb_percent
    }
    result = calculate_bollinger_bands(prices, window, num_std, 'close')
    key = {
        'upper': f'close_bb_upper_{window}_2',
        'middle': f'close_bb_middle_{window}',
        'lower': f'close_bb_lower_{window}_2',
        'width': f'close_bb_width_{window}_2',
        'percent': f'close_bb_percent_{window}_2',
    }
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    ok = (
        abs(result[key['upper']] - expected['upper']) < 1e-9 and
        abs(result[key['middle']] - expected['middle']) < 1e-9 and
        abs(result[key['lower']] - expected['lower']) < 1e-9 and
        abs(result[key['width']] - expected['width']) < 1e-9 and
        abs(result[key['percent']] - expected['percent']) < 1e-9
    )
    print(f"Test passed: {ok}")

    # Insufficient data
    res_insuff = calculate_bollinger_bands(prices.head(3), window, num_std, 'close')
    expected_insuff_keys = [key['upper'], key['middle'], key['lower'], key['width'], key['percent']]
    print(f"Insufficient data: {res_insuff}")
    print(f"Expected: NaNs for all keys")
    print(f"Test passed: {all(pd.isna(res_insuff[k]) for k in expected_insuff_keys)}")

    # Flat series -> std=0, width=0, percent=0.5
    flat = pd.Series([100, 100, 100, 100, 100])
    res_flat = calculate_bollinger_bands(flat, 5, 2.0, 'close')
    print(f"Flat bands: {res_flat}")
    print("Expected percent: 0.5")
    print(f"Test passed: {abs(res_flat['close_bb_percent_5_2'] - 0.5) < 1e-9 and abs(res_flat['close_bb_width_5_2'] - 0.0) < 1e-9}")


def test_volatility_ratio():
    """Simple test for calculate_volatility_ratio function"""
    print("\n" + "="*50)
    print("Testing calculate_volatility_ratio...")

    prices = pd.Series([100, 102, 101, 105, 104, 108, 110, 109, 115, 117, 116])
    short_w = 3
    long_w = 5

    returns = np.log(prices / prices.shift(1)).dropna()
    short_vol = returns.tail(short_w).std()
    long_vol = returns.tail(long_w).std()
    expected = np.nan if long_vol == 0 else short_vol / long_vol

    result = calculate_volatility_ratio(prices, short_w, long_w, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    if np.isnan(expected):
        print(f"Test passed: {pd.isna(result['close_vol_ratio_3_5'])}")
    else:
        print(f"Test passed: {abs(result['close_vol_ratio_3_5'] - expected) < 1e-9}")

    # Insufficient data (len < long_window+1)
    res_insuff = calculate_volatility_ratio(prices.head(4), 2, 5, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_vol_ratio_2_5'])}")

    # Flat long window -> long_vol == 0 -> NaN
    # Construct geometric series with constant return across last long_w periods
    base = 100.0
    r = 1.01
    # Need at least long_w + 1 prices to get long_w returns
    flat_prices = [base]
    for _ in range(6):
        flat_prices.append(flat_prices[-1] * r)
    flat_series = pd.Series(flat_prices)
    res_flat = calculate_volatility_ratio(flat_series, 2, 5, 'close')
    print(f"Flat long window ratio: {res_flat}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_flat['close_vol_ratio_2_5'])}")


def test_parkinson_volatility():
    """Simple test for calculate_parkinson_volatility function"""
    print("\n" + "="*50)
    print("Testing calculate_parkinson_volatility...")

    high = pd.Series([10, 12, 11, 13, 15, 14, 16, 17])
    low = pd.Series([ 9, 10, 10, 11, 13, 12, 14, 15])
    window = 5

    log_hl_ratio = np.log(high / low)
    parkinson_values = log_hl_ratio ** 2
    expected = np.sqrt((1 / (4 * np.log(2))) * parkinson_values.tail(window).mean())

    result = calculate_parkinson_volatility(high, low, window, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Test passed: {abs(result['close_parkinson_5'] - expected) < 1e-9}")

    # Insufficient data
    res_insuff = calculate_parkinson_volatility(high.head(3), low.head(3), window, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_parkinson_5'])}")

    # Flat window (high == low) -> zero volatility
    hf = pd.Series([10, 10, 10, 10, 10, 10])
    lf = pd.Series([10, 10, 10, 10, 10, 10])
    res_flat = calculate_parkinson_volatility(hf, lf, 5, 'close')
    print(f"Flat Parkinson volatility: {res_flat}")
    print("Expected: 0.0")
    print(f"Test passed: {abs(res_flat['close_parkinson_5'] - 0.0) < 1e-9}")


def test_garman_klass_volatility():
    """Simple test for calculate_garman_klass_volatility function"""
    print("\n" + "="*50)
    print("Testing calculate_garman_klass_volatility...")

    open_s = pd.Series([10, 10.5, 10.2, 10.8, 11.0, 10.9])
    high = pd.Series( [10.8, 10.9, 10.6, 11.1, 11.2, 11.0])
    low = pd.Series(  [ 9.8, 10.2, 10.0, 10.6, 10.9, 10.7])
    close = pd.Series([10.6, 10.3, 10.5, 11.0, 11.1, 10.8])
    window = 5

    term1 = 0.5 * (np.log(high / low)) ** 2
    term2 = (2 * np.log(2) - 1) * (np.log(close / open_s)) ** 2
    gk_values = term1 - term2
    expected = float(np.sqrt(gk_values.tail(window).mean()))

    result = calculate_garman_klass_volatility(high, low, open_s, close, window, 'close')
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Test passed: {abs(result['close_gk_5'] - expected) < 1e-9}")

    # Insufficient data
    res_insuff = calculate_garman_klass_volatility(high.head(3), low.head(3), open_s.head(3), close.head(3), window, 'close')
    print(f"Insufficient data: {res_insuff}")
    print("Expected: nan")
    print(f"Test passed: {pd.isna(res_insuff['close_gk_5'])}")


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
    test_rsi()
    test_stochastic()
    test_cci()
    test_roc()
    test_williams_r()
    test_ultimate_oscillator()
    test_mfi()
    test_historical_volatility()
    test_atr()
    test_bollinger_bands()
    test_volatility_ratio()
    test_parkinson_volatility()
