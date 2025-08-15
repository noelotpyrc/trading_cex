#!/usr/bin/env python3
"""
Final Working Smooth RSI - based on the simple approach that works correctly
"""

import pandas as pd
import numpy as np


def calculate_rsi_wilder(prices, period=14):
    """Standard Wilder RSI calculation"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    rsi_values = []
    avg_gain = None
    avg_loss = None
    
    for i in range(len(prices)):
        if i < period:
            rsi_values.append(np.nan)
        elif i == period:
            avg_gain = gains.iloc[1:period+1].mean()
            avg_loss = losses.iloc[1:period+1].mean()
            
            if avg_loss == 0:
                rsi_val = 100
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
            rsi_values.append(rsi_val)
        else:
            avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
            
            if avg_loss == 0:
                rsi_val = 100
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
            rsi_values.append(rsi_val)
    
    return pd.Series(rsi_values, index=prices.index)


def calculate_sma_1h(prices, periods=[14, 21, 35]):
    """
    Calculate Simple Moving Averages for 1H data
    
    Args:
        prices: Series of close prices
        periods: List of periods to calculate SMA for (default: [14, 21, 35])
    
    Returns:
        Dictionary of SMA series with keys like 'SMA_14_1H', 'SMA_21_1H', 'SMA_35_1H'
    """
    sma_dict = {}
    
    for period in periods:
        sma_values = prices.rolling(window=period, min_periods=period).mean()
        sma_dict[f'SMA_{period}_1H'] = sma_values
    
    return sma_dict


def calculate_bollinger_bands_1h(prices, period=20, std_multiplier=2.0):
    """
    Calculate Bollinger Bands for 1H data
    
    Args:
        prices: Series of close prices
        period: Period for moving average and standard deviation (default: 20)
        std_multiplier: Standard deviation multiplier (default: 2.0)
    
    Returns:
        Dictionary with 'BB_Middle_1H', 'BB_Upper_1H', 'BB_Lower_1H', 'BB_Width_1H', 'BB_Percent_1H'
    """
    # Middle band (SMA)
    bb_middle = prices.rolling(window=period, min_periods=period).mean()
    
    # Standard deviation
    bb_std = prices.rolling(window=period, min_periods=period).std()
    
    # Upper and lower bands
    bb_upper = bb_middle + (bb_std * std_multiplier)
    bb_lower = bb_middle - (bb_std * std_multiplier)
    
    # Bollinger Band Width (volatility measure)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # %B (Bollinger Band Percent) - where price is within the bands
    bb_percent = (prices - bb_lower) / (bb_upper - bb_lower)
    
    return {
        'BB_Middle_1H': bb_middle,
        'BB_Upper_1H': bb_upper,
        'BB_Lower_1H': bb_lower,
        'BB_Width_1H': bb_width,
        'BB_Percent_1H': bb_percent
    }


def get_period_start(timestamp, timeframe):
    if timeframe == '4H':
        hour = (timestamp.hour // 4) * 4
        return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
    elif timeframe == '12H':
        hour = (timestamp.hour // 12) * 12
        return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
    elif timeframe == '1D':
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)


def calculate_smooth_rsi_timeframe(df, timeframe):
    """Calculate smooth RSI for specific timeframe using the working simple approach"""
    print(f"Processing smooth {timeframe} RSI...")
    
    # Map timeframe to resample frequency and hours
    tf_mapping = {
        '4H': ('4h', 4),
        '12H': ('12h', 12), 
        '1D': ('1D', 24)
    }
    
    resample_freq, tf_hours = tf_mapping[timeframe]
    
    # Get regular resampled data
    df_resampled = df.resample(resample_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Calculate regular RSI
    df_resampled['RSI_regular'] = calculate_rsi_wilder(df_resampled['close'], 14)
    
    rsi_values = []
    
    for i, timestamp in enumerate(df.index):
        if i % 5000 == 0:  # Progress indicator
            print(f"  Processing {i}/{len(df)} bars...")
            
        current_period_start = get_period_start(timestamp, timeframe)
        
        # Find current period in resampled data
        if current_period_start in df_resampled.index:
            # Determine if this is the last hour of the period
            period_end_hour = current_period_start + pd.Timedelta(hours=tf_hours-1)
            
            if timestamp.replace(minute=0, second=0, microsecond=0) == period_end_hour:
                # This is exactly at period end - use regular RSI
                regular_rsi = df_resampled.loc[current_period_start, 'RSI_regular']
                rsi_values.append(regular_rsi)
            else:
                # This is within the period - calculate live RSI
                period_idx = df_resampled.index.get_loc(current_period_start)
                
                if period_idx >= 14:  # Need enough history
                    # Get historical closes
                    historical_closes = df_resampled['close'].iloc[period_idx-14:period_idx].tolist()
                    
                    # Add current live close
                    current_period_mask = (df.index >= current_period_start) & (df.index <= timestamp)
                    current_period_data = df.loc[current_period_mask]
                    
                    if len(current_period_data) > 0:
                        live_close = current_period_data['close'].iloc[-1]
                        historical_closes.append(live_close)
                        
                        # Calculate RSI with this modified close series
                        closes_series = pd.Series(historical_closes)
                        rsi_series = calculate_rsi_wilder(closes_series, 14)
                        rsi_values.append(rsi_series.iloc[-1])
                    else:
                        rsi_values.append(np.nan)
                else:
                    rsi_values.append(np.nan)
        else:
            rsi_values.append(np.nan)
    
    print(f"  ✅ Completed {timeframe} RSI")
    return rsi_values


if __name__ == "__main__":
    input_file = "data/BINANCE_SOLUSDT.P, 60.csv"
    output_file = "data/binance_solusdt.p_smooth_multi_rsi_corrected.csv"
    
    print("🔧 Final Working Smooth Multi-Timeframe RSI Calculator")
    print("=" * 60)
    
    print("Loading 1H data...")
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    print(f"Loaded {len(df)} 1H bars")
    
    # Calculate 1H RSI
    print("Calculating 1H RSI...")
    df['RSI_1H'] = calculate_rsi_wilder(df['close'], 14)
    
    # Calculate 1H SMAs
    print("Calculating 1H SMAs (14, 21, 35)...")
    sma_dict = calculate_sma_1h(df['close'], [14, 21, 35])
    for sma_name, sma_values in sma_dict.items():
        df[sma_name] = sma_values
    
    # Calculate 1H Bollinger Bands
    print("Calculating 1H Bollinger Bands (20, 2.0)...")
    bb_dict = calculate_bollinger_bands_1h(df['close'], period=20, std_multiplier=2.0)
    for bb_name, bb_values in bb_dict.items():
        df[bb_name] = bb_values
    
    # Calculate smooth RSI for each timeframe
    timeframes = ['4H', '12H', '1D']
    
    for timeframe in timeframes:
        rsi_values = calculate_smooth_rsi_timeframe(df, timeframe)
        df[f'RSI_{timeframe}'] = rsi_values
    
    # Select output columns and reset index
    output_cols = ['open', 'high', 'low', 'close', 'Volume', 'RSI_1H', 'RSI_4H', 'RSI_12H', 'RSI_1D', 
                   'SMA_14_1H', 'SMA_21_1H', 'SMA_35_1H', 
                   'BB_Middle_1H', 'BB_Upper_1H', 'BB_Lower_1H', 'BB_Width_1H', 'BB_Percent_1H']
    result_df = df[output_cols].reset_index()
    
    # Save result
    result_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Process completed!")
    print(f"Output: {output_file}")
    print(f"Shape: {result_df.shape}")
    
    # Validation - check the problematic timestamp
    print(f"\n🔍 Validation:")
    
    # Check 2023-01-03 15:00 vs 4H 12:00
    test_smooth = result_df[result_df['time'].astype(str) == '2023-01-03 15:00:00+00:00']
    if len(test_smooth) > 0:
        smooth_rsi = test_smooth['RSI_4H'].iloc[0]
        smooth_close = test_smooth['close'].iloc[0]
        
        # Load original 4H data
        df_4h_orig = pd.read_csv("data/binance_btcusdt.p_4h_rsi_14.csv")
        test_orig = df_4h_orig[df_4h_orig['time'].astype(str) == '2023-01-03 12:00:00+00:00']
        
        if len(test_orig) > 0:
            orig_rsi = test_orig['RSI_14'].iloc[0]
            orig_close = test_orig['close'].iloc[0]
            
            print(f"2023-01-03 15:00 (smooth): Close={smooth_close}, RSI={smooth_rsi:.8f}")
            print(f"2023-01-03 12:00 (original): Close={orig_close}, RSI={orig_rsi:.8f}")
            print(f"Difference: {abs(smooth_rsi - orig_rsi):.8f}")
            
            if abs(smooth_rsi - orig_rsi) < 0.000001:
                print("✅ Perfect match!")
            else:
                print("❌ Still not matching")
    
    # Show sample data
    print(f"\nSample data (last 5 rows):")
    sample_cols = ['time', 'close', 'RSI_1H', 'RSI_4H', 'RSI_12H', 'RSI_1D', 'SMA_14_1H', 'BB_Upper_1H', 'BB_Lower_1H', 'BB_Width_1H']
    print(result_df[sample_cols].tail().round(2))