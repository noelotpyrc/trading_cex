#!/usr/bin/env python3
"""
Simple RSI Calculator for multiple timeframes
"""

import pandas as pd
import numpy as np


def calculate_rsi(prices, period=14):
    """Calculate RSI using Wilder's smoothing (RMA)"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    rsi_values = []
    avg_gain = None
    avg_loss = None
    
    for i in range(len(prices)):
        if i < period:
            # Not enough data for RSI calculation
            rsi_values.append(np.nan)
        elif i == period:
            # First RSI: simple average of first 'period' gains/losses
            avg_gain = gains.iloc[1:period+1].mean()
            avg_loss = losses.iloc[1:period+1].mean()
            
            if avg_loss == 0:
                rsi_val = 100
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
            rsi_values.append(rsi_val)
        else:
            # Subsequent RSI: Wilder's smoothing (RMA)
            avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
            
            if avg_loss == 0:
                rsi_val = 100
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
            rsi_values.append(rsi_val)
    
    return pd.Series(rsi_values, index=prices.index)


def resample_ohlc(df, timeframe):
    """Resample OHLC data to target timeframe"""
    if 'time' in df.columns:
        df = df.set_index('time')
    
    agg_rules = {
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last'
    }
    
    if 'volume' in df.columns:
        agg_rules['volume'] = 'sum'
    
    return df.resample(timeframe).agg(agg_rules).dropna()


def add_multi_timeframe_rsi(df, timeframes=['1H', '4H', '12H', '1D'], periods=[14]):
    """Add RSI for multiple timeframes to dataframe"""
    result = df.copy()
    
    if 'time' in result.columns:
        result['time'] = pd.to_datetime(result['time'])
        result = result.set_index('time')
    
    for timeframe in timeframes:
        for period in periods:
            # Resample data
            resampled = resample_ohlc(result, timeframe)
            
            # Calculate RSI
            rsi = calculate_rsi(resampled['close'], period)
            
            # Column name
            col_name = f'RSI_{timeframe}' if len(periods) == 1 else f'RSI_{timeframe}_{period}'
            
            # Align with original timeframe
            result[col_name] = rsi.reindex(result.index, method='ffill')
    
    return result.reset_index()


def add_rsi_bars(df, periods=[7, 14, 21, 30]):
    """Add RSI based on number of bars"""
    result = df.copy()
    
    for period in periods:
        col_name = f'RSI_{period}bars'
        result[col_name] = calculate_rsi(result['close'], period)
    
    return result


if __name__ == "__main__":
    input_file = "data/BINANCE_BTCUSDT.P, 60.csv"
    
    # Read data
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    timeframes = [
        ('1H', 'data/binance_btcusdt.p_1h_rsi_14.csv'),
        ('4H', 'data/binance_btcusdt.p_4h_rsi_14.csv'), 
        ('12H', 'data/binance_btcusdt.p_12h_rsi_14.csv'),
        ('1D', 'data/binance_btcusdt.p_1d_rsi_14.csv')
    ]
    
    for timeframe, output_file in timeframes:
        print(f"\nProcessing {timeframe} timeframe...")
        
        if timeframe == '1H':
            # Use original 1H data
            working_df = df.copy()
        else:
            # Resample to target timeframe
            working_df = resample_ohlc(df, timeframe)
        
        # Calculate RSI
        working_df['RSI_14'] = calculate_rsi(working_df['close'], 14)
        
        # Select required columns and reset index
        available_cols = ['open', 'high', 'low', 'close', 'RSI_14']
        if 'Volume' in working_df.columns:
            available_cols.insert(-1, 'Volume')
        output_df = working_df[available_cols].reset_index()
        
        # Save to file
        output_df.to_csv(output_file, index=False)
        
        print(f"✅ {timeframe}: {output_df.shape[0]} bars saved to {output_file}")
        print(f"   Sample RSI (last 3): {output_df['RSI_14'].tail(3).round(2).tolist()}")
    
    print(f"\n🎉 All timeframes completed!")