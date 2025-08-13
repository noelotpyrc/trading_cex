import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_csv_files():
    """Analyze the structure and content of crypto CSV files"""
    
    data_dir = Path("../data/crypto_cex")
    
    # Sample files to analyze
    sample_files = {
        '4h': 'BINANCE_BTCUSDT, 240.csv',
        '12h': 'BINANCE_BTCUSDT, 720.csv', 
        '1d': 'BINANCE_BTCUSDT, 1D.csv',
        '3d': 'BINANCE_BTCUSDT, 3D.csv'
    }
    
    results = {}
    
    for timeframe, filename in sample_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\n=== Analysis for {timeframe} ({filename}) ===")
            df = pd.read_csv(filepath)
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Check for empty/null columns
            null_counts = df.isnull().sum()
            empty_cols = null_counts[null_counts == len(df)].index.tolist()
            mostly_empty_cols = null_counts[null_counts > len(df) * 0.9].index.tolist()
            
            print(f"Completely empty columns: {empty_cols}")
            print(f"Mostly empty columns (>90% null): {mostly_empty_cols}")
            
            # Analyze time column
            print(f"Time column sample: {df['time'].head(3).tolist()}")
            print(f"Time range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
            
            # Check RSI column
            rsi_non_null = df['RSI'].notna().sum()
            print(f"RSI non-null values: {rsi_non_null} / {len(df)} ({rsi_non_null/len(df)*100:.1f}%)")
            
            # Sample RSI values
            rsi_sample = df['RSI'].dropna().head(5)
            print(f"RSI sample values: {rsi_sample.tolist()}")
            
            results[timeframe] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'empty_cols': empty_cols,
                'mostly_empty_cols': mostly_empty_cols,
                'rsi_coverage': rsi_non_null/len(df),
                'time_format': str(type(df['time'].iloc[0])),
                'first_time': df['time'].iloc[0],
                'last_time': df['time'].iloc[-1]
            }
            
        else:
            print(f"File not found: {filepath}")
    
    return results

if __name__ == "__main__":
    results = analyze_csv_files()
    
    # Compare columns across timeframes
    print("\n=== Column Comparison ===")
    all_columns = set()
    for tf, data in results.items():
        all_columns.update(data['columns'])
    
    print(f"Total unique columns across all timeframes: {len(all_columns)}")
    
    # Check column consistency
    base_columns = results.get('4h', {}).get('columns', [])
    for tf, data in results.items():
        if tf != '4h':
            missing = set(base_columns) - set(data['columns'])
            extra = set(data['columns']) - set(base_columns)
            if missing or extra:
                print(f"{tf} vs 4h - Missing: {missing}, Extra: {extra}")
            else:
                print(f"{tf} vs 4h - Columns match perfectly")