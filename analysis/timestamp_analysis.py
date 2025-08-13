import pandas as pd
import numpy as np
from pathlib import Path

def analyze_timestamps():
    """Analyze timestamp alignment between different timeframes"""
    
    data_dir = Path("../data/crypto_cex")
    
    # Load sample data
    files = {
        '4h': 'BINANCE_BTCUSDT, 240.csv',
        '12h': 'BINANCE_BTCUSDT, 720.csv', 
        '1d': 'BINANCE_BTCUSDT, 1D.csv',
        '3d': 'BINANCE_BTCUSDT, 3D.csv'
    }
    
    dataframes = {}
    
    for tf, filename in files.items():
        filepath = data_dir / filename
        df = pd.read_csv(filepath)
        
        # Parse timestamps and normalize timezone
        df['timestamp'] = pd.to_datetime(df['time'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            # For tz-naive timestamps, assume UTC
            df['timestamp'] = df['timestamp']
        
        dataframes[tf] = df[['timestamp', 'RSI']].copy()
        dataframes[tf].columns = ['timestamp', f'RSI_{tf}']
    
    # Analyze overlap periods
    print("=== Timestamp Range Analysis ===")
    for tf, df in dataframes.items():
        print(f"{tf}: {df['timestamp'].min()} to {df['timestamp'].max()} ({len(df)} records)")
    
    # Find common time range
    start_times = [df['timestamp'].min() for df in dataframes.values()]
    end_times = [df['timestamp'].max() for df in dataframes.values()]
    
    common_start = max(start_times)
    common_end = min(end_times)
    
    print(f"\nCommon time range: {common_start} to {common_end}")
    
    # Check alignment for a sample period
    print("\n=== Sample Alignment Check (first 10 4h records) ===")
    base_4h = dataframes['4h'].head(10)
    
    for i, row in base_4h.iterrows():
        ts_4h = row['timestamp']
        print(f"\n4h timestamp: {ts_4h}")
        
        # Check if this timestamp aligns with other timeframes
        for tf in ['12h', '1d', '3d']:
            df_tf = dataframes[tf]
            matches = df_tf[df_tf['timestamp'] <= ts_4h].tail(1)
            if not matches.empty:
                last_match = matches.iloc[0]
                time_diff = ts_4h - last_match['timestamp']
                print(f"  {tf} latest: {last_match['timestamp']} (diff: {time_diff})")
    
    return dataframes

def analyze_empty_columns():
    """Analyze which columns are consistently empty across files"""
    
    data_dir = Path("../data/crypto_cex")
    
    # Check multiple files to confirm empty columns pattern
    test_files = [
        'BINANCE_BTCUSDT, 240.csv',
        'BINANCE_SOLUSDT, 240.csv', 
        'COINBASE_BTCUSD, 240.csv'
    ]
    
    empty_cols_consistent = None
    
    for filename in test_files:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            null_counts = df.isnull().sum()
            completely_empty = set(null_counts[null_counts == len(df)].index)
            
            print(f"\n{filename}:")
            print(f"  Shape: {df.shape}")
            print(f"  Empty columns: {sorted(completely_empty)}")
            
            if empty_cols_consistent is None:
                empty_cols_consistent = completely_empty
            else:
                empty_cols_consistent = empty_cols_consistent.intersection(completely_empty)
    
    print(f"\nConsistently empty columns across all files: {sorted(empty_cols_consistent)}")
    return empty_cols_consistent

if __name__ == "__main__":
    print("Starting timestamp analysis...")
    dataframes = analyze_timestamps()
    
    print("\n" + "="*50)
    empty_cols = analyze_empty_columns()