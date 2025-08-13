import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional

class CryptoDataMerger:
    """
    Merges 4h crypto data with RSI from higher timeframes (12h, 1d, 3d).
    Uses 4h as base table and adds RSI columns from higher timeframes.
    """
    
    def __init__(self, data_dir: str = "../data/crypto_cex"):
        self.data_dir = Path(data_dir)
        
        # Define columns to remove (consistently empty across all files)
        self.empty_columns = [
            'MA №4', 'Futures Open Interest',
            'Crypto Open Interest (Open)', 'Crypto Open Interest (High',
            'Crypto Open Interest (Low)', 'Crypto Open Interest (Close)',
            'Regular Bullish', 'Regular Bullish Label',
            'Regular Bearish', 'Regular Bearish Label',
            'Regular Bullish.1', 'Regular Bullish Label.1',
            'Regular Bearish.1', 'Regular Bearish Label.1'
        ]
    
    def get_all_symbols(self) -> List[str]:
        """Get all unique symbol-exchange combinations from filenames"""
        symbols = set()
        for file in self.data_dir.glob("*.csv"):
            # Extract symbol part (e.g., "BINANCE_BTCUSDT" from "BINANCE_BTCUSDT, 240.csv")
            symbol = file.stem.split(',')[0].strip()
            symbols.add(symbol)
        return sorted(list(symbols))
    
    def load_timeframe_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data for a specific symbol and timeframe"""
        timeframe_map = {
            '4h': '240',
            '12h': '720', 
            '1d': '1D',
            '3d': '3D'
        }
        
        filename = f"{symbol}, {timeframe_map[timeframe]}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
        
        df = pd.read_csv(filepath)
        
        # Parse timestamps and normalize timezone
        df['timestamp'] = pd.to_datetime(df['time'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        return df
    
    def merge_rsi_data(self, base_4h: pd.DataFrame, higher_tf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge RSI data from higher timeframes avoiding look-ahead bias.
        
        Key principle: RSI is calculated using close price, so we can only use 
        COMPLETED higher timeframe RSI values that existed before the current 4h bar.
        
        Logic:
        - For 4h bar at time T, use the most recent COMPLETED higher timeframe RSI
        - This means using the higher timeframe RSI from the previous period
        """
        result = base_4h.copy()
        
        # Sort base data by timestamp
        result = result.sort_values('timestamp').reset_index(drop=True)
        
        # Rename original RSI column for clarity
        result.rename(columns={'RSI': 'RSI_4h'}, inplace=True)
        
        for tf_name, df_higher in higher_tf_data.items():
            if df_higher is None or df_higher.empty:
                result[f'RSI_{tf_name}'] = np.nan
                continue
            
            # Sort higher timeframe data
            df_higher = df_higher.sort_values('timestamp').reset_index(drop=True)
            
            # Use backward fill with strict time alignment to avoid look-ahead bias
            # For each 4h timestamp, find the most recent COMPLETED higher timeframe RSI
            
            # Method: Use merge_asof with direction='backward' 
            # This ensures we only use RSI values from bars that were completed BEFORE the current 4h bar
            
            # Shift higher timeframe timestamps forward by one period to represent when the RSI becomes "available"
            # This simulates that RSI calculated at time T is only known AFTER time T
            if tf_name == '12h':
                # 12h RSI calculated at 00:00 is available for 4h bars after 00:00
                # 12h RSI calculated at 12:00 is available for 4h bars after 12:00
                df_higher_shifted = df_higher.copy()
                df_higher_shifted['timestamp'] = df_higher_shifted['timestamp'] + pd.Timedelta(hours=12)
            elif tf_name == '1d':
                # 1d RSI calculated at 00:00 is available for 4h bars after 00:00 the next day
                df_higher_shifted = df_higher.copy() 
                df_higher_shifted['timestamp'] = df_higher_shifted['timestamp'] + pd.Timedelta(days=1)
            elif tf_name == '3d':
                # 3d RSI calculated at 00:00 is available for 4h bars after 00:00 three days later
                df_higher_shifted = df_higher.copy()
                df_higher_shifted['timestamp'] = df_higher_shifted['timestamp'] + pd.Timedelta(days=3)
            else:
                df_higher_shifted = df_higher.copy()
            
            # Now merge using backward search - this gives us the most recent COMPLETED RSI
            merged = pd.merge_asof(
                result[['timestamp']].reset_index(),
                df_higher_shifted[['timestamp', 'RSI']],
                on='timestamp',
                direction='backward',
                suffixes=('', f'_{tf_name}')
            )
            
            result[f'RSI_{tf_name}'] = merged['RSI'].values
        
        return result
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove empty columns and clean the dataset"""
        # Remove empty columns that exist in the dataframe
        cols_to_remove = [col for col in self.empty_columns if col in df.columns]
        df_clean = df.drop(columns=cols_to_remove)
        
        print(f"Removed {len(cols_to_remove)} empty columns: {cols_to_remove}")
        
        # Reorder columns for better readability
        priority_cols = ['time', 'timestamp', 'open', 'high', 'low', 'close', 'Volume']
        rsi_cols = [col for col in df_clean.columns if col.startswith('RSI_')]
        other_cols = [col for col in df_clean.columns if col not in priority_cols + rsi_cols]
        
        final_order = priority_cols + sorted(rsi_cols) + sorted(other_cols)
        final_order = [col for col in final_order if col in df_clean.columns]
        
        return df_clean[final_order]
    
    def process_symbol(self, symbol: str, save_output: bool = True) -> Optional[pd.DataFrame]:
        """Process a single symbol and merge all timeframes"""
        print(f"\nProcessing {symbol}...")
        
        # Load data for all timeframes
        df_4h = self.load_timeframe_data(symbol, '4h')
        if df_4h is None:
            print(f"Skipping {symbol} - no 4h data found")
            return None
        
        higher_tf_data = {}
        for tf in ['12h', '1d', '3d']:
            df_tf = self.load_timeframe_data(symbol, tf)
            higher_tf_data[tf] = df_tf
            
            if df_tf is not None:
                print(f"  Loaded {tf}: {len(df_tf)} records")
            else:
                print(f"  Warning: No {tf} data for {symbol}")
        
        # Merge RSI data
        merged_data = self.merge_rsi_data(df_4h, higher_tf_data)
        
        # Clean the data
        final_data = self.clean_data(merged_data)
        
        print(f"  Final shape: {final_data.shape}")
        print(f"  RSI coverage - 4h: {final_data['RSI_4h'].notna().sum()}/{len(final_data)}")
        for tf in ['12h', '1d', '3d']:
            if f'RSI_{tf}' in final_data.columns:
                coverage = final_data[f'RSI_{tf}'].notna().sum()
                print(f"  RSI coverage - {tf}: {coverage}/{len(final_data)}")
        
        # Save output if requested
        if save_output:
            output_file = f"merged_{symbol.replace('_', '_').lower()}_4h_with_multi_tf_rsi.csv"
            final_data.to_csv(output_file, index=False)
            print(f"  Saved to: {output_file}")
        
        return final_data
    
    def process_all_symbols(self, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Process all symbols found in the data directory"""
        symbols = self.get_all_symbols()
        
        if limit:
            symbols = symbols[:limit]
            
        print(f"Found {len(symbols)} symbols to process: {symbols}")
        
        results = {}
        for symbol in symbols:
            try:
                result = self.process_symbol(symbol)
                if result is not None:
                    results[symbol] = result
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        print(f"\nSuccessfully processed {len(results)}/{len(symbols)} symbols")
        return results

def main():
    """Main execution function"""
    # Initialize the merger
    merger = CryptoDataMerger()
    
    # Process a sample symbol first
    print("=== SAMPLE PROCESSING ===")
    sample_result = merger.process_symbol("BINANCE_BTCUSDT")
    
    if sample_result is not None:
        print(f"\nSample output columns: {list(sample_result.columns)}")
        print(f"\nSample data (first 5 rows):")
        display_cols = ['time', 'open', 'close', 'RSI_4h', 'RSI_12h', 'RSI_1d', 'RSI_3d']
        available_cols = [col for col in display_cols if col in sample_result.columns]
        print(sample_result[available_cols].head().to_string(index=False))
    
    # Process all symbols
    print("\n=== PROCESSING ALL SYMBOLS ===")
    all_results = merger.process_all_symbols()
    print(f"Processing complete. Generated {len(all_results)} merged files.")

if __name__ == "__main__":
    main()