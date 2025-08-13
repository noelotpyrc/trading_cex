#!/usr/bin/env python3
"""
Strategy Runner Script

Run Conservative RSI Strategy on crypto datasets
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for better compatibility
matplotlib.use('TkAgg')  # Use TkAgg backend for display

import sys
sys.path.append('..')
from strategies.conservative_rsi_strategy import ConservativeRSIStrategy, CryptoRSIData


def run_conservative_rsi_backtest(csv_file, initial_cash=100000, plot=True):
    """
    Run backtest on a single crypto dataset
    
    Args:
        csv_file: Path to merged crypto RSI CSV file
        initial_cash: Starting capital
        plot: Whether to show plots
    """
    
    print(f"\n🚀 Starting Conservative RSI Backtest")
    print(f"📁 Data file: {csv_file}")
    print(f"💰 Initial capital: ${initial_cash:,}")
    
    # Load and prepare data
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    # Remove rows with missing RSI data
    initial_rows = len(df)
    df = df.dropna(subset=['RSI_4h', 'RSI_12h', 'RSI_1d', 'RSI_3d'])
    print(f"📊 Data: {len(df)} rows (removed {initial_rows - len(df)} with missing RSI)")
    print(f"📅 Period: {df.index[0]} to {df.index[-1]}")
    
    # Initialize backtrader
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(ConservativeRSIStrategy, verbose=True)
    
    # Add data feed
    data = CryptoRSIData(dataname=df)
    cerebro.adddata(data)
    
    # Set broker parameters
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Add analyzers for detailed performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run backtest
    print(f"\n⚡ Running backtest...")
    results = cerebro.run()
    
    # Extract analyzer results
    strat = results[0]
    
    print(f"\n📈 ANALYZER RESULTS:")
    print(f"   • Sharpe Ratio: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')}")
    
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"   • Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    
    returns = strat.analyzers.returns.get_analysis()
    print(f"   • Total Return: {returns.get('rtot', 0)*100:.2f}%")
    
    # Plot results
    if plot:
        print(f"\n📊 Generating plots...")
        try:
            # Try to display the plot
            cerebro.plot(style='candlestick', 
                        barup='green', 
                        bardown='red',
                        volume=False,  # Hide volume for cleaner view
                        figsize=(16, 10),  # Larger figure
                        dpi=100,  # Higher resolution
                        numfigs=1)  # Single figure
            print("✅ Plot displayed successfully!")
        except Exception as e:
            print(f"⚠️ Could not display plot: {e}")
            print("💡 Try running in a Jupyter notebook or check matplotlib backend")
    
    return results


def run_all_symbols(data_dir="../data/", initial_cash=100000):
    """Run backtest on all merged crypto datasets"""
    
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("merged_*_4h_with_multi_tf_rsi.csv"))
    
    if not csv_files:
        print("❌ No merged CSV files found! Run merge_crypto_data.py first.")
        return
    
    print(f"\n🔍 Found {len(csv_files)} datasets to backtest")
    
    all_results = []
    
    for csv_file in csv_files[:2]:  # Test with first 2 files initially
        symbol = csv_file.stem.replace('merged_', '').replace('_4h_with_multi_tf_rsi', '')
        
        print(f"\n{'='*60}")
        print(f"📊 TESTING: {symbol.replace('_', ' ').upper()}")
        print(f"{'='*60}")
        
        try:
            results = run_conservative_rsi_backtest(csv_file, initial_cash, plot=False)
            all_results.append({
                'symbol': symbol,
                'results': results[0] if results else None
            })
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    return all_results


def run_single_test():
    """Run test on single dataset"""
    test_file = "../data/merged_binance_btcusdt_4h_with_multi_tf_rsi.csv"
    
    if Path(test_file).exists():
        results = run_conservative_rsi_backtest(test_file, initial_cash=100000, plot=True)
        return results
    else:
        print(f"❌ Test file {test_file} not found!")
        print("Available files:")
        for f in Path(".").glob("merged_*.csv"):
            print(f"  • {f.name}")
        return None


if __name__ == "__main__":
    # Run single test
    results = run_single_test()