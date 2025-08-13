#!/usr/bin/env python3
"""
Simplified Strategy Plotter

Focus on generating and displaying the plot with executed trades
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for display
import matplotlib.pyplot as plt
import backtrader as bt
import pandas as pd
from pathlib import Path
import sys
sys.path.append('..')
from strategies.conservative_rsi_strategy import ConservativeRSIStrategy, CryptoRSIData

def plot_strategy_results():
    """Plot the strategy results with executed trades"""
    
    print("🎨 Generating Strategy Plot...")
    
    # Load data
    csv_file = "../data/merged_binance_btcusdt_4h_with_multi_tf_rsi.csv"
    if not Path(csv_file).exists():
        print(f"❌ Data file {csv_file} not found!")
        return
    
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.dropna(subset=['RSI_4h', 'RSI_12h', 'RSI_1d', 'RSI_3d'])
    
    print(f"📊 Loaded {len(df)} rows of data")
    
    # Initialize backtrader
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(ConservativeRSIStrategy, verbose=False)
    
    # Add data feed
    data = CryptoRSIData(dataname=df)
    cerebro.adddata(data)
    
    # Set broker parameters
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    
    # Run backtest
    print("⚡ Running backtest...")
    results = cerebro.run()
    
    # Plot results
    print("📊 Generating plot...")
    try:
        cerebro.plot(style='candlestick', 
                    barup='green', 
                    bardown='red',
                    volume=False,
                    figsize=(16, 10),
                    dpi=100,
                    numfigs=1)
        print("✅ Plot generated successfully!")
    except Exception as e:
        print(f"⚠️ Plot error: {e}")
        print("💡 Try running in Jupyter notebook or check matplotlib backend")

if __name__ == "__main__":
    plot_strategy_results()
