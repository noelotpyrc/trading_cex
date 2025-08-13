#!/usr/bin/env python3
"""
Debug script for Conservative RSI Strategy
"""

import pandas as pd
import numpy as np

def debug_strategy_logic():
    """Debug the strategy logic step by step"""
    
    print("🔍 DEBUGGING CONSERVATIVE RSI STRATEGY")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('../data/merged_binance_btcusdt_4h_with_multi_tf_rsi.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    print(f"📊 Data loaded: {len(df)} rows")
    print(f"📅 Period: {df.index[0]} to {df.index[-1]}")
    
    # Test entry conditions
    entry_threshold = 30
    exit_threshold = 70
    
    # Find all entry points
    entry_conditions = (df['RSI_4h'] <= entry_threshold) & \
                      (df['RSI_12h'] <= entry_threshold) & \
                      (df['RSI_1d'] <= entry_threshold)
    
    entry_points = df[entry_conditions].copy()
    print(f"\n🎯 Entry conditions met: {len(entry_points)} times")
    
    if len(entry_points) > 0:
        print("\n📈 Sample entry points:")
        print(entry_points[['RSI_4h', 'RSI_12h', 'RSI_1d', 'close']].head(10))
        
        # Simulate a simple backtest
        print(f"\n🚀 SIMULATING SIMPLE BACKTEST")
        print("=" * 40)
        
        initial_cash = 100000
        current_cash = initial_cash
        position_size = 0
        entry_price = 0
        trades = []
        
        for i, (timestamp, row) in enumerate(entry_points.iterrows()):
            # Check if we can enter (no position)
            if position_size == 0:
                # Enter position
                entry_price = row['close']
                position_size = (current_cash * 0.95) / entry_price
                current_cash -= position_size * entry_price
                
                print(f"📈 BUY at {timestamp}: ${entry_price:.2f}, Size: {position_size:.4f}")
                
                # Look for exit
                future_data = df.loc[timestamp:].iloc[1:]  # Skip current bar
                exit_found = False
                
                for j, (exit_time, exit_row) in enumerate(future_data.iterrows()):
                    if exit_row['RSI_4h'] >= exit_threshold:
                        # Exit position
                        exit_price = exit_row['close']
                        pnl = (exit_price - entry_price) * position_size
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                        holding_bars = j + 1
                        
                        current_cash += position_size * exit_price
                        position_size = 0
                        
                        trades.append({
                            'entry_time': timestamp,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'holding_bars': holding_bars
                        })
                        
                        print(f"📉 SELL at {exit_time}: ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:+.1f}%), Bars: {holding_bars}")
                        exit_found = True
                        break
                
                if not exit_found:
                    print(f"⚠️ No exit found for trade at {timestamp}")
                    # Force exit at end of data
                    last_price = df.iloc[-1]['close']
                    pnl = (last_price - entry_price) * position_size
                    pnl_pct = (last_price - entry_price) / entry_price * 100
                    current_cash += position_size * last_price
                    position_size = 0
                    
                    trades.append({
                        'entry_time': timestamp,
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': last_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'holding_bars': 'end_of_data'
                    })
                    
                    print(f"📉 FORCED EXIT at end: ${last_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        
        # Final results
        print(f"\n🏁 SIMULATION RESULTS")
        print("=" * 30)
        print(f"💰 Initial cash: ${initial_cash:,.2f}")
        print(f"💰 Final cash: ${current_cash:,.2f}")
        print(f"📊 Total return: {((current_cash - initial_cash) / initial_cash * 100):+.2f}%")
        print(f"🎯 Total trades: {len(trades)}")
        
        if trades:
            pnls = [t['pnl'] for t in trades]
            pnl_pcts = [t['pnl_pct'] for t in trades]
            holding_bars = [t['holding_bars'] for t in trades if isinstance(t['holding_bars'], int)]
            
            print(f"✅ Winning trades: {sum(1 for pnl in pnls if pnl > 0)}")
            print(f"❌ Losing trades: {sum(1 for pnl in pnls if pnl < 0)}")
            print(f"📋 Win rate: {(sum(1 for pnl in pnls if pnl > 0) / len(trades) * 100):.1f}%")
            print(f"💵 Average P&L: ${np.mean(pnls):.2f}")
            print(f"📊 Average return: {np.mean(pnl_pcts):+.2f}%")
            if holding_bars:
                print(f"📅 Average holding bars: {np.mean(holding_bars):.1f}")
            
            print(f"\n📋 Trade details:")
            for i, trade in enumerate(trades):
                print(f"  {i+1}. {trade['entry_time']} → {trade['exit_time']}: ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.1f}%)")

if __name__ == "__main__":
    debug_strategy_logic()
