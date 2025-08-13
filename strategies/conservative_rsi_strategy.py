#!/usr/bin/env python3
"""
Conservative Multi-Timeframe RSI Strategy

Entry: ALL RSI timeframes (4h, 12h, 1d) must be ≤ 30 (oversold)
Exit: 4h RSI ≥ 70 (overbought)

This is a high-conviction strategy requiring confluence across all timeframes.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path


class ConservativeRSIStrategy(bt.Strategy):
    """
    Conservative RSI Strategy:
    - Buy: ALL RSI timeframes (4h, 12h, 1d) ≤ 30 (maximum confluence)
    - Sell: 4h RSI ≥ 70 (take profits on strength)
    """
    
    params = (
        # RSI Thresholds
        ('entry_threshold', 30),       # All RSI must be ≤ 30
        ('exit_threshold', 60),        # 4h RSI ≥ 70 to exit
        
        # Position Management
        ('position_pct', 0.95),        # Use 95% of available cash
        
        # Strategy Settings
        ('min_price', 1.0),            # Minimum price to avoid penny stocks
        ('verbose', True),             # Enable detailed logging
    )
    
    def __init__(self):
        """Initialize the conservative RSI strategy"""
        
        # Data references
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.volume = self.datas[0].volume
        
        # Multi-timeframe RSI indicators
        self.rsi_4h = self.datas[0].RSI_4h
        self.rsi_12h = self.datas[0].RSI_12h  
        self.rsi_1d = self.datas[0].RSI_1d
        self.rsi_3d = self.datas[0].RSI_3d
        
        # Order and position tracking
        self.order = None
        self.entry_bar = None
        self.entry_price = None
        self.entry_date = None
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        
        # Trade log for analysis
        self.trades = []
        
        self.log(f"Conservative RSI Strategy initialized")
        self.log(f"Entry: ALL RSI ≤ {self.params.entry_threshold}")
        self.log(f"Exit: 4h RSI ≥ {self.params.exit_threshold}")
    
    def log(self, txt, dt=None, force=False):
        """Logging function with timestamp"""
        if self.params.verbose or force:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.strftime("%Y-%m-%d %H:%M")} - {txt}')
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_bar = len(self)
                self.entry_price = order.executed.price
                self.entry_date = self.datas[0].datetime.datetime(0)
                
                self.log(f'🟢 BUY EXECUTED: ${order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.4f}, '
                        f'Cost: ${order.executed.value:.2f}')
                        
            elif order.issell():
                # Calculate trade performance
                if self.entry_price:
                    # Use absolute size since sell orders have negative size
                    trade_size = abs(order.executed.size)
                    pnl = (order.executed.price - self.entry_price) * trade_size
                    pnl_pct = (order.executed.price - self.entry_price) / self.entry_price * 100
                    holding_days = (self.datas[0].datetime.datetime(0) - self.entry_date).days
                    holding_bars = len(self) - self.entry_bar
                    
                    # Update performance metrics
                    self.total_pnl += pnl
                    if pnl > 0:
                        self.win_count += 1
                    self.trade_count += 1
                    
                    # Record trade details
                    trade_record = {
                        'entry_date': self.entry_date,
                        'exit_date': self.datas[0].datetime.datetime(0),
                        'entry_price': self.entry_price,
                        'exit_price': order.executed.price,
                        'holding_days': holding_days,
                        'holding_bars': holding_bars,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': getattr(self, 'exit_reason', 'Unknown')
                    }
                    self.trades.append(trade_record)
                    
                    self.log(f'🔴 SELL EXECUTED: ${order.executed.price:.2f}, '
                            f'P&L: ${pnl:.2f} ({pnl_pct:+.1f}%), '
                            f'Held: {holding_days} days ({holding_bars} bars)')
                    
                    # Reset position tracking
                    self.entry_bar = None
                    self.entry_price = None
                    self.entry_date = None
                    self.exit_reason = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'⚠️ Order {order.status}: {getattr(order, "info", "")}')
        
        self.order = None
    
    def get_rsi_values(self):
        """Get current RSI values for all timeframes, handling NaN"""
        return {
            '4h': self.rsi_4h[0],
            '12h': self.rsi_12h[0], 
            '1d': self.rsi_1d[0]
        }
    
    def check_entry_conditions(self):
        """Check if all entry conditions are met"""
        rsi_values = self.get_rsi_values()
        
        # Price filter
        if self.dataclose[0] < self.params.min_price:
            return False, f"Price too low: ${self.dataclose[0]:.2f}"
        
        # Check for NaN values - skip if any RSI is NaN
        if any(np.isnan(rsi) for rsi in rsi_values.values()):
            return False, "Skipping bar with NaN RSI values"
        
        # Check if ALL RSI timeframes are oversold
        all_oversold = all(rsi <= self.params.entry_threshold for rsi in rsi_values.values())
        
        if all_oversold:
            rsi_str = f"4h:{rsi_values['4h']:.1f}, 12h:{rsi_values['12h']:.1f}, 1d:{rsi_values['1d']:.1f}"
            return True, f"ALL RSI oversold - {rsi_str}"
        else:
            # Find which timeframes are not oversold
            not_oversold = [tf for tf, rsi in rsi_values.items() if rsi > self.params.entry_threshold]
            return False, f"Not all oversold. Above threshold: {not_oversold}"
    
    def check_exit_conditions(self):
        """Check if exit conditions are met"""
        if not self.position.size:
            return False, ""
        
        rsi_values = self.get_rsi_values()
        
        # Check for NaN values - skip if 4h RSI is NaN
        if np.isnan(rsi_values['4h']):
            return False, "Skipping exit check - 4h RSI is NaN"
        
        # Only exit condition: 4h RSI overbought
        if rsi_values['4h'] >= self.params.exit_threshold:
            return True, f"4h RSI overbought ({rsi_values['4h']:.1f})"
        
        return False, ""
    
    def next(self):
        """Main strategy logic"""
        
        # Update performance tracking
        current_value = self.broker.get_value()
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        current_dd = (self.peak_value - current_value) / self.peak_value
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Skip if we have a pending order
        if self.order:
            return
        
        current_position = self.position.size
        
        # Debug logging for position management
        if len(self) % 1000 == 0:  # Log every 1000 bars
            self.log(f'🔍 DEBUG: Bar {len(self)}, Position size: {current_position}, Cash: ${self.broker.get_cash():.2f}')
        
        # Only enter if we have no position (not short, not long)
        if current_position == 0:
            # ENTRY LOGIC
            can_buy, reason = self.check_entry_conditions()
            
            # Debug logging for entry attempts
            if len(self) % 1000 == 0:  # Log every 1000 bars
                self.log(f'🔍 DEBUG: Entry check - can_buy: {can_buy}, reason: {reason}')
            
            if can_buy:
                # Calculate position size
                available_cash = self.broker.get_cash() * self.params.position_pct
                size = available_cash / self.dataclose[0]
                
                # Debug position sizing
                self.log(f'🔍 DEBUG: Available cash: ${available_cash:.2f}, Price: ${self.dataclose[0]:.2f}, Calculated size: {size:.4f}')
                
                # Ensure positive position size
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f'📈 BUY ORDER: {reason}')
                    self.log(f'   Price: ${self.dataclose[0]:.2f}, Size: {size:.4f}')
                else:
                    self.log(f'⚠️ Invalid position size: {size:.4f}')
        
        else:
            # EXIT LOGIC
            should_exit, reason = self.check_exit_conditions()
            
            if should_exit:
                self.exit_reason = reason
                # Use sell() for proper plotting, but ensure we're closing a long position
                if self.position.size > 0:
                    self.order = self.sell(size=self.position.size)
                    self.log(f'📉 SELL ORDER: {reason}')
                else:
                    self.log(f'⚠️ Cannot sell - no long position to close')
    
    def stop(self):
        """Strategy completion - print comprehensive results"""
        
        # Force close any remaining position
        if self.position.size != 0:
            if self.position.size > 0:
                self.close()  # Close long position
            else:
                self.buy(size=abs(self.position.size))  # Close short position
            self.log(f'🔒 FORCED POSITION CLOSE: Size was {self.position.size}')
        
        final_value = self.broker.get_value()
        initial_cash = 100000  # Assuming default
        total_return = (final_value - initial_cash) / initial_cash * 100
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        
        print(f'\n{"="*60}')
        print(f'🏁 CONSERVATIVE RSI STRATEGY RESULTS')
        print(f'{"="*60}')
        print(f'💰 Final Portfolio Value: ${final_value:,.2f}')
        print(f'📊 Total Return: {total_return:+.2f}%')
        print(f'📈 Total P&L: ${self.total_pnl:,.2f}')
        print(f'🎯 Total Trades: {self.trade_count}')
        print(f'✅ Winning Trades: {self.win_count}')
        print(f'❌ Losing Trades: {self.trade_count - self.win_count}')
        print(f'📋 Win Rate: {win_rate:.1f}%')
        print(f'📉 Maximum Drawdown: {self.max_drawdown*100:.2f}%')
        
        if self.trade_count > 0:
            avg_pnl = self.total_pnl / self.trade_count
            print(f'💵 Average P&L per Trade: ${avg_pnl:.2f}')
            
            # Calculate additional metrics from trade records
            if self.trades:
                pnl_pcts = [trade['pnl_pct'] for trade in self.trades]
                holding_days = [trade['holding_days'] for trade in self.trades]
                
                print(f'📊 Average Return per Trade: {np.mean(pnl_pcts):+.2f}%')
                print(f'📅 Average Holding Period: {np.mean(holding_days):.1f} days')
                print(f'🏆 Best Trade: {max(pnl_pcts):+.2f}%')
                print(f'😞 Worst Trade: {min(pnl_pcts):+.2f}%')
        
        print(f'{"="*60}')
        
        # Save detailed results
        if self.trades:
            results_df = pd.DataFrame(self.trades)
            results_filename = f'conservative_rsi_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.csv'
            results_df.to_csv(results_filename, index=False)
            self.log(f'💾 Detailed results saved: {results_filename}', force=True)


# Data feed class for crypto RSI data
class CryptoRSIData(bt.feeds.PandasData):
    """Custom data feed for merged crypto data with multi-timeframe RSI"""
    
    lines = ('RSI_4h', 'RSI_12h', 'RSI_1d', 'RSI_3d')
    
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 'open'),
        ('high', 'high'), 
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('RSI_4h', 'RSI_4h'),
        ('RSI_12h', 'RSI_12h'),
        ('RSI_1d', 'RSI_1d'),
        ('RSI_3d', 'RSI_3d'),
    )