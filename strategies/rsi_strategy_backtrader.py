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
        # Signal Type - 'long' or 'short'
        ('signal_type', 'long'),              # 'long'=buy oversold/sell overbought, 'short'=sell overbought/buy oversold
        
        # Entry Strategy - Dictionary with column names and thresholds
        ('entry_rsi_rules', {
            'RSI_1H': 30,    # 1H RSI must be ≤ 30 (long) or ≥ 70 (short)
            'RSI_4H': 30,    # 4H RSI must be ≤ 30 (long) or ≥ 70 (short)
            'RSI_12H': 30,   # 12H RSI must be ≤ 30 (long) or ≥ 70 (short)
            'RSI_1D': 30     # 1D RSI must be ≤ 30 (long) or ≥ 70 (short)
        }),
        ('entry_mode', 'all'),                # 'all'=all rules must pass, 'any'=any rule passes, 'majority'=majority pass, 'count'=minimum count
        ('min_required_count', 3),            # Minimum rules that must pass (if entry_mode='count')
        
        # Exit Strategy - Dictionary with column names and thresholds
        ('exit_rsi_rules', {
            'RSI_4H': 70,    # Exit when 4H RSI ≥ 70 (long) or ≤ 30 (short)
        }),
        ('exit_mode', 'any'),                 # 'any'=any exit rule triggers exit, 'all'=all exit rules must trigger
        ('enable_profit_target', False),      # Enable profit target exit
        ('profit_target_pct', 0.20),          # 20% profit target
        ('enable_stop_loss', False),          # Enable stop loss exit  
        ('stop_loss_pct', -0.10),             # 10% stop loss
        
        # Position Management
        ('position_pct', 0.95),               # Use 95% of available cash
        
        # Strategy Settings
        ('min_price', 1000.0),                # Minimum BTC price
        ('enable_trend_filter', True),        # Enable SMA trend alignment filter
        ('enable_bb_filter', True),           # Enable Bollinger Band filter
        ('bb_percent_threshold', 0.8),        # BB Percent threshold for short entries
        ('verbose', True),                    # Enable detailed logging
    )
    
    def __init__(self):
        """Initialize the conservative RSI strategy"""
        
        # Data references
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.volume = self.datas[0].volume
        
        # Multi-timeframe RSI indicators (updated column names)
        self.rsi_1h = self.datas[0].RSI_1H
        self.rsi_4h = self.datas[0].RSI_4H
        self.rsi_12h = self.datas[0].RSI_12H  
        self.rsi_1d = self.datas[0].RSI_1D
        
        
        # Create mapping for dynamic access
        self.rsi_indicators = {
            'RSI_1H': self.rsi_1h,
            'RSI_4H': self.rsi_4h,
            'RSI_12H': self.rsi_12h,
            'RSI_1D': self.rsi_1d
        }
        
        # Bollinger Band indicators
        self.bb_percent = self.datas[0].BB_Percent_1H
        
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
        
        self.log(f"Flexible RSI Strategy initialized ({self.params.signal_type.upper()})")
        self.log(f"Signal type: {self.params.signal_type}")
        self.log(f"Entry rules: {self.params.entry_rsi_rules}")
        self.log(f"Entry mode: {self.params.entry_mode}")
        self.log(f"Exit rules: {self.params.exit_rsi_rules}")
        self.log(f"Exit mode: {self.params.exit_mode}")
    
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
                # Buy order executed - could be long entry or short cover
                if self.position.size > 0:
                    # This is a long entry (position became positive)
                    self.entry_bar = len(self)
                    self.entry_price = order.executed.price
                    self.entry_date = self.datas[0].datetime.datetime(0)
                    
                    self.log(f'🟢 LONG ENTRY EXECUTED: ${order.executed.price:.2f}, '
                            f'Size: {order.executed.size:.4f}, '
                            f'Cost: ${order.executed.value:.2f}')
                else:
                    # This is covering a short position (short exit)
                    self.calculate_and_log_trade_performance(order, 'SHORT')
                        
            elif order.issell():
                # Sell order executed - could be short entry or long exit
                if self.position.size < 0:
                    # This is a short entry (position became negative)
                    self.entry_bar = len(self)
                    self.entry_price = order.executed.price
                    self.entry_date = self.datas[0].datetime.datetime(0)
                    
                    self.log(f'🔴 SHORT ENTRY EXECUTED: ${order.executed.price:.2f}, '
                            f'Size: {abs(order.executed.size):.4f}, '
                            f'Value: ${abs(order.executed.value):.2f}')
                else:
                    # This is closing a long position (long exit)
                    self.calculate_and_log_trade_performance(order, 'LONG')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'⚠️ Order {order.status}: {getattr(order, "info", "")}')
        
        self.order = None
    
    def get_rsi_values(self):
        """Get current RSI values for all timeframes, handling NaN"""
        rsi_values = {}
        for column_name, indicator in self.rsi_indicators.items():
            try:
                rsi_values[column_name] = indicator[0]
            except (IndexError, AttributeError):
                rsi_values[column_name] = np.nan
        return rsi_values
    
    def calculate_and_log_trade_performance(self, order, trade_type):
        """Calculate and log trade performance for both long and short trades"""
        if self.entry_price:
            # Use absolute size since sell orders have negative size
            trade_size = abs(order.executed.size)
            
            # Calculate P&L based on trade type
            if trade_type == 'LONG':
                # Long trade: profit when exit price > entry price
                pnl = (order.executed.price - self.entry_price) * trade_size
                pnl_pct = (order.executed.price - self.entry_price) / self.entry_price * 100
                exit_emoji = '🔴'
                exit_action = 'LONG EXIT'
            else:  # SHORT
                # Short trade: profit when entry price > exit price
                pnl = (self.entry_price - order.executed.price) * trade_size
                pnl_pct = (self.entry_price - order.executed.price) / self.entry_price * 100
                exit_emoji = '🟢'
                exit_action = 'SHORT COVER'
            
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
                'trade_type': trade_type,
                'exit_reason': getattr(self, 'exit_reason', 'Unknown')
            }
            self.trades.append(trade_record)
            
            self.log(f'{exit_emoji} {exit_action} EXECUTED: ${order.executed.price:.2f}, '
                    f'P&L: ${pnl:.2f} ({pnl_pct:+.1f}%), '
                    f'Held: {holding_days} days ({holding_bars} bars)')
            
            # Reset position tracking
            self.entry_bar = None
            self.entry_price = None
            self.entry_date = None
            self.exit_reason = None
    
    def check_trend_alignment(self):
        """
        Check if close price and SMAs are in proper trend alignment
        Returns: (is_aligned, trend_direction, reason)
        - is_aligned: True if aligned (ascending or descending), False if mixed
        - trend_direction: 'ascending', 'descending', or 'mixed'
        - reason: Explanation string
        """
        try:
            # Get current values
            close = self.dataclose[0]
            sma_14 = self.sma_14[0]
            sma_21 = self.sma_21[0]
            sma_35 = self.sma_35[0]
            
            # Check if any SMA is NaN
            if np.isnan(sma_14) or np.isnan(sma_21) or np.isnan(sma_35):
                return False, 'insufficient_data', "SMA data not available"
            
            # Create ordered list for comparison
            values = [close, sma_14, sma_21, sma_35]
            
            # Check if ascending (bullish alignment): close > sma_14 > sma_21 > sma_35
            is_ascending = all(values[i] >= values[i+1] for i in range(len(values)-1))
            
            # Check if descending (bearish alignment): close < sma_14 < sma_21 < sma_35  
            is_descending = all(values[i] <= values[i+1] for i in range(len(values)-1))
            
            if is_ascending:
                reason = f"Bullish alignment: Close({close:.2f}) > SMA14({sma_14:.2f}) > SMA21({sma_21:.2f}) > SMA35({sma_35:.2f})"
                return True, 'ascending', reason
            elif is_descending:
                reason = f"Bearish alignment: Close({close:.2f}) < SMA14({sma_14:.2f}) < SMA21({sma_21:.2f}) < SMA35({sma_35:.2f})"
                return True, 'descending', reason
            else:
                reason = f"Mixed alignment: Close({close:.2f}), SMA14({sma_14:.2f}), SMA21({sma_21:.2f}), SMA35({sma_35:.2f})"
                return False, 'mixed', reason
                
        except (IndexError, AttributeError) as e:
            return False, 'error', f"Error accessing SMA data: {e}"
    
    def check_bb_filter(self):
        """
        Check Bollinger Band filter for entry conditions
        Returns: (passes_filter, reason)
        """
        try:
            bb_percent_value = self.bb_percent[0]
            
            # Check if BB Percent data is available
            if np.isnan(bb_percent_value):
                return False, "BB Percent data not available"
            
            # Apply filter based on signal type
            if self.params.signal_type == 'short':
                # For short signals: only enter when BB% <= threshold (not too extended above upper band)
                if bb_percent_value <= self.params.bb_percent_threshold:
                    return True, f"BB filter passed: BB%({bb_percent_value:.3f}) <= {self.params.bb_percent_threshold}"
                else:
                    return False, f"BB filter failed: BB%({bb_percent_value:.3f}) > {self.params.bb_percent_threshold} (too extended)"
            else:
                # For long signals: could add reverse logic (BB% >= 1-threshold), but not implemented yet
                # For now, long signals pass BB filter
                return True, f"BB filter passed for long: BB%({bb_percent_value:.3f})"
                
        except (IndexError, AttributeError) as e:
            return False, f"Error accessing BB data: {e}"
    
    def check_entry_conditions(self):
        """Check if entry conditions are met based on flexible RSI rules"""
        rsi_values = self.get_rsi_values()
        
        # Price filter
        if self.dataclose[0] < self.params.min_price:
            return False, f"Price too low: ${self.dataclose[0]:.2f}"
        
        # Bollinger Band filter
        if self.params.enable_bb_filter:
            bb_passes, bb_reason = self.check_bb_filter()
            if not bb_passes:
                return False, f"BB filter rejected: {bb_reason}"
        
        # Evaluate entry rules
        rule_results = []
        rule_details = []
        
        for column, threshold in self.params.entry_rsi_rules.items():
            if column not in rsi_values or np.isnan(rsi_values[column]):
                # Skip rules with missing data
                continue
                
            current_rsi = rsi_values[column]
            
            # Apply logic based on signal type
            if self.params.signal_type == 'long':
                # Long: buy when RSI is oversold (≤ threshold)
                rule_passed = current_rsi <= threshold
                rule_details.append(f"{column}:{current_rsi:.1f}(≤{threshold})")
            else:  # short
                # Short: sell when RSI is overbought (≥ threshold)
                rule_passed = current_rsi >= threshold
                rule_details.append(f"{column}:{current_rsi:.1f}(≥{threshold})")
                
            rule_results.append(rule_passed)
        
        if not rule_results:
            return False, "No valid RSI data for entry rules"
        
        # Apply entry mode logic
        if self.params.entry_mode == 'all':
            entry_signal = all(rule_results)
            reason = f"ALL rules pass - {', '.join(rule_details)}" if entry_signal else f"Not all rules pass - {', '.join(rule_details)}"
        elif self.params.entry_mode == 'any':
            entry_signal = any(rule_results)
            reason = f"At least one rule passes - {', '.join(rule_details)}" if entry_signal else f"No rules pass - {', '.join(rule_details)}"
        elif self.params.entry_mode == 'majority':
            entry_signal = sum(rule_results) > len(rule_results) / 2
            reason = f"Majority pass ({sum(rule_results)}/{len(rule_results)}) - {', '.join(rule_details)}"
        elif self.params.entry_mode == 'count':
            entry_signal = sum(rule_results) >= self.params.min_required_count
            reason = f"Required count met ({sum(rule_results)}>={self.params.min_required_count}) - {', '.join(rule_details)}"
        else:
            return False, f"Unknown entry_mode: {self.params.entry_mode}"
        
        return entry_signal, reason
    
    def check_exit_conditions(self):
        """Check if exit conditions are met based on flexible RSI rules and other exits"""
        if not self.position.size:
            return False, ""
        
        rsi_values = self.get_rsi_values()
        
        # Check RSI-based exit rules
        rsi_exit_triggered = False
        rsi_exit_reason = ""
        
        if self.params.exit_rsi_rules:
            rule_results = []
            rule_details = []
            
            for column, threshold in self.params.exit_rsi_rules.items():
                if column not in rsi_values or np.isnan(rsi_values[column]):
                    continue
                    
                current_rsi = rsi_values[column]
                
                # Apply logic based on signal type and position
                if self.params.signal_type == 'long' and self.position.size > 0:
                    # Long exit: sell when RSI is overbought (≥ threshold)
                    rule_passed = current_rsi >= threshold
                    rule_details.append(f"{column}:{current_rsi:.1f}(≥{threshold})")
                elif self.params.signal_type == 'short' and self.position.size < 0:
                    # Short exit: cover when RSI is oversold (≤ threshold)
                    rule_passed = current_rsi <= threshold
                    rule_details.append(f"{column}:{current_rsi:.1f}(≤{threshold})")
                else:
                    # No matching position type or signal type
                    continue
                    
                rule_results.append(rule_passed)
            
            if rule_results:
                if self.params.exit_mode == 'any':
                    rsi_exit_triggered = any(rule_results)
                elif self.params.exit_mode == 'all':
                    rsi_exit_triggered = all(rule_results)
                
                if rsi_exit_triggered:
                    rsi_exit_reason = f"RSI exit - {', '.join(rule_details)}"
        
        # Check profit target exit
        if self.params.enable_profit_target and self.entry_price:
            if self.position.size > 0:  # Long position
                pnl_pct = (self.dataclose[0] - self.entry_price) / self.entry_price
            else:  # Short position
                pnl_pct = (self.entry_price - self.dataclose[0]) / self.entry_price
            
            if pnl_pct >= self.params.profit_target_pct:
                return True, f"Profit target hit: {pnl_pct*100:.2f}%"
        
        # Check stop loss exit
        if self.params.enable_stop_loss and self.entry_price:
            if self.position.size > 0:  # Long position
                pnl_pct = (self.dataclose[0] - self.entry_price) / self.entry_price
            else:  # Short position
                pnl_pct = (self.entry_price - self.dataclose[0]) / self.entry_price
            
            if pnl_pct <= self.params.stop_loss_pct:
                return True, f"Stop loss hit: {pnl_pct*100:.2f}%"
        
        # Return RSI exit result
        return rsi_exit_triggered, rsi_exit_reason
    
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
            can_enter, reason = self.check_entry_conditions()
            
            # Debug logging for entry attempts
            if len(self) % 1000 == 0:  # Log every 1000 bars
                self.log(f'🔍 DEBUG: Entry check - can_enter: {can_enter}, reason: {reason}')
            
            if can_enter:
                # Calculate position size
                available_cash = self.broker.get_cash() * self.params.position_pct
                size = available_cash / self.dataclose[0]
                
                # Debug position sizing
                self.log(f'🔍 DEBUG: Available cash: ${available_cash:.2f}, Price: ${self.dataclose[0]:.2f}, Calculated size: {size:.4f}')
                
                # Ensure positive position size
                if size > 0:
                    if self.params.signal_type == 'long':
                        # Long signal: buy when oversold
                        self.order = self.buy(size=size)
                        self.log(f'📈 LONG ENTRY: {reason}')
                        self.log(f'   Price: ${self.dataclose[0]:.2f}, Size: {size:.4f}')
                    else:  # short
                        # Short signal: sell when overbought
                        self.order = self.sell(size=size)
                        self.log(f'📉 SHORT ENTRY: {reason}')
                        self.log(f'   Price: ${self.dataclose[0]:.2f}, Size: {size:.4f}')
                else:
                    self.log(f'⚠️ Invalid position size: {size:.4f}')
        
        else:
            # EXIT LOGIC
            should_exit, reason = self.check_exit_conditions()
            
            if should_exit:
                self.exit_reason = reason
                
                if self.position.size > 0:
                    # Close long position by selling
                    self.order = self.sell(size=self.position.size)
                    self.log(f'📉 LONG EXIT: {reason}')
                elif self.position.size < 0:
                    # Close short position by buying to cover
                    self.order = self.buy(size=abs(self.position.size))
                    self.log(f'📈 SHORT COVER: {reason}')
                else:
                    self.log(f'⚠️ No position to close')
    
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
        print(f'🏁 FLEXIBLE RSI STRATEGY RESULTS')
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
            from pathlib import Path
            
            # Create results directory if it doesn't exist
            results_dir = Path("../results")
            results_dir.mkdir(exist_ok=True)
            
            results_df = pd.DataFrame(self.trades)
            results_filename = results_dir / f'flexible_rsi_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.csv'
            results_df.to_csv(results_filename, index=False)
            self.log(f'💾 Detailed results saved: {results_filename}', force=True)


# Data feed class for smooth multi-timeframe RSI data
class SmoothRSIData(bt.feeds.PandasData):
    """Custom data feed for smooth multi-timeframe RSI data and Bollinger Bands"""
    
    lines = ('RSI_1H', 'RSI_4H', 'RSI_12H', 'RSI_1D', 'BB_Percent_1H')
    
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 'open'),
        ('high', 'high'), 
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('RSI_1H', 'RSI_1H'),
        ('RSI_4H', 'RSI_4H'),
        ('RSI_12H', 'RSI_12H'),
        ('RSI_1D', 'RSI_1D'),
        ('BB_Percent_1H', 'BB_Percent_1H'),
    )