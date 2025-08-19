#!/usr/bin/env python3
"""
Conservative Multi-Timeframe RSI Strategy using Backtesting.py Framework

Features:
- Multi-timeframe RSI analysis (1H, 4H, 12H, 1D)
- Bollinger Band percent filter
- Support for both long and short signals
- Flexible entry/exit modes (all, any, majority)
- Built-in optimization capabilities
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from pathlib import Path

# Custom indicator functions (since pandas_ta has compatibility issues)
def calculate_bb_percent(close, length=20, std=2.0):
    """Calculate Bollinger Band percent"""
    sma = close.rolling(length).mean()
    rolling_std = close.rolling(length).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    bb_percent = (close - lower) / (upper - lower)
    return bb_percent


class ConservativeRSIStrategy(Strategy):
    """
    Conservative RSI Strategy with Multi-Timeframe Analysis
    
    Parameters (optimizable):
    - signal_type: 'long' or 'short'
    - entry_mode: 'all', 'any', 'majority', 'count'
    - rsi thresholds for different timeframes
    - bollinger band filters
    """
    
    # === SIGNAL TYPE ===
    signal_type = 'long'  # 'long' or 'short'
    
    # === ENTRY RSI THRESHOLDS ===
    rsi_1h_threshold = 30
    rsi_4h_threshold = 30
    rsi_12h_threshold = 30
    rsi_1d_threshold = 30
    
    # === ENTRY MODE ===
    entry_mode = 'all'  # 'all', 'any', 'majority', 'count'
    min_required_count = 3  # for 'count' mode
    
    # === EXIT RSI THRESHOLDS ===
    exit_1h_threshold = None
    exit_4h_threshold = None
    exit_12h_threshold = None
    exit_1d_threshold = None
    enabled_exit_timeframes = ['4H']  # Default to 4H if not specified
    
    # === BOLLINGER BAND FILTER ===
    enable_bb_filter = True
    bb_percent_threshold = 0.8  # For short signals
    
    # === PROFIT/LOSS EXITS ===
    enable_profit_target = True
    profit_target_pct = 0.15
    enable_stop_loss = True
    stop_loss_pct = 0.01
    
    # === POSITION MANAGEMENT ===
    min_price = 1000.0
    
    def init(self):
        """Initialize indicators and setup strategy"""
        
        # Ensure required columns exist
        required_columns = ['RSI_1H', 'RSI_4H', 'RSI_12H', 'RSI_1D']
        for col in required_columns:
            if not hasattr(self.data, col):
                raise ValueError(f"Data missing required column: {col}")
        
        # RSI indicators (wrap with self.I for proper array handling)
        self.rsi_1h = self.I(lambda x: x, self.data.RSI_1H)
        self.rsi_4h = self.I(lambda x: x, self.data.RSI_4H)
        self.rsi_12h = self.I(lambda x: x, self.data.RSI_12H)
        self.rsi_1d = self.I(lambda x: x, self.data.RSI_1D)
        
        # Bollinger Band percent (if available)
        if hasattr(self.data, 'BB_Percent_1H'):
            self.bb_percent = self.I(lambda x: x, self.data.BB_Percent_1H)
        else:
            # Calculate BB Percent if not in data
            self.bb_percent = self.I(calculate_bb_percent, self.data.Close, 20, 2.0)
        
        # Track entry price for profit/loss exits
        self.entry_price = None
        
        # Statistics tracking
        self.trade_count = 0
        self.win_count = 0
        
        print(f"Strategy initialized: {self.signal_type.upper()} mode")
        print(f"Entry thresholds: 1H={self.rsi_1h_threshold}, 4H={self.rsi_4h_threshold}, "
              f"12H={self.rsi_12h_threshold}, 1D={self.rsi_1d_threshold}")
        print(f"Entry mode: {self.entry_mode}")
        print(f"BB filter: {self.enable_bb_filter} (threshold: {self.bb_percent_threshold})")

    def check_price_filter(self):
        """Check if price meets minimum requirements"""
        return self.data.Close[-1] >= self.min_price

    def check_bb_filter(self):
        """Check Bollinger Band filter for entries"""
        if not self.enable_bb_filter:
            return True, "BB filter disabled"
        
        if not hasattr(self, 'bb_percent'):
            return False, "BB data not available"
        
        try:
            bb_value = self.bb_percent[-1]
        except (IndexError, AttributeError):
            return False, "BB data not accessible"
            
        if np.isnan(bb_value):
            return False, "BB data is NaN"
        
        if self.signal_type == 'short':
            # For short: only enter when not too extended above upper band
            if bb_value <= self.bb_percent_threshold:
                return True, f"BB filter passed: {bb_value:.3f} <= {self.bb_percent_threshold}"
            else:
                return False, f"BB filter failed: {bb_value:.3f} > {self.bb_percent_threshold}"
        else:
            # For long: could add reverse logic, but pass for now
            return True, f"BB filter passed for long: {bb_value:.3f}"

    def check_rsi_entry_conditions(self):
        """Check RSI entry conditions based on signal type and mode"""
        
        # Get current RSI values (ensure proper array access)
        try:
            rsi_values = {
                'RSI_1H': self.rsi_1h[-1],
                'RSI_4H': self.rsi_4h[-1], 
                'RSI_12H': self.rsi_12h[-1],
                'RSI_1D': self.rsi_1d[-1]
            }
        except (IndexError, AttributeError):
            return False, "RSI data not accessible"
        
        # Define thresholds
        thresholds = {
            'RSI_1H': self.rsi_1h_threshold,
            'RSI_4H': self.rsi_4h_threshold,
            'RSI_12H': self.rsi_12h_threshold,
            'RSI_1D': self.rsi_1d_threshold
        }
        
        # Check each RSI condition
        rule_results = []
        rule_details = []
        
        for timeframe, rsi_value in rsi_values.items():
            if np.isnan(rsi_value):
                continue  # Skip NaN values
            
            threshold = thresholds[timeframe]
            
            if self.signal_type == 'long':
                # Long: buy when RSI <= threshold (oversold)
                rule_passed = rsi_value <= threshold
                rule_details.append(f"{timeframe}:{rsi_value:.1f}(≤{threshold})")
            else:
                # Short: sell when RSI >= threshold (overbought) 
                rule_passed = rsi_value >= threshold
                rule_details.append(f"{timeframe}:{rsi_value:.1f}(≥{threshold})")
            
            rule_results.append(rule_passed)
        
        if not rule_results:
            return False, "No valid RSI data"
        
        # Apply entry mode logic
        if self.entry_mode == 'all':
            entry_signal = all(rule_results)
        elif self.entry_mode == 'any':
            entry_signal = any(rule_results)
        elif self.entry_mode == 'majority':
            entry_signal = sum(rule_results) > len(rule_results) / 2
        elif self.entry_mode == 'count':
            entry_signal = sum(rule_results) >= self.min_required_count
        else:
            return False, f"Unknown entry_mode: {self.entry_mode}"
        
        reason = f"{self.entry_mode.upper()} mode - {', '.join(rule_details)}"
        return entry_signal, reason

    def check_rsi_exit_conditions(self):
        """Check RSI exit conditions for only the enabled timeframes"""
        if not self.position:
            return False, ""
        
        # Check only the timeframes specified in enabled_exit_timeframes
        for timeframe in self.enabled_exit_timeframes:
            if timeframe == '1H' and self.exit_1h_threshold is not None:
                try:
                    rsi_value = self.rsi_1h[-1]
                    if np.isnan(rsi_value):
                        continue
                    
                    if self.signal_type == 'long' and self.position.size > 0:
                        if rsi_value >= self.exit_1h_threshold:
                            return True, f"RSI exit: 1H RSI {rsi_value:.1f} >= {self.exit_1h_threshold}"
                    elif self.signal_type == 'short' and self.position.size < 0:
                        if rsi_value <= self.exit_1h_threshold:
                            return True, f"RSI exit: 1H RSI {rsi_value:.1f} <= {self.exit_1h_threshold}"
                except (IndexError, AttributeError):
                    continue
            
            elif timeframe == '4H' and self.exit_4h_threshold is not None:
                try:
                    rsi_value = self.rsi_4h[-1]
                    if np.isnan(rsi_value):
                        continue
                    
                    if self.signal_type == 'long' and self.position.size > 0:
                        if rsi_value >= self.exit_4h_threshold:
                            return True, f"RSI exit: 4H RSI {rsi_value:.1f} >= {self.exit_4h_threshold}"
                    elif self.signal_type == 'short' and self.position.size < 0:
                        if rsi_value <= self.exit_4h_threshold:
                            return True, f"RSI exit: 4H RSI {rsi_value:.1f} <= {self.exit_4h_threshold}"
                except (IndexError, AttributeError):
                    continue
            
            elif timeframe == '12H' and self.exit_12h_threshold is not None:
                try:
                    rsi_value = self.rsi_12h[-1]
                    if np.isnan(rsi_value):
                        continue
                    
                    if self.signal_type == 'long' and self.position.size > 0:
                        if rsi_value >= self.exit_12h_threshold:
                            return True, f"RSI exit: 12H RSI {rsi_value:.1f} >= {self.exit_12h_threshold}"
                    elif self.signal_type == 'short' and self.position.size < 0:
                        if rsi_value <= self.exit_12h_threshold:
                            return True, f"RSI exit: 12H RSI {rsi_value:.1f} <= {self.exit_12h_threshold}"
                except (IndexError, AttributeError):
                    continue
            
            elif timeframe == '1D' and self.exit_1d_threshold is not None:
                try:
                    rsi_value = self.rsi_1d[-1]
                    if np.isnan(rsi_value):
                        continue
                    
                    if self.signal_type == 'long' and self.position.size > 0:
                        if rsi_value >= self.exit_1d_threshold:
                            return True, f"RSI exit: 1D RSI {rsi_value:.1f} >= {self.exit_1d_threshold}"
                    elif self.signal_type == 'short' and self.position.size < 0:
                        if rsi_value <= self.exit_1d_threshold:
                            return True, f"RSI exit: 1D RSI {rsi_value:.1f} <= {self.exit_1d_threshold}"
                except (IndexError, AttributeError):
                    continue
        
        return False, ""

    def check_profit_loss_exits(self):
        """Check profit target and stop loss exits"""
        if not self.position or self.entry_price is None:
            return False, ""
        
        current_price = self.data.Close[-1]
        
        # Calculate P&L based on position type
        if self.position.size > 0:  # Long position
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Check profit target
        if self.enable_profit_target and pnl_pct >= self.profit_target_pct:
            return True, f"Profit target hit: {pnl_pct*100:.2f}%"
        
        # Check stop loss
        if self.enable_stop_loss and pnl_pct <= -self.stop_loss_pct:
            return True, f"Stop loss hit: {pnl_pct*100:.2f}%"
        
        return False, ""

    def next(self):
        """Main strategy logic called for each bar"""
        
        # Debug output every 1000 bars
        bar_num = len(self.data)
        if bar_num % 1000 == 0:
            rsi_4h = self.rsi_4h[-1]
            bb_val = self.bb_percent[-1] if self.bb_percent else np.nan
            print(f"Bar {bar_num:4d}: RSI_4H={rsi_4h:5.1f}, BB%={bb_val:5.3f}, Price=${self.data.Close[-1]:8.2f}, Pos={self.position.size}")
        
        # Skip if price filter fails
        if not self.check_price_filter():
            return
        
        # ENTRY LOGIC
        if not self.position:
            # Check all entry conditions
            bb_pass, bb_reason = self.check_bb_filter()
            if not bb_pass:
                return
            
            rsi_signal, rsi_reason = self.check_rsi_entry_conditions()
            if not rsi_signal:
                return
            
            # All conditions met - enter position
            if self.signal_type == 'long':
                self.buy()
                self.entry_price = self.data.Close[-1]
                print(f"🟢 LONG ENTRY: Bar {bar_num} - {rsi_reason}")
            else:  # short
                self.sell()
                self.entry_price = self.data.Close[-1]
                print(f"🔴 SHORT ENTRY: Bar {bar_num} - {rsi_reason}")
        
        # EXIT LOGIC
        else:
            # Check RSI exit conditions
            rsi_exit, rsi_exit_reason = self.check_rsi_exit_conditions()
            if rsi_exit:
                self.position.close()
                self.entry_price = None
                print(f"🚪 RSI EXIT: Bar {bar_num} - {rsi_exit_reason}")
                return
            
            # Check profit/loss exits
            pl_exit, pl_reason = self.check_profit_loss_exits()
            if pl_exit:
                self.position.close()
                self.entry_price = None
                print(f"🎯 P&L EXIT: Bar {bar_num} - {pl_reason}")


def load_crypto_data(file_path):
    """Load and format crypto data for backtesting.py"""
    print(f"Loading data from: {file_path}")
    
    # Load CSV file
    df = pd.read_csv(file_path)
    
    # Set datetime index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Rename columns to match backtesting.py requirements (case-sensitive)
    df.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close'
    }, inplace=True)
    
    # Keep additional columns (RSI, BB indicators)
    print(f"Data shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def run_backtest(
    data_file="../data/binance_btcusdt.p_smooth_multi_rsi_corrected.csv",
    signal_type='long',
    entry_mode='all',
    rsi_thresholds={'1H': 30, '4H': 30, '12H': 30, '1D': 30},
    exit_1h_threshold=None,
    exit_4h_threshold=None,
    exit_12h_threshold=None,
    exit_1d_threshold=None,
    enabled_exit_timeframes=['4H'],
    enable_bb_filter=True,
    bb_percent_threshold=0.8,
    enable_profit_target=False,
    profit_target_pct=0.15,
    enable_stop_loss=False,
    stop_loss_pct=0.08,
    initial_cash=100000,
    commission=0.001,
    show_plot=True,
    plot_filename=None,
    **kwargs  # Allow additional parameters
):
    """Run backtest with specified parameters"""
    
    # Load data
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return None
    
    df = load_crypto_data(data_file)
    
    # Prepare strategy parameters for bt.run()
    strategy_params = {
        'signal_type': signal_type,
        'entry_mode': entry_mode,
        'rsi_1h_threshold': rsi_thresholds['1H'],
        'rsi_4h_threshold': rsi_thresholds['4H'], 
        'rsi_12h_threshold': rsi_thresholds['12H'],
        'rsi_1d_threshold': rsi_thresholds['1D'],
        'exit_1h_threshold': exit_1h_threshold,
        'exit_4h_threshold': exit_4h_threshold,
        'exit_12h_threshold': exit_12h_threshold,
        'exit_1d_threshold': exit_1d_threshold,
        'enabled_exit_timeframes': enabled_exit_timeframes,
        'enable_bb_filter': enable_bb_filter,
        'bb_percent_threshold': bb_percent_threshold,
        'enable_profit_target': enable_profit_target,
        'profit_target_pct': profit_target_pct,
        'enable_stop_loss': enable_stop_loss,
        'stop_loss_pct': stop_loss_pct
    }
    
    # Add any additional parameters from kwargs
    strategy_params.update(kwargs)
    
    print("\n🎯 BACKTESTING.PY RSI STRATEGY")
    print("=" * 50)
    print(f"💰 Initial Cash: ${initial_cash:,}")
    print(f"🎭 Signal Type: {signal_type.upper()}")
    print(f"📊 Entry Mode: {entry_mode.upper()}")
    print(f"📈 RSI Thresholds: {rsi_thresholds}")
    # Build exit info string 
    exit_info = []
    if exit_1h_threshold is not None and '1H' in enabled_exit_timeframes:
        exit_info.append(f"1H={exit_1h_threshold}")
    if exit_4h_threshold is not None and '4H' in enabled_exit_timeframes:
        exit_info.append(f"4H={exit_4h_threshold}")
    if exit_12h_threshold is not None and '12H' in enabled_exit_timeframes:
        exit_info.append(f"12H={exit_12h_threshold}")
    if exit_1d_threshold is not None and '1D' in enabled_exit_timeframes:
        exit_info.append(f"1D={exit_1d_threshold}")
    
    print(f"📉 Exit Thresholds: {', '.join(exit_info) if exit_info else 'None'}")
    print(f"🎛️  BB Filter: {enable_bb_filter} (threshold: {bb_percent_threshold})")
    print(f"🎯 Profit Target: {enable_profit_target} ({profit_target_pct*100:.1f}%)")
    print(f"🛑 Stop Loss: {enable_stop_loss} ({stop_loss_pct*100:.1f}%)")
    print("=" * 50)
    
    # Run backtest
    bt = Backtest(
        data=df,
        strategy=ConservativeRSIStrategy,  # Use base class
        cash=initial_cash,
        commission=commission,
        exclusive_orders=True
    )
    
    # Pass parameters directly to bt.run() - this properly overrides defaults!
    stats = bt.run(**strategy_params)
    
    # Display results
    print("\n📊 BACKTEST RESULTS:")
    print("=" * 30)
    print(stats)
    
    # Show plot if requested
    if show_plot:
        try:
            # Use provided filename or create a default one
            if plot_filename is None:
                plot_filename = f"{signal_type}_{entry_mode}_strategy.html"
            
            bt.plot(filename=plot_filename)
            print(f"✅ Plot saved to: {plot_filename}")
        except Exception as e:
            print(f"⚠️ Could not generate plot: {e}")
    
    return bt, stats


def optimize_strategy(
    data_file="../data/binance_btcusdt.p_smooth_multi_rsi_corrected.csv",
    signal_type='long',
    initial_cash=100000
):
    """Optimize strategy parameters"""
    
    print("🔧 OPTIMIZING STRATEGY PARAMETERS...")
    
    # Load data
    df = load_crypto_data(data_file)
    
    # Create base strategy
    class OptimizableStrategy(ConservativeRSIStrategy):
        signal_type = signal_type
    
    # Run optimization
    bt = Backtest(df, OptimizableStrategy, cash=initial_cash, commission=0.001)
    
    stats = bt.optimize(
        rsi_1h_threshold=range(20, 45, 5),
        rsi_4h_threshold=range(20, 45, 5),
        rsi_12h_threshold=range(20, 45, 5),
        rsi_1d_threshold=range(20, 45, 5),
        exit_rsi_4h_threshold=range(60, 85, 5),
        bb_percent_threshold=[0.7, 0.8, 0.9],
        maximize='Sharpe Ratio',
        constraint=lambda p: all([
            p.rsi_1h_threshold < p.exit_rsi_4h_threshold,
            p.rsi_4h_threshold < p.exit_rsi_4h_threshold,
            p.rsi_12h_threshold < p.exit_rsi_4h_threshold,
            p.rsi_1d_threshold < p.exit_rsi_4h_threshold
        ])
    )
    
    print("🏆 OPTIMIZATION RESULTS:")
    print("=" * 30)
    print(stats)
    
    return stats


if __name__ == "__main__":
    # Example usage
    
    # 1. Run basic long strategy
    print("Running LONG strategy...")
    bt_long, stats_long = run_backtest(
        signal_type='long',
        entry_mode='majority',
        rsi_thresholds={'1H': 30, '4H': 30, '12H': 30, '1D': 30},
        exit_threshold=70,
        show_plot=True
    )
    
    
    # 3. Run optimization (uncomment to use)
    # print("\nRunning optimization...")
    # optimized_stats = optimize_strategy(signal_type='long')