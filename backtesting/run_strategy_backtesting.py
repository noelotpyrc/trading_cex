#!/usr/bin/env python3
"""
Run Backtesting.py RSI Strategy with JSON Configuration

This script runs the backtesting.py framework strategy using the same
JSON configuration files as the backtrader strategies.
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from strategies.rsi_strategy_backtesting import run_backtest, optimize_strategy, ConservativeRSIStrategy


def load_config(config_path):
    """Load strategy configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def map_config_to_backtest_params(config):
    """Map JSON config parameters to backtesting.py parameters"""
    
    # Extract RSI thresholds
    rsi_thresholds = {}
    if 'entry_rsi_rules' in config:
        for timeframe, threshold in config['entry_rsi_rules'].items():
            # Map timeframe names
            if timeframe == 'RSI_1H':
                rsi_thresholds['1H'] = threshold
            elif timeframe == 'RSI_4H':
                rsi_thresholds['4H'] = threshold
            elif timeframe == 'RSI_12H':
                rsi_thresholds['12H'] = threshold
            elif timeframe == 'RSI_1D':
                rsi_thresholds['1D'] = threshold
    
    # Default thresholds if not specified
    if not rsi_thresholds:
        if config.get('signal_type', 'long') == 'long':
            rsi_thresholds = {'1H': 30, '4H': 30, '12H': 30, '1D': 30}
        else:
            rsi_thresholds = {'1H': 75, '4H': 75, '12H': 75, '1D': 75}
    
    # Extract exit thresholds - only enable specified timeframes
    exit_1h_threshold = None
    exit_4h_threshold = None  
    exit_12h_threshold = None
    exit_1d_threshold = None
    enabled_exit_timeframes = []
    
    if 'exit_rsi_rules' in config:
        for timeframe, threshold in config['exit_rsi_rules'].items():
            if timeframe == 'RSI_1H':
                exit_1h_threshold = threshold
                enabled_exit_timeframes.append('1H')
            elif timeframe == 'RSI_4H':
                exit_4h_threshold = threshold
                enabled_exit_timeframes.append('4H')
            elif timeframe == 'RSI_12H':
                exit_12h_threshold = threshold
                enabled_exit_timeframes.append('12H')
            elif timeframe == 'RSI_1D':
                exit_1d_threshold = threshold
                enabled_exit_timeframes.append('1D')
    
    # If no exit rules specified, use default based on signal type
    if not enabled_exit_timeframes:
        if config.get('signal_type', 'long') == 'long':
            exit_4h_threshold = 70
            enabled_exit_timeframes = ['4H']
        else:
            exit_4h_threshold = 30
            enabled_exit_timeframes = ['4H']
    
    # No adjustment needed for short strategy since we use exact thresholds from config
    
    # Map parameters
    params = {
        'signal_type': config.get('signal_type', 'long'),
        'entry_mode': config.get('entry_mode', 'majority'),
        'rsi_thresholds': rsi_thresholds,
        'exit_1h_threshold': exit_1h_threshold,
        'exit_4h_threshold': exit_4h_threshold,
        'exit_12h_threshold': exit_12h_threshold,
        'exit_1d_threshold': exit_1d_threshold,
        'enabled_exit_timeframes': enabled_exit_timeframes,
        'enable_bb_filter': config.get('enable_bb_filter', False),
        'bb_percent_threshold': config.get('bb_percent_threshold', 0.8),
        'enable_profit_target': config.get('enable_profit_target', False),
        'profit_target_pct': config.get('profit_target_pct', 0.15),
        'enable_stop_loss': config.get('enable_stop_loss', False),
        'stop_loss_pct': abs(config.get('stop_loss_pct', 0.08)),  # Ensure positive
        'initial_cash': config.get('initial_cash', 100000),
        'show_plot': config.get('show_plot', True)
    }
    
    return params


def run_strategy_from_config(config_path, data_file=None):
    """Run strategy using JSON configuration"""
    
    print(f"🔧 LOADING CONFIGURATION: {config_path}")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    
    # Get config filename for plot
    config_name = Path(config_path).stem
    
    # Display config info
    print(f"Strategy: {config.get('strategy_name', 'Unknown')}")
    print(f"Description: {config.get('description', 'No description')}")
    print()
    
    # Map to backtest parameters
    params = map_config_to_backtest_params(config)
    
    # Use default data file if not specified
    if data_file is None:
        # Get the project root directory (parent of backtesting folder)
        project_root = Path(__file__).parent.parent
        data_file = project_root / "data" / "binance_btcusdt.p_smooth_multi_rsi_corrected.csv"
    
    # Print mapped parameters
    print("🎯 MAPPED PARAMETERS:")
    print("-" * 30)
    for key, value in params.items():
        print(f"{key}: {value}")
    print()
    
    # Run backtest
    try:
        bt, stats = run_backtest(
            data_file=data_file,
            plot_filename=f"{config_name}_backtest.html",
            **params
        )
        
        print("\n✅ BACKTEST COMPLETED SUCCESSFULLY!")
        
        # Save results if specified in config
        if config.get('save_results', False):
            results_file = f"results_{Path(config_path).stem}.json"
            save_results(stats, results_file)
            print(f"💾 Results saved to: {results_file}")
        
        return bt, stats
        
    except Exception as e:
        print(f"❌ BACKTEST FAILED: {e}")
        return None, None


def save_results(stats, filename):
    """Save backtest results to JSON file"""
    
    # Convert stats to JSON-serializable format
    results = {}
    for key, value in stats.items():
        if isinstance(value, (int, float, str, bool)):
            results[key] = value
        elif hasattr(value, 'total_seconds'):  # timedelta
            results[key] = str(value)
        else:
            results[key] = str(value)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)






def main():
    """Main command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Run Backtesting.py RSI Strategy with JSON Configuration"
    )
    
    parser.add_argument(
        'config', 
        help='Path to JSON configuration file or directory'
    )
    
    parser.add_argument(
        '--data-file', '-d',
        help='Path to data file (default: data/binance_btcusdt.p_smooth_multi_rsi_corrected.csv)'
    )
    
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable interactive plotting'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    # Check if config file exists
    if config_path.is_file():
        # Run single config
        bt, stats = run_strategy_from_config(config_path, args.data_file)
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("🎯 BACKTESTING.PY RSI STRATEGY RUNNER")
        print("=" * 50)
        print()
        print("Usage: python run_backtesting_strategy.py <config_file.json>")
        print("Example: python run_backtesting_strategy.py configs/long_rsi_strategy.json")
    else:
        main()