#!/usr/bin/env python3
"""
Strategy Runner Script

Run Conservative RSI Strategy on crypto datasets with full customization
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import warnings
import json
warnings.filterwarnings('ignore')

# Set matplotlib backend for better compatibility
matplotlib.use('TkAgg')  # Use TkAgg backend for display

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategies.conservative_rsi_strategy import ConservativeRSIStrategy, SmoothRSIData


def run_custom_strategy(
    # === ENTRY STRATEGY SETTINGS ===
    entry_rsi_rules={'RSI_4H': 30, 'RSI_12H': 30, 'RSI_1D': 30},
    entry_mode='all',  # 'all', 'any', 'majority', 'count'
    min_required_count=2,  # Only used if entry_mode='count'
    
    # === EXIT STRATEGY SETTINGS ===
    exit_rsi_rules={'RSI_4H': 70},
    exit_mode='any',  # 'any', 'all'
    
    # === PROFIT/LOSS EXITS ===
    enable_profit_target=False,
    profit_target_pct=0.15,  # 15% profit target
    enable_stop_loss=False,
    stop_loss_pct=-0.08,  # 8% stop loss
    
    # === POSITION MANAGEMENT ===
    position_pct=0.95,  # Use 95% of available cash
    min_price=1000.0,   # Minimum BTC price filter
    
    # === DISPLAY SETTINGS ===
    initial_cash=100000,
    show_plot=True,
    verbose=True
):
    """
    Run a fully customizable RSI strategy with your own entry/exit rules.
    
    ENTRY PARAMETERS:
    - entry_rsi_rules: dict of {'RSI_1H': 30, 'RSI_4H': 30, ...} (when to buy)
    - entry_mode: 'all'=all rules must pass, 'any'=any rule passes, 
                  'majority'=majority pass, 'count'=minimum count
    - min_required_count: minimum rules that must pass (if entry_mode='count')
    
    EXIT PARAMETERS:
    - exit_rsi_rules: dict of {'RSI_4H': 70, ...} (when to sell)
    - exit_mode: 'any'=any exit rule triggers, 'all'=all must trigger
    - enable_profit_target: True/False for profit target exit
    - profit_target_pct: 0.15 = 15% profit target
    - enable_stop_loss: True/False for stop loss exit
    - stop_loss_pct: -0.08 = 8% stop loss
    
    EXAMPLES:
    
    # Conservative (all timeframes oversold, exit on 4H overbought)
    run_custom_strategy(
        entry_rsi_rules={'RSI_4H': 30, 'RSI_12H': 30, 'RSI_1D': 30},
        entry_mode='all',
        exit_rsi_rules={'RSI_4H': 70}
    )
    
    # Aggressive (any timeframe oversold, early exit with profit target)
    run_custom_strategy(
        entry_rsi_rules={'RSI_1H': 35, 'RSI_4H': 35},
        entry_mode='any',
        exit_rsi_rules={'RSI_1H': 65},
        enable_profit_target=True,
        profit_target_pct=0.12
    )
    
    # Risk-managed (majority entry, multiple exits)
    run_custom_strategy(
        entry_rsi_rules={'RSI_1H': 30, 'RSI_4H': 30, 'RSI_12H': 30, 'RSI_1D': 30},
        entry_mode='majority',
        exit_rsi_rules={'RSI_1H': 65, 'RSI_4H': 70},
        exit_mode='any',
        enable_profit_target=True,
        profit_target_pct=0.20,
        enable_stop_loss=True,
        stop_loss_pct=-0.06
    )
    """
    
    test_file = "../data/binance_btcusdt.p_smooth_multi_rsi_corrected.csv"
    
    if not Path(test_file).exists():
        print(f"❌ Test file {test_file} not found!")
        return None
    
    # Build strategy parameters
    strategy_params = {
        'entry_rsi_rules': entry_rsi_rules,
        'entry_mode': entry_mode,
        'min_required_count': min_required_count,
        'exit_rsi_rules': exit_rsi_rules,
        'exit_mode': exit_mode,
        'enable_profit_target': enable_profit_target,
        'profit_target_pct': profit_target_pct,
        'enable_stop_loss': enable_stop_loss,
        'stop_loss_pct': stop_loss_pct,
        'position_pct': position_pct,
        'min_price': min_price,
        'verbose': verbose
    }
    
    print("🎯 CUSTOM RSI STRATEGY CONFIGURATION")
    print("=" * 50)
    print(f"💰 Initial Capital: ${initial_cash:,}")
    print(f"📊 Position Size: {position_pct*100:.0f}% of available cash")
    print()
    print("📈 ENTRY RULES:")
    for rsi_col, threshold in entry_rsi_rules.items():
        print(f"   • {rsi_col} ≤ {threshold}")
    print(f"   • Entry Mode: {entry_mode.upper()}")
    if entry_mode == 'count':
        print(f"   • Min Required: {min_required_count} rules")
    print()
    print("📉 EXIT RULES:")
    if exit_rsi_rules:
        for rsi_col, threshold in exit_rsi_rules.items():
            print(f"   • {rsi_col} ≥ {threshold}")
        print(f"   • Exit Mode: {exit_mode.upper()}")
    if enable_profit_target:
        print(f"   • Profit Target: {profit_target_pct*100:.0f}%")
    if enable_stop_loss:
        print(f"   • Stop Loss: {stop_loss_pct*100:.0f}%")
    print("=" * 50)
    print()
    
    # Run the backtest
    results = run_conservative_rsi_backtest(
        test_file, 
        initial_cash=initial_cash, 
        plot=show_plot, 
        strategy_params=strategy_params
    )
    
    return results


def run_conservative_rsi_backtest(csv_file, initial_cash=100000, plot=True, strategy_params=None):
    """
    Run backtest on a single crypto dataset
    
    Args:
        csv_file: Path to smooth multi-timeframe RSI CSV file
        initial_cash: Starting capital
        plot: Whether to show plots
        strategy_params: Optional strategy parameters dictionary
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
    df = df.dropna(subset=['RSI_1H', 'RSI_4H', 'RSI_12H', 'RSI_1D'])
    print(f"📊 Data: {len(df)} rows (removed {initial_rows - len(df)} with missing RSI)")
    print(f"📅 Period: {df.index[0]} to {df.index[-1]}")
    
    # Initialize backtrader
    cerebro = bt.Cerebro()
    
    # Add strategy with optional custom parameters
    if strategy_params:
        cerebro.addstrategy(ConservativeRSIStrategy, **strategy_params)
    else:
        cerebro.addstrategy(ConservativeRSIStrategy, verbose=True)
    
    # Add data feed
    data = SmoothRSIData(dataname=df)
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
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            # Create results directory if it doesn't exist
            results_dir = Path("../results")
            results_dir.mkdir(exist_ok=True)
            
            # Generate plot filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            plot_filename = results_dir / f"backtest_plot_{timestamp}.png"
            
            # Generate and save the plot
            figs = cerebro.plot(style='candlestick', 
                        barup='green', 
                        bardown='red',
                        volume=False,  # Hide volume for cleaner view
                        figsize=(16, 10),  # Larger figure
                        dpi=100,  # Higher resolution
                        numfigs=1,  # Single figure
                        returnfig=True)  # Return figure for saving
            
            # Save the plot - handle different return types from backtrader
            plot_saved = False
            if figs:
                if isinstance(figs, list) and len(figs) > 0:
                    # Handle list of figures
                    for fig_item in figs:
                        if isinstance(fig_item, list) and len(fig_item) > 0:
                            fig = fig_item[0]  # Get the first figure from nested list
                            if hasattr(fig, 'savefig'):
                                fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                                print(f"✅ Plot saved: {plot_filename}")
                                plot_saved = True
                                break
                        elif hasattr(fig_item, 'savefig'):
                            fig_item.savefig(plot_filename, dpi=150, bbox_inches='tight')
                            print(f"✅ Plot saved: {plot_filename}")
                            plot_saved = True
                            break
                elif hasattr(figs, 'savefig'):
                    # Handle single figure
                    figs.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    print(f"✅ Plot saved: {plot_filename}")
                    plot_saved = True
            
            # Fallback: save current matplotlib figure
            if not plot_saved:
                current_fig = plt.gcf()
                if current_fig.get_axes():  # Check if figure has content
                    current_fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    print(f"✅ Plot saved (fallback): {plot_filename}")
                    plot_saved = True
            
            # Also try to display if possible
            plt.show()
            print("✅ Plot displayed successfully!")
            
            if not plot_saved:
                print("⚠️ Could not save plot - no valid figure found")
            
        except Exception as e:
            print(f"⚠️ Could not generate plot: {e}")
            print("💡 Try running in a Jupyter notebook or check matplotlib backend")
    
    return results


def load_strategy_config(config_file):
    """Load strategy configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file {config_file} not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return None


def create_example_config():
    """Create example configuration files"""
    configs = {
        "conservative_strategy.json": {
            "strategy_name": "Conservative Strategy",
            "description": "All timeframes must agree for entry, exit on 4H overbought",
            "entry_rsi_rules": {
                "RSI_1H": 30,
                "RSI_4H": 30,
                "RSI_12H": 30,
                "RSI_1D": 30
            },
            "entry_mode": "all",
            "min_required_count": 2,
            "exit_rsi_rules": {
                "RSI_4H": 70
            },
            "exit_mode": "any",
            "enable_profit_target": False,
            "profit_target_pct": 0.15,
            "enable_stop_loss": False,
            "stop_loss_pct": -0.08,
            "position_pct": 0.95,
            "min_price": 1000.0,
            "initial_cash": 100000,
            "show_plot": True,
            "verbose": True
        },
        "aggressive_strategy.json": {
            "strategy_name": "Aggressive Strategy",
            "description": "Any timeframe oversold triggers entry, early exit with profit target",
            "entry_rsi_rules": {
                "RSI_1H": 35,
                "RSI_4H": 35
            },
            "entry_mode": "any",
            "exit_rsi_rules": {
                "RSI_1H": 65
            },
            "exit_mode": "any",
            "enable_profit_target": True,
            "profit_target_pct": 0.12,
            "enable_stop_loss": True,
            "stop_loss_pct": -0.05,
            "position_pct": 0.90,
            "min_price": 1000.0,
            "initial_cash": 100000,
            "show_plot": True,
            "verbose": False
        },
        "majority_strategy.json": {
            "strategy_name": "Majority Strategy",
            "description": "Majority of RSI must be oversold, exit on 12H overbought",
            "entry_rsi_rules": {
                "RSI_1H": 30,
                "RSI_4H": 30,
                "RSI_12H": 30,
                "RSI_1D": 30
            },
            "entry_mode": "majority",
            "exit_rsi_rules": {
                "RSI_12H": 70
            },
            "exit_mode": "any",
            "enable_profit_target": False,
            "enable_stop_loss": False,
            "position_pct": 0.95,
            "min_price": 1000.0,
            "initial_cash": 100000,
            "show_plot": True,
            "verbose": True
        }
    }
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    for filename, config in configs.items():
        filepath = config_dir / filename
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created example config: {filepath}")
    
    return configs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Customizable RSI Strategy Backtest')
    parser.add_argument('--config', type=str, help='JSON config file path')
    parser.add_argument('--create_examples', action='store_true',
                       help='Create example configuration files')
    parser.add_argument('--show_examples', action='store_true', 
                       help='Show strategy configuration examples')
    
    args = parser.parse_args()
    
    if args.create_examples:
        print("🎯 Creating example configuration files...")
        create_example_config()
        print("\n💡 Use --config configs/majority_strategy.json to test!")
    
    elif args.config:
        print(f"🎯 Loading strategy from config file: {args.config}")
        config = load_strategy_config(args.config)
        if config:
            print(f"📋 Strategy: {config.get('strategy_name', 'Custom Strategy')}")
            print(f"📝 Description: {config.get('description', 'No description')}")
            print()
            
            # Extract strategy parameters (filter out metadata)
            strategy_params = {k: v for k, v in config.items() 
                             if k not in ['strategy_name', 'description']}
            results = run_custom_strategy(**strategy_params)
    
    elif args.show_examples:
        print("🎯 RSI STRATEGY CONFIGURATION EXAMPLES")
        print("=" * 60)
        print()
        print("1️⃣ CONSERVATIVE STRATEGY (All timeframes must agree):")
        print("run_custom_strategy(")
        print("    entry_rsi_rules={'RSI_4H': 30, 'RSI_12H': 30, 'RSI_1D': 30},")
        print("    entry_mode='all',")
        print("    exit_rsi_rules={'RSI_4H': 70}")
        print(")")
        print()
        print("2️⃣ AGGRESSIVE STRATEGY (Any timeframe triggers entry):")
        print("run_custom_strategy(")
        print("    entry_rsi_rules={'RSI_1H': 35, 'RSI_4H': 35},")
        print("    entry_mode='any',")
        print("    exit_rsi_rules={'RSI_1H': 65},")
        print("    enable_profit_target=True,")
        print("    profit_target_pct=0.12")
        print(")")
        print()
        print("3️⃣ RISK-MANAGED STRATEGY (Multiple exits):")
        print("run_custom_strategy(")
        print("    entry_rsi_rules={'RSI_1H': 30, 'RSI_4H': 30, 'RSI_12H': 30},")
        print("    entry_mode='majority',")
        print("    exit_rsi_rules={'RSI_1H': 65, 'RSI_4H': 70},")
        print("    exit_mode='any',")
        print("    enable_profit_target=True,")
        print("    profit_target_pct=0.20,")
        print("    enable_stop_loss=True,")
        print("    stop_loss_pct=-0.06")
        print(")")
        print()
        print("4️⃣ PATIENT STRATEGY (Wait for daily RSI):")
        print("run_custom_strategy(")
        print("    entry_rsi_rules={'RSI_1D': 25},")
        print("    entry_mode='all',")
        print("    exit_rsi_rules={'RSI_1D': 80}")
        print(")")
        print()
        print("=" * 60)
        print("💡 Edit the parameters at the bottom of this file to customize!")
        print("💡 Or use --create_examples to generate JSON config files!")
    
    else:
        print("🚀 Running default custom RSI strategy...")
        print("💡 Use --config <file.json> to load custom parameters")
        print("💡 Use --create_examples to generate example config files")
        print()
        
        # ===== DEFAULT STRATEGY CONFIGURATION =====
        results = run_custom_strategy(
            # Entry Strategy - When to BUY (majority of 4 RSI <= 30)
            entry_rsi_rules={'RSI_1H': 30, 'RSI_4H': 30, 'RSI_12H': 30, 'RSI_1D': 30},
            entry_mode='majority',  # majority must be <= 30
            
            # Exit Strategy - When to SELL (12H RSI >= 70)
            exit_rsi_rules={'RSI_12H': 70},
            exit_mode='any',   # any, all
            
            # Additional Exits (disabled)
            enable_profit_target=False,
            enable_stop_loss=False,
            
            # Settings
            initial_cash=100000,
            show_plot=True,
            verbose=True
        )
        # ==========================================