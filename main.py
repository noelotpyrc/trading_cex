#!/usr/bin/env python3
"""
Main Entry Point for Crypto CEX Trading Strategy

This script provides easy access to all functionality from the crypto_cex root directory.
"""

import sys
from pathlib import Path

def main():
    """Main entry point"""
    print("🚀 Crypto CEX Trading Strategy")
    print("=" * 40)
    print("Available commands:")
    print("1. Run backtest: python main.py backtest")
    print("2. Plot strategy: python main.py plot")
    print("3. Debug strategy: python main.py debug")
    print("4. Merge data: python main.py merge")
    print("5. Run analysis: python main.py analyze")
    
    if len(sys.argv) < 2:
        print("\n❌ Please specify a command!")
        return
    
    command = sys.argv[1].lower()
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    if command == "backtest":
        from strategy_backtesting.run_strategy import run_single_test
        run_single_test()
    elif command == "plot":
        from strategy_backtesting.plot_strategy import plot_strategy_results
        plot_strategy_results()
    elif command == "debug":
        from strategy_backtesting.debug_strategy import debug_strategy_logic
        debug_strategy_logic()
    elif command == "merge":
        from data.merge_crypto_data import main as merge_main
        merge_main()
    elif command == "analyze":
        from analysis.rsi_analysis import main as analyze_main
        analyze_main()
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
