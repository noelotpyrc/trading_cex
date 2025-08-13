# Crypto CEX Trading Strategy

Playing around simple trading strategies

## 🗂️ Project Structure

```
trading_cex/
├── strategies/           # Strategy implementations
│   └── conservative_rsi_strategy.py
├── data/                # Data files and processing
│   ├── merged_*.csv     # Merged crypto datasets
│   └── merge_crypto_data.py
├── backtesting/         # Backtesting scripts
│   ├── run_strategy.py  # Execute backtest
│   ├── plot_strategy.py # Visualize results
│   └── debug_strategy.py # Debug strategy logic
├── analysis/            # Analysis and EDA scripts
│   ├── rsi_analysis.py  # RSI distribution analysis
│   ├── eda_analysis.py  # Exploratory data analysis
│   └── timestamp_analysis.py # Time series analysis
├── results/             # Backtest results and plots
├── utils/               # Utility functions
│   └── test_plot.py     # Plotting utilities
├── config/              # Configuration files
├── main.py              # Main entry point
├── requirements.txt     # Dependencies
└── venv/                # Virtual environment
```

## 🚀 Quick Start

### Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Commands

**Using main.py (recommended):**
```bash
# Run backtest
python main.py backtest

# Plot strategy results
python main.py plot

# Debug strategy logic
python main.py debug

# Merge crypto data
python main.py merge

# Run RSI analysis
python main.py analyze
```

**From individual directories:**
```bash
# Run backtest
cd backtesting
python run_strategy.py

# Plot strategy
cd backtesting
python plot_strategy.py

# Debug strategy
cd backtesting
python debug_strategy.py
```

## 📊 Strategy Performance

The Conservative RSI Strategy has shown:
- **Total Return**: +98.50% over ~5.5 years
- **Win Rate**: 72.7% (8 wins out of 11 trades)
- **Sharpe Ratio**: 1.56
- **Max Drawdown**: 24.12%

## 🔧 Dependencies

**Core Libraries:**
- `backtrader>=1.9.76.123` - Backtesting framework
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization
- `scikit-learn>=1.1.0` - Machine learning utilities

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Data Sources

The strategy uses merged 4-hour cryptocurrency data with multi-timeframe RSI indicators from:
- Binance (BTC/USDT, SOL/USDT, SOL/USD)
- Coinbase (BTC/USD, SOL/USD)

