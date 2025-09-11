## Crypto CEX Trading Strategy

Research repo for feature engineering, ML forecasting of future log-returns, signal generation (quantiles and RSI), and backtesting/visualization on centralized exchanges (CEX). Works primarily with Binance/Coinbase OHLCV and multi-timeframe RSI features.

### What you can do now

- **Engineer features and targets**: Scripts under `feature_engineering/` build multi-timeframe features, lags, and supervised targets; tools to merge features/targets are included.
- **Train and evaluate models**: Config-driven LightGBM pipeline in `model/` (`run_lgbm_pipeline.py`, `run_configs_batch.py`) with support for Huber/Quantile objectives, per-split artifacts, and diagnostics.
- **Merge and plot quantile predictions**: `model/merge_quantile_predictions.py` combines runs; interactive plots via `model/plot_predictions_interactive.py` and `model/plot_predictions_interactive_train.py` with candlesticks and overlays.
- **Generate trading signals**: Quantile-to-signal utilities in `strategies/` (`quantile_signals.py`, `compute_signals_from_quantiles.py`) and visual trade inspection in `strategies/plot_trades_candles.py`.
- **Backtest heuristic RSI strategies**: JSON-configurable RSI strategies using Backtesting.py and Backtrader in `backtesting/` (`run_strategy_backtesting.py`, `run_strategy_backtrader.py`) with example configs in `backtesting/configs/`.
- **EDA and utilities**: Exploratory analyses in `analysis/` and data helpers (e.g., Binance klines download/merge) in `utils/`.

### Current status

- **RSI strategies**: Implemented and backtested; interactive HTML reports saved under `results/`.
- **Modeling pipeline**: Operational for several horizons (e.g., 24h/48h/120h/168h) with Huber/Quantile objectives; interactive diagnostics and prediction plots available.
- **Signals**: Quantile-based signal flow implemented and under active tuning; integration with systematic backtests is ongoing.
- **Data**: Sample OHLCV and multi-timeframe RSI CSVs included in `data/` to reproduce plots/backtests quickly.
- **Docs**: See `docs/` for methodology and pipeline usage; documentation is evolving alongside experiments.

### High-level layout

```
trading_cex/
├── feature_engineering/   # Build features, lags, targets; merge utilities
├── model/                 # LightGBM pipeline, batch runs, diagnostics, plots
├── strategies/            # Signals from models/RSI and chart inspection
├── backtesting/           # Backtesting (Backtesting.py & Backtrader) + configs
├── analysis/              # EDA scripts and notebooks
├── utils/                 # Data utilities (download/merge/generate TA)
├── data/                  # Example OHLCV/features targets CSVs
├── results/               # Backtests and interactive plots
├── docs/                  # Methodology and usage notes
└── requirements.txt       # Project dependencies
```

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data sources

The repository uses hourly/4h/daily OHLCV and derived RSI features from:
- Binance (e.g., BTC/USDT, SOL/USDT)
- Coinbase (e.g., BTC/USD, SOL/USD)

Refer to `docs/` for details on feature/target engineering and signal flow.