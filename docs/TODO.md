# Engineering task

List as priority high to low

1. Add feature store management with duckDB
2. Refine HMM model training/inference pipelines
3. Set up an artifacts sharing/report process with shared storage and unified interfaces.
4. Create a pipeline to generate permutation test results
5. Create a pipeline (framework) to generate target variables based on certain strategy, given the historical ohlcv data:
    1. The target generation implementation may be similar to the current target function in feature engineering, for each row (candle stick) takes a forward window as input data, create target variables based on strategy criteria
    1. The strategy should always have entry and exit rules
6. Create a EDA process to run prescreening on targets, including but not limited:
    1. Basic stats, distribution, skewness, etc.
    1. Backtesting results (based on sizing, fees, etc.) or simplified simulated gain/loss (simple sizing assumption).
    1. Create (interactive) visualizations.
    1. Permutation test results. 

# Modeling task

1. Regime detection models, do more research to improve and try different ways to combine them with lgbm models.
2. Try different classification targets with lgbm model.
3. Try LSTM for directly prediction or regime detection.
4. Come up with crucial stats for target EDA
5. More detailed model inspection, e.g., combined feature importance, etc.

# Strategy task

1. Try some simple strategy on 1min DOGE data

# Research task

1. Diffusion model
2. LSTM for regime detection