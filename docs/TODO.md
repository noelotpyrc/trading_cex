# Engineering task

List as priority high to low

1. Creat a production pipeline, including:
    1. Inference from models
    1. Live data pulling (batch)
    1. Dashboard for live data and signal trigger alert
2. Create a pipeline to generate permutation test results
3. Create a pipeline (framework) to generate target variables based on certain strategy, given the historical ohlcv data:
    1. The target generation implementation may be similar to the current target function in feature engineering, for each row (candle stick) takes a forward window as input data, create target variables based on strategy criteria
    1. The strategy should always have entry and exit rules
4. Create a EDA process to run prescreening on targets, including but not limited:
    1. Basic stats, distribution, skewness, etc.
    1. Backtesting results (based on sizing, fees, etc.) or simplified simulated gain/loss (simple sizing assumption).
    1. Create (interactive) visualizations.
    1. Permutation test results. 

# Modeling task

1. Remove the early stopping part so we could have longer time period for test
2. Regime detection methods.
3. Come up with crucial stats for target EDA
4. More detailed model inspection, e.g., combined feature importance, etc.

# Strategy task

1. TBD

# Research task

1. Diffusion model
2. HMM for regime detection