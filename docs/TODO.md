# Engineering task

1. Refactor model training pipeline, use config file to start model training, parameterize:
    1. Feature selection
    2. Target selection, including objective selection
    3. Split method
    4. Hyper parameter search space
2. Document run process and file persist system

# Modeling task

1. Add hyper parameter tuning process to try to overcome underfitting issue
2. Use different sets of hyper parameters for different objectives, including different quantiles
3. Add evaluation process to pick the best model (based on validation set) for each config

# Strategy task

1. For regression model, add numerical metric (expected value based on quantile predictions) for creating trading signals

# Research task

1. Permutation of lookback data and forward data