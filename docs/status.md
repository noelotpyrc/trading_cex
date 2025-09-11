# Engineering pipelines

1. All data preparation, feature engineering and model training/diagnose pipelines are done
2. Model training (lgbm) can be set up with configs now

# Modeling

Current modeling is based on 30 days lookback windows built for 1h, 4h, 12h and 1d timeframes, predicting a few different targets.

1. Tried to fit different targets with diffferent objectives (quantiles, huber, regression, etc.)
2. Forward 7 day log return prediction shows some promising results from simulation

# Strategy

Only tried a RSI based strategy, no other strategy set up yet.