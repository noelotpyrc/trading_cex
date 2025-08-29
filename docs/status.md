# Engineering pipelines

1. All data preparation, feature engineering and model training/diagnose pipelines are done
2. Most of them are implemented as standalone run script, which limits the efficiency of trying different configs

# Modeling

1. Two majore regression targets are used for model training, log return and MFE
2. Two sets of features were used for model training, basic lags and combined features
3. Both of the regression models have some underfitting issues for different target objectives or different hyper parameters
4. Forward 7 day regression prediction shows some potential of trading strategy

# Strategy

Zero progress