# Unified training pipeline (single config → full run)

```
python model/run_lgbm_pipeline.py \
  --config configs/model_configs/btc_24h_logret_example.json \
  --log-level INFO
```

## Config example (supports all LightGBM regression objectives)

```
{
  "input_data": "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv",
  "output_dir": "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60",
  "target": {
    "variable": "y_logret_24h",
    "objective": { "name": "quantile", "params": {"alpha": 0.05} }
    // or { "name": "regression" } | "regression_l1" | "huber" | "fair" | "poisson" | "mape" | "gamma" | "tweedie"
  },
  "split": { "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15 },
  "model": {
    "type": "lgbm",
    "hyperparameter_tuning_method": "grid", // or "bayesian"
    "hyperparameter_search_space": {
      "learning_rate": [0.05, 0.1],
      "num_leaves": [31, 63],
      "max_depth": [6, 9]
    },
    "params": {
      "learning_rate": 0.1,
      "num_leaves": 31,
      "max_depth": 6,
      "min_data_in_leaf": 100,
      "feature_fraction": 0.9,
      "bagging_fraction": 0.9,
      "bagging_freq": 5,
      "lambda_l1": 0.0,
      "lambda_l2": 0.0,
      "num_boost_round": 1000,
      "early_stopping_rounds": 100,
      "seed": 42
    },
    "eval_metrics": ["rmse"]
  }
}
```

## Artifacts
- prepared data: `<output_dir>/prepared_{target}/X_*.csv, y_*.csv, prep_metadata.json`
- run directory: `<output_dir>/{target}/lgbm_{objective}/run_YYYYmmdd_HHMMSS/`
  - model.txt, metrics.json, feature_importance.csv
  - pred_train.csv, pred_val.csv, pred_test.csv
  - run_metadata.json (captures original config and resolved best params)