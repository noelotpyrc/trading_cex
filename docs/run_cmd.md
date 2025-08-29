# Prepare train/val/test split

```
python model/prepare_training_data.py \
  --input "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv" \
  --output-dir "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/prepared" \
  --target {target_name}
```

# Train lgbm model

```
python model/train_lgbm_quantiles.py \
  --data-dir "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/prepared_{target_name}" \
  --target {target_name} \
  --quantiles 0.05 0.5 0.95
```

# Visualization lgbm model results

```
python model/plot_predictions_interactive.py \
  --run-dir "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/lgbm_y_logret_168h/run_20250823_183508" \
  --prefix quantile_preds_interactive \
  --ohlcv-csv "data/BINANCE_BTCUSDT.P, 60.csv"
```