### Modeling (LightGBM pipeline)

This folder provides a config-driven LightGBM training pipeline and helper tools for batch runs, diagnostics, merging quantile predictions, and interactive plotting.

### Quickstart

Run a single config through the full pipeline (prepare splits → optional tuning → train → persist artifacts):

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/run_lgbm_pipeline.py \
  --config /Users/noel/projects/trading_cex/configs/model_configs/btc_24h_logret_example.json \
  --log-level INFO
```

Outputs are written under `<output_dir>/run_<timestamp>_lgbm_<target>_<objective>/`, including:
- model.txt, pred_train.csv, pred_val.csv, pred_test.csv
- feature_importance.csv, metrics.json
- pipeline_config.json, best_params.json, run_metadata.json, paths.json
- tuning_trials.csv (if tuning enabled)

Reusing prepared splits: set `split.existing_dir` in the config to a previously created prepared folder (contains `X_*`/`y_*`).

### Batch runs for multiple configs

Filter by input_data and target.variable, then execute each config:

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/run_configs_batch.py \
  --config-dir /Users/noel/projects/trading_cex/configs/model_configs \
  --input-data "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv" \
  --target y_logret_24h \
  --log-level INFO
```

Or run explicit configs:

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/run_configs_batch.py \
  --configs \
    /Users/noel/projects/trading_cex/configs/model_configs/binance_btcusdt_p60_quantile_y_logret_24h_q05.json \
    /Users/noel/projects/trading_cex/configs/model_configs/binance_btcusdt_p60_quantile_y_logret_24h_q50.json \
  --log-level INFO
```

Tip: `--input-data` matching is exact; ensure the same absolute path in your configs for discovery/merging tools.

### Merge quantile predictions

Merge multiple quantile runs’ predictions into a single file for plotting. Either specify run dirs explicitly:

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/merge_quantile_predictions.py \
  --run-dirs \
    "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/run_20250901_120000_lgbm_y_logret_24h_quantile" \
    "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/run_20250901_121500_lgbm_y_logret_24h_quantile" \
  --split test \
  --out-dir "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/merged_quantiles"
```

…or auto-discover by models root, target, and input_data:

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/merge_quantile_predictions.py \
  --models-root "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60" \
  --target y_logret_24h \
  --input-data "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv" \
  --split val \
  --out-dir "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/"
```

Writes `pred_<split>.csv` with columns: `timestamp` (if present), `y_true`, `pred_q05`, `pred_q10`, …

### Interactive plots

Val/test plots with candlesticks and optional signals overlays:

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/plot_predictions_interactive.py \
  --run-dir "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_120h" \
  --ohlcv-csv "/Users/noel/projects/trading_cex/data/BINANCE_BTCUSDT.P, 60.csv"
```

Train-only plot from a merged `pred_train.csv`:

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/plot_predictions_interactive_train.py \
  --pred-train "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/merged_quantiles/pred_train.csv" \
  --ohlcv-csv "/Users/noel/projects/trading_cex/data/BINANCE_BTCUSDT.P, 60.csv"
```

Options:
- `--signals-val/--signals-test` or `--signals-train` to overlay expected returns from signals CSVs
- `--show-prob-up` and `--prob-thresholds` to overlay probabilities
- `--mark-cross-violations` to annotate cross_violations on expected return

### Inspect tuning best iterations (quantiles)

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/inspect_tuning_best_iterations.py \
  --models-root "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60" \
  --target y_logret_24h \
  --input-data "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv" \
  --out "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/best_iterations.csv"
```

### Prepare training splits manually (optional)

If you prefer preparing splits outside the pipeline:

```bash
/Users/noel/projects/trading_cex/venv/bin/python model/prepare_training_data.py \
  --input "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv" \
  --output-dir "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/prepared" \
  --target y_logret_24h \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

This writes `X_*`, `y_*`, and `prep_metadata.json` into `<output-dir>_<target>/`.


