# Model context

1. Model is using lightGBM, predicting 168h forward window log return.
2. The predictors are built with 30 days lookback data, for 1h, 4h, 12h and 1d timeframes.
3. Feature engineering is based on the lookback data of all timeframes.

# For inference

1. We will have a separate data feed pipeline pulling 1h OHLCV data, stored as .csv file or in a DB table.
2. Inference pipeline should expect an input OHLCV DF, with starting timestamp at least 30 days from the current timestamp and the ending timestamp as the nearest 1h candle stick data's timestamp.
3. With this input DF, we can use lookback data building functions to build the lookback data sets, then use multi timeframe feature engineering functions to build the features for inference, store the features for current 1h OHLCV data in a DB table (append).
4. With the features, generate the prediction for current 1h OHLCV data, store the result in a DB table (append).

# Further tasks

1. Pull based dashboard to show the predictions with the up-to-date OHLCV data
2. Alerting system when prediction based signal is triggered

---------------------------------------------------------------------------------

# Implementation specifications (initial inference pipeline)

## Model selection
- Model location is an input. Accept either:
  - A specific run directory containing `model.txt`, or
  - A models root directory; pick the latest `run_*`/`model.txt`.

## Input data contract
- Source: hourly OHLCV feed (CSV or DB extract) ending at the latest complete hour.
- History requirement: at least 30 days of 1H data, with a small buffer (e.g., +2–6 hours).
- Data must be UTC-normalized, deduplicated, sorted ascending.

## Lookbacks and resampling
- Use the same utilities as training (right-closed (t−Δ, t] bins):
  - 1H: raw right-aligned slice
  - 4H/12H/1D: `resample_ohlcv_right_closed` on the 1H lookback window
- Build lookbacks in-memory for the latest timestamp (no PKL persistence during inference).

## Feature computation
- Build multi-timeframe features from the generated lookbacks for the latest 1H bar only.
- Ensure feature schema alignment to the model via `booster.feature_name()`; add missing columns with NaN if ever needed (but prefer fail-fast on insufficient history).

## Prediction and persistence
- Load the selected LightGBM `model.txt`.
- Produce `y_pred` for the latest bar.
- Persistence:
  - DuckDB (separate DB module): append predictions (and optionally features) with minimal columns: `dataset`, `timestamp`, `model_run` (or path), `y_pred`.
  - Also write debug CSV/Parquet artifacts to a temp/debug directory for inspection.

## Scheduling/cadence
- Run hourly, close to the hour start (on the first complete candle). Small grace offset allowed.

## Failure policy
- If history is insufficient for the required lookbacks/features, fail the run with clear diagnostics (missing bars per timeframe). We can revisit a tolerant mode after observing real data quality.

## Deferred decisions
- Feature schema pinning vs dynamic from `booster.feature_name()` (decide after initial implementation).
- Backfill/replay behavior (not required for the first iteration).
- Model metadata/version logging, alert thresholds/channels, dashboard stack (later).

## Minimal CLI surface (first iteration)
- `--model-root` or `--model-path` (required: one of them)
- `--input-csv` or `--db-conn` (first iteration: CSV)
- `--dataset` (e.g., `BINANCE_BTCUSDT.P, 60`)
- `--debug-dir` (optional)
- Output to DuckDB handled by a separate writer module (configurable destination).
