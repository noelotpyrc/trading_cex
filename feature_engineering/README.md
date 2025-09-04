### Feature engineering

This folder contains leakage-safe utilities and CLIs to build features and targets for time-series modeling/backtesting. There are two ways to generate features:

- Current-bar + lag features (fast/lean)
- Rich multi-timeframe features (broad TA/stat/liquidity set)

Both paths use right-closed resampling and per-row lookbacks to avoid future data.

### Data conventions

- OHLCV columns: `open, high, low, close, volume` (lowercase)
- Index/time: prefer a `DatetimeIndex`. If a `timestamp` column exists, it will be parsed to UTC-naive and used.
- Timeframe suffixes: columns end with `_1H`, `_4H`, `_12H`, `_1D`.
- Lag naming: replace `_current` with `_lag_k` (e.g., `close_logret_current_4H` → `close_logret_lag_2_4H`). Interactions apply lag on both sides.

### Quickstart

1) Build lookbacks once per dataset

```bash
/Users/noel/projects/trading_cex/venv/bin/python feature_engineering/build_lookbacks.py \
  --input "/Users/noel/projects/trading_cex/data/BINANCE_BTCUSDT.P, 60.csv" \
  --output "/Volumes/Extreme SSD/trading_data/cex/lookbacks" \
  --timeframes 1H 4H 12H 1D \
  --lookback 168
```

2) Build current-bar features and lag-1..K for selected timeframes

```bash
/Users/noel/projects/trading_cex/venv/bin/python feature_engineering/build_current_bar_lag_features.py \
  --dataset "BINANCE_BTCUSDT.P, 60" \
  --lookbacks-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks" \
  --ohlcv-1h "/Users/noel/projects/trading_cex/data/BINANCE_BTCUSDT.P, 60.csv" \
  --timeframes 1H 4H 12H 1D \
  --max-lag 3 \
  --output "/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60/current_bar_with_lags_1H-4H-12H-1D_lags3.csv"
```

Tips:
- List available datasets under the lookbacks dir: `--list`

3) Build forward-window targets (returns, MFE/MAE, optional triple-barrier)

```bash
/Users/noel/projects/trading_cex/venv/bin/python feature_engineering/build_targets.py \
  --input "/Users/noel/projects/trading_cex/data/BINANCE_BTCUSDT.P, 60.csv" \
  --base-dir "/Volumes/Extreme SSD/trading_data/cex/targets" \
  --dataset "BINANCE_BTCUSDT.P, 60" \
  --output targets.csv \
  --freq 1H \
  --horizons 3 6 12 24 \
  --barriers 0.015:0.01
```

4) Merge lags features with targets (drop rows with NA)

```bash
/Users/noel/projects/trading_cex/venv/bin/python feature_engineering/merge_lag_features_targets.py \
  --features-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60" \
  --targets-dir "/Volumes/Extreme SSD/trading_data/cex/targets/BINANCE_BTCUSDT.P, 60" \
  --training-dir "/Volumes/Extreme SSD/trading_data/cex/training" \
  --dataset "BINANCE_BTCUSDT.P, 60"
```

Output goes to `<training-dir>/<dataset>/merged_lags_targets.csv` (or `.parquet` if chosen).

### Alternative: rich multi-timeframe features

You can build a broader feature set from the same lookbacks using `multi_timeframe_features.py` under different timeframes, then merge.

1) Build 1H-only features file

```bash
/Users/noel/projects/trading_cex/venv/bin/python feature_engineering/build_multi_timeframe_features.py \
  --dataset "BINANCE_BTCUSDT.P, 60" \
  --base-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks" \
  --timeframes 1H \
  --output features_1h.csv
```

2) Build higher-TF features file (4H/12H/1D)

```bash
/Users/noel/projects/trading_cex/venv/bin/python feature_engineering/build_multi_timeframe_features.py \
  --dataset "BINANCE_BTCUSDT.P, 60" \
  --base-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks" \
  --timeframes 4H 12H 1D \
  --output features_4h12h1d.csv
```

3) Build targets (same command as above) then merge, clean, and align

```bash
/Users/noel/projects/trading_cex/venv/bin/python feature_engineering/merge_features_targets.py \
  --features-dir "/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60" \
  --features-1h features_1h.csv \
  --features-multi features_4h12h1d.csv \
  --targets-dir "/Volumes/Extreme SSD/trading_data/cex/targets/BINANCE_BTCUSDT.P, 60" \
  --warmup-rows 720 \
  --training-dir "/Volumes/Extreme SSD/trading_data/cex/training" \
  --dataset "BINANCE_BTCUSDT.P, 60"
```

Notes:
- Defaults drop 720 warmup rows and an EDA-derived list of problematic columns. Use `--no-default-feature-drop` to keep them and `--extra-feature-drop` to drop more.
- Timestamps are normalized and checked; the merger attempts row-order recovery if a targets timestamp column is implausible.

### Script catalog

- build_lookbacks.py: Persist per-timeframe right-closed lookback stores (`lookbacks_1H.pkl`, `lookbacks_4H.pkl`, ...).
- build_current_bar_lag_features.py: Current-bar features (t,t−1) for 1H and higher TFs, plus lag-1..K.
- build_multi_timeframe_features.py: Rich multi-TF feature table from lookbacks using `multi_timeframe_features.py`.
- build_targets.py: Forward-window targets (log/simple returns, MFE/MAE, triple-barrier with configurable tie policy).
- merge_lag_features_targets.py: Merge current-bar-with-lags features with targets on timestamp; drop NA rows.
- merge_features_targets.py: Merge 1H and higher-TF feature files, clean (warmup, column drops), align with targets; write CSV/Parquet.

### Library modules

- current_bar_features.py
  - `compute_current_bar_features(df, timeframe_suffix, include_original)`
  - `compute_current_and_lag_features_from_lookback(lookback, timeframe_suffix, max_lag)`

- multi_timeframe_features.py
  - Large suite of TA, volatility, liquidity, statistical, and entropy features (52+ families)

- targets.py
  - `TargetGenerationConfig`, `generate_targets_for_row`, `extract_forward_window`

- utils.py
  - `get_lookback_window`, `resample_ohlcv_right_closed`, `validate_ohlcv_data`

### Troubleshooting

- If merges have zero intersection, printouts show sample timestamps from each side; check timezone parsing and dataset names.
- If `--ohlcv-1h` is missing or mis-specified, 1H features path will error in `build_current_bar_lag_features.py`.
- NA handling: the rich-features merge path drops warmup rows and optionally all NA rows; you can disable NA-drop with `--no-drop-na` in `merge_features_targets.py`.

### See also

- docs/feature_engineering_methodology.md — architecture and leakage safety
- docs/feature_engineering_lag_methodology.md — current-bar lags across timeframes

