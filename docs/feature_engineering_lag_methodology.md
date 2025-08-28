### Lag Feature Methodology (Current Bar Features)

This document specifies how to build lag versions of the Current Bar Features for each timeframe in a leakage-safe way.

#### Definitions
- Current Bar Features: features computed using only bar t and bar t−1 at a given timeframe, per the spec in `docs/feature_engineering.md` under “Current Bar Features (t and t−1 only)”.
- Lookback DataFrame (per timeframe tf): for each base timestamp ts, a right-closed OHLCV window resampled to tf; the last row is the completed tf bar at or before ts; previous rows are earlier completed tf bars.

#### 1H Timeframe (vectorized path)
1. Use the full 1H OHLCV table (one row per base timestamp).
2. Run `compute_current_bar_features(df_1h, timeframe_suffix='1H')` once to get all current-bar features per row.
3. Create lag-N by shifting these feature columns down by N (`.shift(N)`) over the base index.
4. Naming:
   - If a column name contains `_current_`, replace with `_lag_N_`.
   - Otherwise insert `_lag_N` before the timeframe suffix.

#### Higher Timeframes (4H/12H/1D) per-lookback path
For each base timestamp ts:
1. Retrieve the tf lookback DataFrame: `lb = stores[tf]['rows'][ts_key]` (right-closed, label=end).
2. Compute Current Bar Features for every row in `lb` with `compute_current_bar_features(lb, timeframe_suffix=None)`.
3. Extract:
   - Current features: take the last row of the computed feature frame.
   - Lag-k features: take the row `t−k` within the same lookback (if available). Do this for k=1..N.
4. Apply naming rules and attach the timeframe suffix.
5. The output for ts is a single row containing current + lag-1..lag-N current-bar features.

Notes:
- The lookback DataFrame is a normal OHLCV df; step (2) is just a vectorized transform; steps (3-4) are index-based picks and renames.
- This per-ts evaluation ensures no future bars are included. Between higher-TF boundaries, the “current” higher-TF bar remains the same until the boundary completes.

#### Naming Rules (examples)
- `close_logret_current_4H` → `close_logret_lag_1_4H`, `close_logret_lag_2_4H`, ...
- `log_volume_1D` → `log_volume_lag_1_1D`, `log_volume_lag_2_1D`, ...
- Interactions follow the same rule, e.g., `high_low_range_pct_current_12H_x_log_volume_12H` → `high_low_range_pct_lag_1_12H_x_log_volume_lag_1_12H` when sourced entirely from row t−1.

#### Edge Cases
- First N rows (1H) or when a lookback has fewer than N+1 rows (higher TF): lag-k is NaN.
- Division by zero: normalized features using `O_t` denom yield NaN if `O_t == 0`.
- Volume log delta requires both `V_t` and `V_{t−1}` to be finite; otherwise NaN.

#### Integration Points
- Helpers provided in `feature_engineering/current_bar_features.py`:
  - `compute_current_bar_features(df, timeframe_suffix, include_original)` — vectorized over a df.
  - `compute_current_and_lag_features_from_lookback(lookback, timeframe_suffix, max_lag)` — per-lookback, returns a flat dict for current + lag 1..N.
  - (Optional) `build_current_bar_features_from_store(store, timeframe)` — per-tf current-only (no lags) table.

#### Recommended Pipeline
1. 1H: compute once, add shift-lags.
2. 4H/12H/1D: iterate base_index, call `compute_current_and_lag_features_from_lookback` with desired `max_lag`, collect rows.
3. Outer-join all timeframes on base_index to form the final feature matrix.


