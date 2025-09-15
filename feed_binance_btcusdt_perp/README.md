# Binance BTCUSDT Perp 1H OHLCV Feed

Semi real-time feed and backfill utilities for Binance USDT-M Futures 1h OHLCV. Uses public REST (`fapi/v1/klines`) and DuckDB for a minimal append-only store.

## Conventions
- Timestamps: UTC-naive `TIMESTAMP` in DuckDB; CSVs use ISO without timezone. Internally, mapping is from Binance `openTime` (ms) to `timestamp`.
- Closed candles only: a candle is considered closed if `_close_time <= now_utc.floor('h') - 1ms`.

## Modules
- `feed_binance_btcusdt_perp/api.py`
  - `fetch_klines(symbol, interval, limit)`: pull recent klines (no auth).
  - `klines_to_dataframe(klines)`: map to DataFrame with columns `timestamp, open, high, low, close, volume, _close_time`.
  - `compute_target_hour(now=None)`: returns `(now_floor, target_hour)`.

- `feed_binance_btcusdt_perp/db.py`
  - `ensure_table(db_path)`: creates `ohlcv_btcusdt_1h` if missing.
  - `read_last_n_rows_ending_before(db_path, n, end_exclusive)`.
  - `append_row_if_absent(db_path, row)`: guarded insert by timestamp.
  - `coverage_stats(db_path)`: `(min_ts, max_ts, count)`.

- `feed_binance_btcusdt_perp/validation.py`
  - `validate_window(api_df, db_df, target_hour, tolerance=1e-8)`: validates `[t-N+1..t-1]` overlap by timestamps and OHLCV within tolerance.

- `feed_binance_btcusdt_perp/persistence.py`
  - `PersistConfig(root_dir, dataset_slug)` and `write_raw_snapshot(cfg, run_id, df)`.

- `feed_binance_btcusdt_perp/cli.py` (hourly feed)
  - Runs one cycle: pull recent, filter closed, validate overlap, append `t` (latest closed hour). Always writes a raw snapshot CSV.
  - `--catch-up`: validates overlap and appends all missing closed rows in the API window (multi-row catch-up).

- Backfill helpers
  - `feed_binance_btcusdt_perp/backfill_ohlcv_binance_1h_from_csv.py`: backfill large history from a merged Binance Vision CSV (inspect, clean, validate, insert-only-missing).
  - `feed_binance_btcusdt_perp/backfill_recent_from_api.py`: backfill recent hours from the API; validates overlap vs DB and appends only new rows.

## Typical Workflows

### 1) Initial backfill from Binance Vision
1. Download monthly ZIPs: `utils/download_binance_monthly_klines.py` (use `--interval 1h`).
2. For the current partial month, download daily ZIPs: `utils/download_binance_daily_klines.py`.
3. Merge ZIPs into a single CSV: `utils/merge_binance_klines.py` (point `--in-dir` to the folder with ZIPs).
4. Backfill DB from the merged CSV:
   - `python feed_binance_btcusdt_perp/backfill_ohlcv_binance_1h_from_csv.py --csv "/path/to/merged.csv" --duckdb "/path/to/ohlcv.duckdb" --stop-on-gap`

### 2) Backfill “today so far” from API
- `python feed_binance_btcusdt_perp/backfill_recent_from_api.py --duckdb "/path/to/ohlcv.duckdb" --n-recent 48 --db-validate-rows 72`
- Validates overlapping rows vs DB and appends any closed rows newer than the DB max.

### 3) Hourly feed (one closed hour)
- Dry-run first:
  - `python -m feed_binance_btcusdt_perp.cli --duckdb "/path/to/ohlcv.duckdb" --persist-dir "/path/to/feed_artifacts" --dataset binance_btcusdt_perp_1h --n-recent 12 --dry-run --debug`
- Write to DB:
  - `python -m feed_binance_btcusdt_perp.cli --duckdb "/path/to/ohlcv.duckdb" --persist-dir "/path/to/feed_artifacts" --dataset binance_btcusdt_perp_1h --n-recent 12`

### 4) Catch up multiple missing hours
- `python -m feed_binance_btcusdt_perp.cli --duckdb "/path/to/ohlcv.duckdb" --persist-dir "/path/to/feed_artifacts" --dataset binance_btcusdt_perp_1h --n-recent 48 --catch-up`
- Validates overlap, then appends all closed rows strictly after the DB max timestamp.

## Flags (feed CLI)
- `--n-recent`: recent bars to request (recommend 12–48).
- `--persist-dir` and `--dataset`: where snapshots like `{run_id}_api_pull.csv` are written.
- `--dry-run`: no DB writes; still persists raw snapshot.
- `--catch-up`: append all missing rows in the API window.
- `--debug`: verbose logging.

## Testing and E2E Checks
- Offline unit tests:
  - `python feed_binance_btcusdt_perp/tests/test_api.py`
  - `python feed_binance_btcusdt_perp/tests/test_db.py`
  - `python feed_binance_btcusdt_perp/tests/test_validation.py`
  - `python feed_binance_btcusdt_perp/tests/test_persistence.py`
  - `python feed_binance_btcusdt_perp/tests/test_cli.py` (dry-run path)
  - `python feed_binance_btcusdt_perp/tests/test_cli_catchup.py` (multi-row catch-up)
- Live API E2E (optional):
  - `python feed_binance_btcusdt_perp/tests/test_api_e2e.py --run`
  - Prints which row would be considered the last closed bar relative to `now_utc`.

## Notes and Caveats
- Binance API last row is often the currently forming candle; the feed filters to closed candles only.
- Numeric comparisons use a small absolute tolerance to avoid float round-off mismatches; consider DECIMAL columns if strict equality is required.
- The DB schema is append-only; inserts are guarded by a unique index on `timestamp` and `INSERT ... WHERE NOT EXISTS` patterns.

