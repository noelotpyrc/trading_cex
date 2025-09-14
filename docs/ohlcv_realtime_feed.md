## Binance 1H OHLCV Semi Real-Time Feed (Design)

### Goals
- **Provide** a semi real-time 1H OHLCV feed for Binance USDT-M Futures.
- **Guarantee** last 30d + buffer hours of clean, continuous, UTC-naive bars for inference.
- **Maintain** a minimal append-only historical store for BTCUSDT 1H.
- **Integrate** with `run/run_inference_lgbm.py` without changing model/feature code.

### High-level pipeline behavior
1. **Hourly pull**: At minute 1–3 each hour, pull the most recent **N** 1H bars from the API (configurable, e.g., N=6).
2. **Historical DB maintenance**: Keep a minimal DuckDB table for BTCUSDT 1H, append-only, one row per hour.
3. **Validation via DB window**: Compare the pulled bars for window `[t-N+1, t-1]` against the last N-1 rows in DB. If any mismatch → fail the DB append for this run (investigate), still persist raw artifacts.
4. **Append only validated new row**: If validation passes, append only the bar at `t` (the latest closed hour) to the DB. Never rewrite previous rows.
5. **Always persist artifacts**: For every pull, write a timestamped CSV snapshot of the pulled API bars and a timestamped "for_inference" CSV assembled from DB (last 30d + buffer), both under the run directory.

### Data source
- **Endpoint**: `fapi/v1/klines` (public, no auth). Example curl:
```bash
curl "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=3"
```
- **Symbol/interval**: Hard coded as 1h for now.
- **Mapping**: use `openTime` as `timestamp` (UTC, naive), keep `open, high, low, close, volume` only.
- **Closed candles only**: Sanity check whether close time stamp is earlier than current time.

### Artifacts and schema
- **Raw API snapshot (per run)**: `{path_for_persistance}/{dataset_slug}/{run_time}_api_pull.csv` containing the N most recent bars.
- **Inference CSV (per run)**: `{path_for_persistance}/{dataset_slug}/{run_time}_for_inference.csv` containing last 30d + buffer bars assembled from DB, aligned to the latest closed hour.
- **Columns (both files)**: `timestamp, open, high, low, close, volume` (lowercase, UTC-naive, ascending).
- Keep separate from backfill/historical data files.

### Historical database
- **Engine**: DuckDB file under a configured path (e.g., `/Volumes/Extreme SSD/trading_data/cex/db/ohlcv.duckdb`).
- **Table (BTCUSDT 1H)**: `ohlcv_btcusdt_1h` (append-only; one row per hour; unique by `timestamp`).
```sql
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_1h (
  timestamp TIMESTAMP PRIMARY KEY,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume DOUBLE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
- **Access patterns**:
  - Read last N-1 rows for validation window.
  - Read last `720 + buffer` rows to build the inference CSV.

### Inference CSV source options (API vs DB)
- **Option A — API (pull 30d + buffer directly for each run)**
  - **Pros**: simplest path; fresh data independent of DB; avoids DB lag; single external source of truth.
  - **Cons**: may hit API limitation; higher latency/failure surface; harder reproducibility without persisting entire snapshot; validation must re-run client-side; potential inconsistencies if API revises history.
- **Option B — DB (assemble from validated history)**
  - **Pros**: deterministic and reproducible; minimal API usage during inference; central validation in feed; simpler, faster inference; easier auditing.
  - **Cons**: depends on DB freshness/health; requires feed to run before inference; additional component to maintain; risk of lag if schedules drift.
- **Recommendation**: default to **DB** for inference CSV; use cache file if DB lag is a concern; support **API** as a fallback mode when DB is unavailable or flagged inconsistent.

### Hourly operation (detailed)
1. Compute `now_floor = now().floor('h')` and `target_hour = now_floor - 1h`.
   - **Example**: If current time is `2024-01-15 14:23:45 UTC`, then:
     - `now_floor = 2024-01-15 14:00:00 UTC` (hour boundary)
     - `target_hour = 2024-01-15 13:00:00 UTC` (the latest complete hour we want to validate/append)
2. Pull **N** most recent bars via API (`limit=N`). Enforce that the last pulled bar has `openTime == target_hour` and its close time < now.
3. Map to the canonical schema (`timestamp=openTime` and OHLCV lowercase), cast numerics, sort ascending.
4. Query DB for the last N-1 rows ending at `target_hour - 1h`.
5. **Validation rules**:
   - Timestamps match exactly between API `[t-N+1 .. t-1]` and DB last N-1 rows.
   - No duplicates; strictly hourly spacing.
   - OHLCV values for `[t-N+1 .. t-1]` match DB exactly (fail-fast on any difference).
6. If validation passes, **append only** the bar at `t = target_hour` into `ohlcv_btcusdt_1h`.
7. Persist artifacts:
   - Write the raw API snapshot to `{run_time}_api_pull.csv`.
   - Build and write `{run_time}_for_inference.csv` by selecting from DB the last `720 + buffer` rows ending at `t` (or from API if `--inference-source=api`).
8. Log concise stats (pulled=N, validated=N-1, appended=1) and any warnings.

### Continuity and validations (DB-focused)
- DB must be strictly continuous hourly, unique `timestamp`, no NaNs in OHLCV.
- On each run, after appending, re-check that the last `720 + buffer` hours in DB have no gaps or duplicates.
- If a validation failure occurs (mismatch with API historical rows):
  - Do not modify DB; persist artifacts; alert/log for investigation.
  - Optionally, a repair workflow can be added later (manual or automated backfill).

### CLI and usage
- New CLI: `run/ohlcv_feed_binance.py`
  - `--n-recent` (default: 6) number of most recent bars to request per run
  - `--duckdb` path to the DuckDB file (for `ohlcv_btcusdt_1h`)
  - `--buffer-hours` (default: 6) for inference CSV assembly
  - `--inference-source` (`db`|`api`, default: `db`) choose source for `{run_time}_for_inference.csv`
  - `--persist-dir` (root path for artifacts)
  - `--dataset` (e.g., `BINANCE_BTCUSDT.P, 60`) used to form `dataset_slug` directory
  - `--once` (default) or `--watch --sleep-sec 300`
  - `--dry-run`, `--debug`
- Scheduling: cron/systemd at minute 1–3 each hour.

### Integration with inference
- Point `--input-csv` in `run/run_inference_lgbm.py` to the latest `{run_time}_for_inference.csv` produced by the feed.
- The inference pipeline already enforces:
  - Minimum history (30d + buffer), strict continuity
  - Lookbacks build and base-window trimming
  - Feature computation and strict model feature validation
  - Prediction + DuckDB/file debug artifacts

### Reliability and limits
- Basic retries with exponential backoff; conservative request cadence (hourly).
- Respect Binance request limits; the volume is small (≤ 1–2 calls per run).
- Clock sanity check vs UTC; warn on large local clock skew.
- Log concise stats: fetched rows, validated rows, appended rows, final coverage.

### Testing
- Unit tests (offline fixtures):
  - API array → DataFrame mapping
  - Closed-candle filter at the hour boundary
  - DB validation logic for `[t-N+1 .. t-1]`
  - Append-only semantics and uniqueness on `timestamp`
  - Continuity validator on DB window
- Integration smoke test:
  - Seed DB with ~31d; run the feed once; assert one new row appended and inference CSV produced; optionally run `run/run_inference_lgbm.py` against it.

### Curl usage (manual checks)
- Ad-hoc validation command:
```bash
curl "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=3"
```

### Milestones
- **M1**: CLI scaffold, N-recent pull, raw artifact write.
- **M2**: DuckDB table + validation `[t-N+1..t-1]` + append-only new row.
- **M3**: Build inference CSV from DB + scheduling docs + integration smoke test.
- **M4**: Optional retention policies and repair/backfill workflow.

### Open questions
- Preferred `N` for cross-validation window (e.g., 6 vs 12)?
- Exact artifact directory structure and rotation policy for `{run_time}_*.csv`.
- Whether to add alerting on validation mismatch or missing bar.
