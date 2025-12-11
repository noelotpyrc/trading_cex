# CEX Data Cleaning & Transformation

## Original Data Sources

| File | Granularity | Date Range | Columns |
|------|-------------|------------|---------|
| `BTCUSDT-1h-premium-index-2020-01-2025-11.csv` | 1-hour | 2020-01 to 2025-11 | open_time, open, high, low, close, volume, ... |
| `BTCUSDT-funding-rate-2020-01-2025-11.csv` | 8-hour | 2020-01 to 2025-11 | calc_time, funding_interval_hours, last_funding_rate |
| `BTCUSDT-metrics-2021-12-2025-11.csv` | 5-min | 2021-12 to 2025-11 | create_time, symbol, sum_open_interest, sum_open_interest_value, L/S ratios |

**Location:** `/Volumes/Extreme SSD/trading_data/cex/ohlvc/`

## Derived Files

| File | Source | Transformation |
|------|--------|----------------|
| `btcusdt_perp_1h_premium_funding_aligned.csv` | Premium + Funding | Forward-filled funding to hourly |
| `btcusdt_perp_5m_metrics_with_hour_bucket.csv` | Metrics | Added `hour_bucket` column |
| `btcusdt_perp_5m_metrics_with_ohlc.csv` | Metrics + DuckDB OHLC | Merged 1h OHLC on hour_bucket |

---

## Transformations

### 1. Timestamp Normalization
- All timestamps converted to **naive UTC** (consistent across sources)
- Parse from Unix ms or datetime strings as needed

### 2. Granularity Alignment
- 5-min metrics get `hour_bucket = timestamp.dt.floor('h')` for joining with 1h data
- Hourly features computed on hourly-aggregated data, then merged back to 5-min

### 3. Reindexing (Fill Missing Hours)
```python
full_hours = pd.date_range(start=min, end=max, freq='h')
hourly = hourly.reindex(full_hours)
hourly['sum_open_interest'] = hourly['sum_open_interest'].ffill()
```
- Ensures continuous hourly index for rolling/pct_change calculations
- Forward-fill assumes "OI held constant during gap"

---

## Edge Cases & Cleaning

### 1. Inf/NaN from pct_change
- **Cause:** Division by zero when previous value is 0
- **Fix:** `pct.replace([np.inf, -np.inf], np.nan)`

### 2. Extreme ROC Values (>100%)
- **Cause:** Data quality issues or gaps causing apparent 100%+ changes
- **Fix:** `roi.where((roc > -1) & (roc < 1), np.nan)`

### 3. RSI Division by Zero
- **Cause:** Zero loss over period → division by zero
- **Result:** RSI = 100 or NaN (handled by inf replacement)

### 4. Correlation with Low N
- **Fix:** Require minimum 30 observations for correlation calculation
- Return NaN if insufficient data

---

## Feature Generation Scripts
- `align_cex_derivatives_data.py` - Aligns premium/funding to hourly, adds hour_bucket to metrics
- `add_ohlc_to_metrics.py` - Merges OHLC from DuckDB into metrics on hour_bucket
- `eda_metrics.py` - Computes OI features (z-score, ROC, EMA, correlations) on-the-fly

*Last updated: 2025-12-10*
