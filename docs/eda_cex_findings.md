# CEX EDA Findings

**Dataset:** BTCUSDT Perpetual Futures (2021-12 to 2025-10)  
**Target:** `y_logret_24h` (24-hour forward log return)

---

## Premium Index & Funding Rate

### Key Observations

1. **Low Correlations from Raw Features**
   - Raw premium index and funding rate show weak correlations with 24h target
   - Need feature engineering (normalization) to extract signal

2. **Funding Rate ↔ Premium Index Correlation**
   - Highly correlated with each other (same underlying basis dynamics)
   - May have redundancy if using both as features

3. **Z-Score Normalization Improves Signal**
   - **168h window** appears to be a good aggregation period
   - Relative deviation from recent history is more predictive than absolute levels

4. **High/Low Weaker Than Close**
   - Close represents "final consensus," less noisy than intraday extremes

---

## Open Interest & L/S Ratios

### Key Findings

1. **Overall Correlations Weak** (|r| < 0.1)
   - **Best feature: `count_long_short_ratio`** with r ≈ -0.10
   - Higher retail long positioning → lower future returns
   - Other notable: `oi_roc_ema_168h`, `oi_zscore_168h`, `price_lag1h_oi_corr_24h`

2. **Seasonality Effects Boost Correlations**
   | Segment | Effect |
   |---------|--------|
   | Fri/Sat/Sun | Enhanced correlations |
   | Week 1 of month | Stronger relationships |
   | Month segments | Different features peak in different months |

3. **Features Explored**
   - Z-Score OI (24h, 168h, 720h)
   - OI ROC & ROC EMA
   - OI-Price Correlation (concurrent + lagged)
   - OI RSI, OI Volatility
   - L/S Ratios

4. **2nd Derivative Features Show Better Correlation**
   - Acceleration (rate of change of ROC) captures momentum inflection points
   - OI and Price acceleration both show meaningful correlations with forward returns
   - 168h smoothing window works well for reducing noise

5. **OI-Price Interaction Features Show Better Correlation**
   - Divergence and product-based interactions outperform standalone features
   - Notable features with stronger signal:
     - `oi_acceleration_168h`, `price_accel_168h`
     - `oi_price_accel_div_168h` (OI accel - Price accel divergence)
     - `oi_price_momentum_168h` (OI slope × Price slope)
     - `oi_price_divergence_168h` (OI distance - Price distance)
     - `oi_price_accel_product_168h` (OI accel × Price accel)

6. **Volume Interactions Did Not Add Value**
   - Volume × OI and Volume × Price interactions showed no correlation
   - Multiple normalization approaches tried (z-score, percentile) - none worked
   - May be data quality issue or crypto-specific behavior

7. **L/S Ratio Aggregation**
   - Raw L/S ratios are 5-min granularity but target is hourly
   - Tested mean, median, and last for 1h aggregation
   - **Last** works best for hourly aggregation (end-of-hour snapshot)

---

## Notable Feature Calculations (Exact)

**Base Calculations:**
```python
oi_roc_1h = sum_open_interest.pct_change(1)
price_roc_1h = close.pct_change(1)
```

**1. OI Acceleration (signed to handle direction):**
```python
oi_acceleration = np.sign(oi_roc_1h) * oi_roc_1h.diff(1)
oi_acceleration_168h = oi_acceleration.ewm(span=168).mean()
```

**2. Price Acceleration:**
```python
price_acceleration = np.sign(price_roc_1h) * price_roc_1h.diff(1)
price_accel_168h = price_acceleration.ewm(span=168).mean()
```

**3. OI-Price Accel Divergence:**
```python
oi_price_accel_div_168h = oi_acceleration_168h - price_accel_168h
```

**4. OI-Price Momentum (slope product):**
```python
oi_ema_slope_168h = oi.ewm(span=168).mean().pct_change(1)
price_ema_slope_168h = close.ewm(span=168).mean().pct_change(1)
oi_price_momentum_168h = oi_ema_slope_168h * price_ema_slope_168h
```

**5. OI-Price Divergence (distance difference):**
```python
oi_ema_distance_168h = (oi - oi.ewm(168).mean()) / oi.ewm(168).mean()
price_ema_distance_168h = (close - close.ewm(168).mean()) / close.ewm(168).mean()
oi_price_divergence_168h = oi_ema_distance_168h - price_ema_distance_168h
```

**6. OI-Price Accel Product:**
```python
oi_price_accel_product_168h = oi_acceleration_168h * price_accel_168h
```

---

## Unified 1-Hour Dataset

**Output:** `btcusdt_perp_1h_unified_with_features.csv`

| Stat | Value |
|------|-------|
| **Rows** | 34,332 |
| **Date range** | 2021-12-01 02:00 to 2025-10-31 23:00 |
| **All features** | 100% non-null |

### Source Files
- **OHLCV:** `binance_btcusdt_perp_ohlcv.duckdb` → `ohlcv_btcusdt_1h`
- **OI, L/S, Target:** `btcusdt_perp_5m_metrics_with_hour_bucket.csv` (aggregated to 1h using **last**)
- **Premium:** `btcusdt_perp_1h_premium_funding_aligned.csv`

**Source File Generation** (via `align_cex_derivatives_data.py`):

| Output File | Input | Processing |
|-------------|-------|------------|
| `btcusdt_perp_5m_metrics_with_hour_bucket.csv` | `BTCUSDT-metrics-*.csv` (5m) | Added `hour_bucket` column (floor to hour) |
| `btcusdt_perp_1h_premium_funding_aligned.csv` | Premium index (1h) + Funding rate (8h) | Merged, funding rate forward-filled to 1h |

### Data Cleaning
| Issue | Handling |
|-------|----------|
| Zero OI (42 rows) | Replaced with NaN, then forward-filled |
| Missing L/S ratio (478 rows) | Forward-filled |
| Missing premium (990 rows at end) | Forward-filled |
| Pre-2021-12 data (no metrics) | Removed (17,095 rows) |
| Post-2025-10-31 data (stale premium) | Removed (697 rows) |
| Warmup rows (2 rows) | Removed (OI pct_change + diff) |

### Columns
**Base:** `timestamp`, `open`, `high`, `low`, `close`, `volume`, `sum_open_interest`, `count_long_short_ratio`, `y_logret_24h`, `premium_idx_close`

**Derived Features (13):**
- `oi_roc_ema_168h`, `oi_acceleration_24h/168h`, `oi_ema_distance_168h`, `oi_ema_slope_168h`
- `price_accel_24h/168h`
- `oi_price_accel_div_24h/168h`, `oi_price_momentum_168h`, `oi_price_divergence_168h`, `oi_price_accel_product_168h`
- `premium_zscore_168h`

---

## Acceleration Feature Refinement (2025-12-12)

### Key Discovery: Old "Acceleration" Was Actually Volatility

Through mathematical analysis, we discovered that the original `oi_acceleration` feature (calculated as `sign(ROC) × diff(ROC)`) was not measuring true acceleration, but rather **volatility** - specifically, it's mathematically equivalent to the Parkinson volatility estimator

The `sign()` function makes the formula measure the **magnitude** of change in momentum regardless of direction, which is the definition of volatility (dispersion), not acceleration (rate of change of velocity).

### Updated Features

| Feature | Formula | Meaning |
|---------|---------|---------|
| `oi_volatility` | `sign(ROC) × diff(ROC)` | Magnitude of momentum change (like Parkinson) |
| `oi_accel_scaled` | `diff(ROC) / volatility` | True 2nd derivative, normalized by volatility regime |
| `price_volatility` | Same as OI | Price momentum volatility |
| `price_accel_scaled` | Same as OI | Price acceleration, volatility-normalized |

### Correlation Improvement

The scaled acceleration features show **stronger correlations** with forward returns compared to raw acceleration:
- Interactions between OI and price scaled accelerations are more interpretable
- Lead/lag relationships (OI leading price or vice versa) are clearer with normalized features

### Updated Derived Features for further analysis

**Volatility (sign*diff formula):**
- `oi_volatility_24h`, `oi_volatility_168h`

**Scaled Acceleration (accel/volatility):**
- `oi_accel_scaled_24h`, `oi_accel_scaled_168h`
- `price_accel_scaled_24h`, `price_accel_scaled_168h`

**Scaled Accel Interactions:**
- `oi_price_accel_div_24h`, `oi_price_accel_div_168h` (divergence)
- `oi_price_accel_product_168h` (product)
- `oi_accel_vs_price_lag1h`, `price_accel_vs_oi_lag1h` (lead/lag)

**Existing Features from previous analysis:**
- `oi_roc_ema_168h`, `oi_ema_distance_168h`, `oi_ema_slope_168h`
- `oi_price_momentum_168h`, `oi_price_divergence_168h`
- `premium_zscore_168h`

---

## Next Steps
1. Focus on acceleration and divergence features for modeling
2. Explore interaction features (OI × L/S ratio)
3. Consider regime-dependent models (weekend vs weekday)
4. Test shorter forward return horizons (1h, 4h, 12h)
5. Combine with Premium Index features

*Last updated: 2025-12-11*
