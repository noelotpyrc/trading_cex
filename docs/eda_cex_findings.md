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

## Feature Consolidation for Modeling (2025-12-14)

### Removed Features

The following features were **removed** from the final modeling set due to low signal or redundancy:

| Feature | Reason |
|---------|--------|
| `oi_roc_ema_168h` | High correlation with `oi_ema_distance_168h` |
| `oi_ema_slope_168h` | High correlation with `oi_ema_distance_168h` |
| `oi_zscore_168h` | Excluded from final modeling (calculated but not selected) |

### Added Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `oi_zscore_168h` | `(OI - rolling_mean_168h) / rolling_std_168h` | Relative OI positioning vs recent history |
| `oi_price_ratio_spread` | `(OI_EMA_24h / OI_EMA_168h) - (Price_EMA_24h / Price_EMA_168h)` | Short-term trend divergence between OI and price |

### Final OI/Premium Feature Set

The consolidated feature set used in `merge_features_for_modeling.py`:

**Base Features:**
- `count_long_short_ratio` — Retail L/S ratio (aggregated with `last`)
- `premium_idx_close` — Raw premium index close
- `premium_zscore_168h` — Premium z-score (168h rolling)

**OI Volatility & Distance:**
- `oi_volatility_24h`, `oi_volatility_168h` — OI momentum volatility (sign × diff)
- `oi_ema_distance_168h` — Distance from 168h EMA (normalized)

**Scaled Acceleration (accel / volatility):**
- `oi_accel_scaled_24h`, `oi_accel_scaled_168h`
- `price_accel_scaled_24h`, `price_accel_scaled_168h`

**Scaled Accel Interactions:**
- `oi_price_accel_div_24h`, `oi_price_accel_div_168h` — OI accel - Price accel
- `oi_price_accel_product_168h` — OI accel × Price accel
- `oi_accel_vs_price_lag1h`, `price_accel_vs_oi_lag1h` — Lead/lag relationships

**OI-Price EMA Interactions:**
- `oi_price_momentum_168h` — OI slope × Price slope
- `oi_price_divergence_168h` — OI distance - Price distance
- `oi_price_ratio_spread` — Short/long EMA ratio divergence

### Data Pipeline

```
btcusdt_perp_1h_unified.csv
    ↓ add_features_to_unified.py
btcusdt_perp_1h_unified_with_features.csv
    ↓ merge_features_for_modeling.py (combines with technical features)
merged_features_for_modeling.csv
```

---

## Premium & Spot Volume EDA (2025-12-14)

Explored premium index and spot volume features for signal discovery.

### Theory

1. **Premium = Emotion + Liquidity**
   - Premium index captures directional sentiment + liquidity stress
   - Unstable premium (high volatility) indicates market makers pulling back

2. **Spot = Long-term Intent, Perp = Short-term Speculation**
   - Spot volume represents mid/long-term market participants
   - OI and perp volume represent short-term gamblers
   - Divergence between them may signal smart money accumulation

### Promising Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `three_way_divergence` | `spot_vol_zscore - oi_zscore - premium_zscore` | All 3 conflicting = unstable market |
| `imbalance_price_corr_168h` | `rolling_corr(imbalance_zscore, price_roc, 168h)` | Buy/sell imbalance predictive of price? |
| `premium_volatility_24h` | `rolling_std(premium, 24h)` | Short-term liquidity stress |
| `premium_per_oi` | `premium / (OI / OI_ema_168h)` | Overleveraged sentiment |
| `spot_vol_price_corr_168h` | `rolling_corr(spot_vol_roc, price_roc, 168h)` | Spot volume's predictive power |
| `spot_vs_oi_price_corr_48h` | `spot_vol_price_corr_48h - oi_price_corr_48h` | Who's driving price? |

### Key Findings

**1. Three-Way Divergence (0.065):** When spot volume, OI, and premium all diverge, the market is unstable.

**2. Imbalance-Price Correlation (0.054):** Rolling correlation between taker buy/sell imbalance and price changes — captures how well order flow predicts price direction.

**3. Spot vs OI Price Correlation (48h):** Measures whether spot volume or OI is more predictive of price. Positive = spot driving (institutional), negative = OI driving (speculative).

### Feature Calculations

```python
# Three-way divergence
three_way_divergence = spot_vol_zscore_168h - oi_zscore_168h - premium_zscore_168h

# Imbalance-price correlation
imbalance = taker_buy - (spot_vol - taker_buy)  # buy - sell
imbalance_zscore = zscore(imbalance, 168h)
imbalance_price_corr_168h = rolling_corr(imbalance_zscore, close.pct_change(1), 168)

# Spot vs OI price correlation (48h)
spot_vol_price_corr_48h = rolling_corr(spot_vol.pct_change(1), close.pct_change(1), 48)
oi_price_corr_48h = rolling_corr(oi.pct_change(1), close.pct_change(1), 48)
spot_vs_oi_price_corr_48h = spot_vol_price_corr_48h - oi_price_corr_48h
```

### EDA App

Analysis performed in `apps/eda_premium_spot.py`:
```bash
streamlit run apps/eda_premium_spot.py --server.port 8503
```

---

## Candidate Features for Modeling

**Total: 14 stationary features** selected from EDA for forward return prediction.

---

### `three_way_divergence`
**Description:** Measures the divergence between spot volume, open interest, and premium. 

```python
spot_vol_zscore = (spot_vol - spot_vol.rolling(168).mean()) / spot_vol.rolling(168).std()
oi_zscore = (oi - oi.rolling(168).mean()) / oi.rolling(168).std()
premium_zscore = (premium - premium.rolling(168).mean()) / premium.rolling(168).std()
three_way_divergence = spot_vol_zscore - oi_zscore - premium_zscore
```

---

### `premium_volatility_48h`
**Description:** Short-term volatility of the premium index. High premium volatility indicates liquidity stress and unstable funding conditions.

```python
premium_volatility_48h = premium.rolling(48, min_periods=24).std()
```

---

### `spot_vol_price_corr_168h`
**Description:** Rolling 7-day correlation between spot volume changes and price changes. Measures how well spot volume activity predicts price direction.

```python
spot_vol_roc = spot_volume.pct_change(1)
price_roc = close.pct_change(1)
spot_vol_price_corr_168h = spot_vol_roc.rolling(168, min_periods=24).corr(price_roc)
```

---

### `spot_vs_oi_price_corr_48h`
**Description:** Difference between spot volume-price correlation and OI-price correlation. Positive values indicate spot market is driving price (institutional), negative indicates perp market is driving price (speculative).

```python
spot_vol_price_corr_48h = spot_vol_roc.rolling(48).corr(price_roc)
oi_price_corr_48h = oi_roc.rolling(48).corr(price_roc)
spot_vs_oi_price_corr_48h = spot_vol_price_corr_48h - oi_price_corr_48h
```

---

### `imbalance_price_corr_168h`
**Description:** Rolling correlation between taker buy/sell imbalance and price changes. Measures how well order flow imbalance predicts price direction.

```python
imbalance = spot_taker_buy_volume - (spot_volume - spot_taker_buy_volume)  # buy - sell
imbalance_mean = imbalance.rolling(168).mean()
imbalance_std = imbalance.rolling(168).std()
imbalance_zscore = (imbalance - imbalance_mean) / imbalance_std
imbalance_price_corr_168h = imbalance_zscore.rolling(168).corr(price_roc)
```

---

### `spot_dom_vol_24h`
**Description:** 24-hour volatility of spot dominance ratio. High volatility indicates unstable market leadership between spot and perp markets.

```python
spot_dominance = spot_volume / (spot_volume + perp_volume)
spot_dom_vol_24h = spot_dominance.rolling(24, min_periods=12).std()
```

---

### `spot_dom_roc_price_roc_corr_24h`
**Description:** 24-hour rolling correlation between spot dominance rate-of-change and price rate-of-change. Measures whether changes in market leadership correlate with price moves.

```python
spot_dom_roc = spot_dominance.pct_change(1)
price_roc = close.pct_change(1)
spot_dom_roc_price_roc_corr_24h = spot_dom_roc.rolling(24, min_periods=12).corr(price_roc)
```

---

### `spot_dom_roc_oi_roc_corr_48h` / `spot_dom_roc_oi_roc_corr_168h`
**Description:** Rolling correlation between spot dominance rate-of-change and OI rate-of-change. Measures whether spot market leadership changes correlate with perp positioning changes.

```python
spot_dom_roc = spot_dominance.pct_change(1)
oi_roc = oi.pct_change(1)
spot_dom_roc_oi_roc_corr_48h = spot_dom_roc.rolling(48, min_periods=24).corr(oi_roc)
spot_dom_roc_oi_roc_corr_168h = spot_dom_roc.rolling(168, min_periods=24).corr(oi_roc)
```

---

### `trade_size_premium_divergence`
**Description:** Difference between average trade size z-score and premium z-score. Positive values indicate large trades (whales) during low/negative premium (bearish sentiment) — contrarian signal.

```python
avg_trade_size_usd = spot_quote_volume / spot_num_trades
ats_mean = avg_trade_size_usd.rolling(168).mean()
ats_std = avg_trade_size_usd.rolling(168).std()
avg_trade_size_zscore = (avg_trade_size_usd - ats_mean) / ats_std
premium_zscore = (premium - premium.rolling(168).mean()) / premium.rolling(168).std()
trade_size_premium_divergence = avg_trade_size_zscore - premium_zscore
```

---

### `trade_count_lead_price_corr_168h`
**Description:** Rolling correlation between *lagged* trade count changes and current price changes. Tests whether yesterday's trading activity predicts today's price direction.

```python
trade_count_shifted = spot_num_trades.shift(24)  # 24h lag
tc_shifted_roc = trade_count_shifted.pct_change(1)
trade_count_lead_price_corr_168h = tc_shifted_roc.rolling(168, min_periods=24).corr(price_roc)
```

---

### `trade_size_price_corr_168h`
**Description:** Rolling correlation between average trade size changes and price changes. Measures whether larger trades correlate with price direction.

```python
avg_trade_size_usd = spot_quote_volume / spot_num_trades
avg_size_roc = avg_trade_size_usd.pct_change(1)
price_roc = close.pct_change(1)
trade_size_price_corr_168h = avg_size_roc.rolling(168, min_periods=24).corr(price_roc)
```

---

### `trade_count_oi_corr_168h`
**Description:** Rolling correlation between trade count changes and OI changes. Negative correlation (spot activity up, OI down) may indicate spot accumulation while perps unwind — bullish signal.

```python
trade_count_roc = spot_num_trades.pct_change(1)
oi_roc = oi.pct_change(1)
trade_count_oi_corr_168h = trade_count_roc.rolling(168, min_periods=24).corr(oi_roc)
```

---

### `trade_count_spot_dom_corr_168h`
**Description:** Rolling correlation between trade count changes and spot dominance changes. Negative correlation indicates high trading activity when spot market share is declining — potential divergence signal.

```python
trade_count_roc = spot_num_trades.pct_change(1)
spot_dom_roc = spot_dominance.pct_change(1)
trade_count_spot_dom_corr_168h = trade_count_roc.rolling(168, min_periods=24).corr(spot_dom_roc)
```

---

## Status & Next Steps

### Remaining
- [ ] Add candidate features to modeling pipeline
- [ ] Test feature stability across different time periods

*Last updated: 2025-12-15*
