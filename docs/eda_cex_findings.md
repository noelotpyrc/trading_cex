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

---

## Next Steps
1. More OI feature engineering
2. Explore interaction features (OI × L/S ratio)
3. Consider regime-dependent models (weekend vs weekday)
4. Test shorter forward return horizons (1h, 4h, 12h)

*Last updated: 2025-12-10*
