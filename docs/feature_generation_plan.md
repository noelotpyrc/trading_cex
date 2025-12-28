# Feature Generation Plan

Based on derived_features.py window comments.

## State Variables

- `price_vwap_distance_zscore_24_168`
- `price_vwap_distance_zscore_168_168`
- `price_vwap_distance_zscore_720_168`
- `price_ema_distance_zscore_24_168`
- `price_ema_distance_zscore_24_720`
- `price_roc_over_volatility_24_24`
- `price_roc_over_volatility_24_168`
- `price_roc_over_volatility_24_720`

## Momentum Persistence (ρ)

- `return_autocorr_48`
- `return_autocorr_168`
- `variance_ratio_24_48`
- `variance_ratio_24_168`
- `variance_ratio_24_720`
- `oi_price_accel_product_168`
- `oi_price_momentum_168`
- `taker_imb_cvd_slope_24`
- `taker_imb_cvd_slope_168`
- `taker_imb_zscore_168`
- `relative_volume_7`
- `relative_volume_14`
- `relative_volume_30`
- `trade_count_lead_price_corr_24_168`

## Mean-Reversion Strength (γ)

- `pullback_slope_ema_24_48`
- `pullback_slope_ema_24_168`
- `pullback_slope_ema_168_48`
- `pullback_slope_ema_168_168`
- `pullback_slope_ema_720_48`
- `pullback_slope_ema_720_168`
- `pullback_slope_vwap_24_48`
- `pullback_slope_vwap_24_168`
- `pullback_slope_vwap_168_48`
- `pullback_slope_vwap_168_168`
- `pullback_slope_vwap_720_48`
- `pullback_slope_vwap_720_168`
- `mean_cross_rate_ema_24_48`
- `mean_cross_rate_ema_24_168`
- `mean_cross_rate_ema_168_48`
- `mean_cross_rate_ema_168_168`
- `mean_cross_rate_ema_720_48`
- `mean_cross_rate_ema_720_168`
- `mean_cross_rate_vwap_24_48`
- `mean_cross_rate_vwap_24_168`
- `mean_cross_rate_vwap_168_48`
- `mean_cross_rate_vwap_168_168`
- `mean_cross_rate_vwap_720_48`
- `mean_cross_rate_vwap_720_168`
- `oi_zscore_168`
- `oi_ema_distance_zscore_24_168`
- `oi_ema_distance_zscore_168_168`
- `oi_ema_distance_zscore_720_168`
- `premium_zscore_168`
- `long_short_ratio_zscore_48`
- `long_short_ratio_zscore_168`
- `spot_vol_zscore_168`
- `avg_trade_size_zscore_48`
- `avg_trade_size_zscore_168`
- `taker_imb_price_corr_168`
- `avg_trade_size_price_corr_168`

## Regime Indicators

- `efficiency_avg_24`
- `efficiency_avg_168`
- `vol_ratio_24_168`
- `vol_ratio_24_720`
- `oi_volatility_168`
- `oi_vol_ratio_24_168`
- `cvar_var_ratio_168`
- `cvar_var_ratio_720`
- `tail_skewness_168`
- `tail_skewness_720`
- `premium_vol_ratio_24_48`
- `premium_vol_ratio_24_168`
- `spot_dominance_zscore_168`
- `spot_dom_vol_ratio_24_168`

## Interactions

- `displacement_speed_product_168_24`
- `displacement_speed_product_168_48`
- `displacement_speed_product_720_24`
- `displacement_speed_product_720_48`
- `range_chop_interaction_24`
- `range_chop_interaction_168`
- `range_stretch_interaction_168_24`
- `range_stretch_interaction_168_168`
- `range_stretch_interaction_720_24`
- `range_stretch_interaction_720_168`
- `scaled_acceleration_24`
- `scaled_acceleration_168`
- `oi_price_ratio_spread_24_168`
- `spot_vol_price_corr_168`
- `oi_vol_price_corr_168`
- `spot_dom_price_corr_24`
- `spot_dom_price_corr_168`
- `spot_dom_oi_corr_24`
- `spot_dom_oi_corr_168`
- `trade_count_oi_corr_168`
- `trade_count_spot_dom_corr_168`
- `relative_amihud_168`
- `oi_volume_efficiency_24_168`
- `oi_volume_efficiency_signed_pos_48_168`
- `oi_volume_efficiency_signed_neg_48_168`
- `oi_volume_efficiency_signed_pos_168_168`
- `oi_volume_efficiency_signed_neg_168_168`

## Primitives (from ESF KEEP list)

- `parkinson_volatility_24`
- `parkinson_volatility_168`
- `historical_volatility_24`
- `historical_volatility_168`
- `rsi_24`
- `rsi_168`
- `rsi_720`
- `adx_24`
- `adx_168`
- `adx_720`

## Time Features (cyclical encoding)

- `hour_of_day_sin`
- `hour_of_day_cos`
- `day_of_week_sin`
- `day_of_week_cos`
- `week_of_month_sin`
- `week_of_month_cos`
- `month_of_year_sin`
- `month_of_year_cos`

## Total: 117 features

---

# Features by Data Source

## 1. Perp OHLCV Only

Features using only perpetual open/high/low/close/volume:

- `price_vwap_distance_zscore_*` (3) keep
- `price_ema_distance_zscore_*` (2) remove
- `price_roc_over_volatility_*` (3) remove
- `return_autocorr_*` (2) keep
- `variance_ratio_*` (3) keep
- `relative_volume_*` (3) keep (30)
- `pullback_slope_ema_*` (6) remove
- `pullback_slope_vwap_*` (6) keep
- `mean_cross_rate_ema_*` (6) keep
- `mean_cross_rate_vwap_*` (6) remove
- `efficiency_avg_*` (2) keep
- `vol_ratio_*` (2) keep
- `cvar_var_ratio_*` (2) keep
- `tail_skewness_*` (2) remove
- `displacement_speed_product_*` (4) remove
- `range_chop_interaction_*` (2) remove
- `range_stretch_interaction_*` (4) keep
- `scaled_acceleration_*` (2) keep (168)
- `relative_amihud_168` keep
- `parkinson_volatility_*` (2) keep (168)
- `historical_volatility_*` (2) remove
- `rsi_*` (3) keep
- `adx_*` (3) keep
- `hour_of_day_*`, `day_of_week_*`, `week_of_month_*`, `month_of_year_*` (8) keep

**Count: ~76 features**

## 2. Open Interest & Long/Short Ratio

Features using `sum_open_interest`, `long_short_ratio`, `long_account`, `short_account`:

- `oi_price_accel_product_168`
- `oi_price_momentum_168`
- `oi_zscore_168`
- `oi_ema_distance_zscore_*` (3)
- `long_short_ratio_zscore_*` (2)
- `oi_volatility_168`
- `oi_vol_ratio_24_168`
- `oi_price_ratio_spread_24_168`
- `oi_vol_price_corr_168`
- `oi_volume_efficiency_24_168`
- `oi_volume_efficiency_signed_*` (4)

**Count: ~20 features**

## 3. Spot Volume & Num Trades

Features using `spot_volume`, `spot_num_trades`, `spot_taker_buy_volume`:

- `taker_imb_cvd_slope_*` (2)
- `taker_imb_zscore_168`
- `taker_imb_price_corr_168`
- `spot_vol_zscore_168`
- `avg_trade_size_zscore_*` (2)
- `avg_trade_size_price_corr_168`
- `trade_count_lead_price_corr_24_168`
- `spot_dominance_zscore_168`
- `spot_dom_vol_ratio_24_168`
- `spot_vol_price_corr_168`
- `spot_dom_price_corr_*` (2)
- `trade_count_spot_dom_corr_168`
- `spot_dom_oi_corr_*` (2)
- `trade_count_oi_corr_168`

**Count: ~15 features**

## 4. Premium Index

Features using `premium_idx_*`:

- `premium_zscore_168`
- `premium_vol_ratio_*` (2)

**Count: 3 features**

---

**Note:** Some features use multiple data sources (e.g., `spot_dom_oi_corr` uses both spot and OI). Listed under primary source.

---

# Known Data Gaps (2022-01-27 to 2025-08-31)

## Raw Data Gaps

| Data Type | Period | Duration | Issue |
|:----------|:-------|:---------|:------|
| OI + L/S | 2024-02-16 14:00 → 22:00 | 9h | NaN |
| L/S Ratio | 2023-11-11 21:00 | 1h | Zero |
| L/S Ratio | 2023-11-23 03:00 | 1h | Zero |
| L/S Ratio | 2025-07-21 16:00 | 1h | Zero |
| Spot | 2023-03-24 13:00 | 1h | NaN |

## Downstream Feature Impact

### OI Gap (9h) → 176-344 NaN rows

- `oi_volume_efficiency_signed_*_168_168`: 344 NaN (1.09%)
- `oi_volume_efficiency_*_48_168`: 224 NaN (0.71%)
- `oi_volume_efficiency_24_168`: 200 NaN (0.64%)
- `oi_zscore_168`: 176 NaN (0.56%)
- `oi_ema_distance_zscore_*`: 176 NaN (0.56%)
- `long_short_ratio_zscore_168`: 176 NaN (0.56%)
- `long_short_ratio_zscore_48`: 56 NaN (0.18%)

### Spot Gap (1h) → 2-168 NaN rows

- `spot_vol_zscore_168`: 168 NaN (0.53%)
- `taker_imb_zscore_168`: 168 NaN (0.53%)
- `spot_dominance_zscore_168`: 168 NaN (0.53%)
- `avg_trade_size_zscore_168`: 168 NaN (0.53%)
- `taker_imb_price_corr_168`: 83 NaN (0.26%)
- `avg_trade_size_zscore_48`: 48 NaN (0.15%)
- `taker_imb_cvd_slope_*`: 2 NaN (0.01%)

---

## Gap Mitigation Strategies

To reduce NaN propagation from raw data gaps, we use relaxed `min_periods`:

| Function | min_periods | Tolerates |
|:---------|:------------|:----------|
| `zscore()` | 94% of window | ~10h gap for 168-window, ~3h for 48-window |
| `normalize_ratio()` | 60% of window | ~68h gap for 168-window |

This allows rolling calculations to produce values even with partial windows, significantly reducing NaN from short data gaps.

---

## Updated Statistics (After Mitigation)

| Feature | NaN Rows | % |
|:--------|----------:|--:|
| `long_short_ratio_zscore_48` | 53 | 0.17% |
| `oi_volume_efficiency_*` (5 features) | 18 each | 0.06% |
| `oi_zscore_168`, `oi_ema_distance_*` | 9 each | 0.03% |
| `long_short_ratio_zscore_168` | 9 | 0.03% |
| `spot_*_zscore_168` (4 features) | 1 each | 0.00% |
| `taker_imb_cvd_slope_*` | 2 | 0.01% |

### Summary

- **99 features** with 0 NaN ✅
- **18 features** with minimal NaN
- **69 rows** (0.22%) have any feature NaN
- **31,420 rows** (99.78%) are completely clean


