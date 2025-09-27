# Feature Stationarity & Normalization Plan

Target model: `BINANCE_BTCUSDT.P, 60` → run `binance_btcusdt_perp_1h/y_logret_168h`

Source of features: `feature_engineering/build_multi_timeframe_features.py`

Reference for what the current model actually uses: `/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h/feature_importance.csv` (326 features; top signals include EMA12, ADL, VaR/CVaR, autocorr, rolling min/max, MACD, entropy, skew/kurtosis, ATR/HV, BB metrics, RSI, Amihud, OBV, ADX/DI, YZ/RS/GK vol, percentiles, spectral entropy, Donchian, Aroon, etc.).

## Goals

- Reduce distribution drift across train/test/prod by replacing scale‑dependent levels with ratios, differences of stationary transforms, and volatility-normalized measures.
- Prefer dimensionless or bounded features; standardize with rolling z‑scores or robust ranks when appropriate.
- Minimize code churn: reuse existing helpers in `feature_engineering/multi_timeframe_features.py` and add a few small transforms in the builder.

## Normalization Toolkit (preferred order)

1) Ratios/log‑ratios to a local baseline (MA, VWAP, percentile, Donchian range): `log(price / baseline)` or `(price − baseline) / baseline`.
2) Volatility (ATR/HV) normalization for any residual level/diff: `(x − ref) / ATR14`, `value / HV20`.
3) First differences or slopes, normalized: `ΔEMA / EMA`, `ΔOBV / rolling_dollar_volume`.
4) Rolling/EWMA standardization on stationary series: z‑scores, robust z‑scores.
5) Rank/percentile/position‑in‑range for regime invariance.
6) Cyclical encodings for calendar features (sin/cos).

## Family‑by‑Family Plan (patterns map to multiple timeframes: `1H`, `4H`, `12H`, `1D`)

### Moving Averages (SMA/EMA/WMA)
- Patterns: `*_sma_*`, `*_ema_*`, `*_wma_*` (e.g., `close_ema_12_12H`).
- Issue: level features track the BTC price regime → non‑stationary.
- Keep/Prefer:
  - Distance/ratio to MA: already have `close_ma_distance_pct_sma20_*`. Add for EMA12/WMA20.
  - New: `close_log_ratio_ema12_{tf} = log(close / EMA12)`.
  - New: ATR‑normalized distance: `close_dist_ema12_atr_{tf}` using `calculate_atr_normalized_distance`.
- Drop/De‑emphasize: raw `*_sma_*`, `*_ema_*`, `*_wma_*` levels in modeling.

Implementation hook: compute EMA series and last value right after EMA in `feature_engineering/build_multi_timeframe_features.py:196`, then add log‑ratio and ATR‑normalized distance once ATR is computed at `feature_engineering/build_multi_timeframe_features.py:224`.

### MA Crossovers
- Patterns: `*_ma_cross_diff_*`, `*_ma_cross_ratio_*`, `*_ma_cross_signal_*`.
- Issue: `diff` is scale‑dependent.
- Keep: `ratio` (dimensionless), `signal` (binary).
- Normalize/Replace: if retaining `diff`, add `diff / ATR14` and/or `diff / slow_MA`.

### MACD
- Patterns: `*_macd_line_*`, `*_macd_signal_*`, `*_macd_histogram_*`.
- Issue: `line`/`hist` are in price units.
- Normalize:
  - `macd_line / ATR14`, `macd_line / close`.
  - `macd_histogram / ATR14`.
- Keep: `signal` as trend smoother if used with normalization above.

### Bollinger Bands
- Patterns: `*_bb_upper_*`, `*_bb_lower_*`, `*_bb_middle_*`, `*_bb_width_*`, `*_bb_percent_*`.
- Keep: `bb_percent` (bounded 0..1).
- Normalize: replace/augment
  - `bb_width_pct = bb_width / bb_middle` (or `/ close`), `log(bb_width / bb_middle)`.
- Drop: raw `upper/lower/middle` levels.

### Volatility: HV/ATR/RS/GK/YZ
- Patterns: `*_hv_*`, `*_atr_*`, `*_rs_*`, `*_gk_*`, `*_yz_*`.
- Issue: regime‑varying magnitude.
- Normalize: rolling z‑score or divide by a baseline volatility:
  - `hv_z = z(hv_20)`, `atr_z = z(atr_14)`, `yz_over_hv = yz_20 / hv_20`.
  - For cross‑feature normalization, prefer z‑scores computed on returns‑based vols.

### Tail Risk: VaR/CVaR
- Patterns: `*_var_*_*`, `*_cvar_*_*` (e.g., `close_cvar_5_50_*`).
- Note: already in return space; can vary with volatility.
- Normalize: optional z‑score vs. `hv_20`; otherwise keep as is (signed values convey direction of tail risk).

### Percentiles and Range Position
- Patterns: `*_percentile_*_*`, `*_position_in_range_*`, `*_donchian_pos_*`, `*_donchian_*_dist_*`.
- Status: already dimensionless/bounded; keep.
- Optional: z‑score within window if needed.

### Rolling Extremes
- Patterns: `*_rolling_min_*`, `*_rolling_max_*`.
- Issue: level values.
- Replace with distances/ratios:
  - `close / rolling_max − 1`, `rolling_min / close − 1` or use Donchian normalized distances already provided.
- Keep: `position_in_range` and Donchian distances; drop raw min/max.

### RSI/Stochastic/CCI/WilliamsR/UO/MFI
- Patterns: `*_rsi_*`, `*_stoch_*`, `*_cci_*`, `*_williams_r_*`, `*_uo_*`, `*_mfi_*`.
- Status: bounded or scale‑reduced. Keep as is; optionally apply light winsorization.
- Special: `CCI` can be volatile; optionally z‑score within window.

### Volume Features
- Patterns: `volume_sma_*`, `volume_roc_*`, `volume_rvol_*`, `turnover_z_*`.
- Keep: `volume_rvol_*` (dimensionless), `turnover_z_*` (already z‑scored), `volume_roc_*` (pct change).
- Drop/Replace: raw `volume_sma_*` in favor of `rvol` or z‑scored turnover.

### OBV, ADL, Chaikin
- Patterns: `*_obv`, `*_adl`, `*_chaikin_*`.
- Issue: cumulative level grows with history/scale.
- Normalize:
  - OBV: `ΔOBV / rolling_dollar_volume` or OBV divided by `rolling_sum(volume)` or `rolling dollar volume`.
  - ADL: `ADL / rolling_dollar_volume`, `ΔADL / rolling_dollar_volume`.
  - Chaikin (EMA of ADL) is closer to stationary; still divide by ATR or dollar turnover for scale control if needed.

### VWAP and Typical/OHLC Averages
- Patterns: `*_vwap`, `*_typical_price_*`, `*_ohlc_average_*`.
- Issue: price levels.
- Replace with ratios:
  - `log(close / vwap)`.
  - `typical_price / close − 1`, `ohlc_average / close − 1`.

### Entropy, Spectral Entropy, Permutation Entropy
- Patterns: `*_entropy_*`, `*_spectral_entropy_*`, `*_perm_entropy_*`.
- Status: mostly [0,1] bounded. Keep.

### Autocorrelation, Hurst, Ljung–Box p
- Patterns: `*_autocorr_*`, `*_hurst_*`, `*_ljung_p_*`.
- Status: autocorr and p‑values are bounded; keep. Hurst: optionally log/center → `z(hurst)` over long window.

### ADX/DI, Aroon, Donchian, Return Z‑Score
- Patterns: `*_adx_*`, `*_di_plus_*`, `*_di_minus_*`, `*_aroon_*`, `*_donchian_*`, `*_ret_zscore_*`.
- Status: bounded or standardized; keep. If DI levels are noisy across regimes, add mild z‑score.

### Amihud Illiquidity, Roll Spread
- Patterns: `*_amihud_*`, `*_roll_spread_*`.
- Status: scale/lightly heavy‑tailed.
- Normalize: log transform + z‑score: `z(log(amihud))`, `z(roll_spread)`.

### Time/Calendar
- Patterns: `time_hour_of_day`, `time_day_of_week`, `time_day_of_month`, `time_month_of_year`.
- Issue: integer encoding implies false ordinality.
- Replace with cyclical encodings:
  - Hour: `sin(2π·hour/24)`, `cos(2π·hour/24)`.
  - DOW: `sin(2π·dow/7)`, `cos(2π·dow/7)`.
  - Month: `sin(2π·m/12)`, `cos(2π·m/12)`.

### Dominant Cycle
- Patterns: `*_dominant_cycle_length_*`, `*_cycle_strength_*`.
- Issue: length in bars varies by window.
- Normalize: `cycle_len_ratio = dominant_cycle_length / window`, keep `cycle_strength` (already relative power).

## Prioritized Changes (based on top importance)

- Replace raw MA/EMA/WMA levels with ratios and ATR‑normalized distances; add EMA12 variants.
- Normalize MACD line/hist by ATR and/or close.
- Convert BB width to percent of middle; drop upper/lower/middle levels.
- Normalize ADL/OBV by rolling dollar volume (or use their first differences over turnover).
- Switch VWAP and typical/average prices to log‑ratios to close.
- Add cyclical encodings for time features; drop raw ints.
- Z‑score volatility estimators (HV/ATR/YZ/RS/GK) and heavy‑tailed liquidity (Amihud, roll spread).
- Prefer Donchian and position‑in‑range over raw rolling min/max.

## Implementation Hooks (where to add in code)

- EMA/ATR‑normalized distances and log‑ratios:
  - After EMA: `feature_engineering/build_multi_timeframe_features.py:196`.
  - After ATR14: `feature_engineering/build_multi_timeframe_features.py:224` using `calculate_atr_normalized_distance` from `feature_engineering/multi_timeframe_features.py:1269`.

- MACD normalization (after MACD at `feature_engineering/build_multi_timeframe_features.py:208`, and after ATR at `:224`).

- BB width percent (after BB at `feature_engineering/build_multi_timeframe_features.py:225`).

- ADL/OBV normalization: compute rolling dollar volume from `close*volume`, then divide values or their first differences (OBV: `feature_engineering/multi_timeframe_features.py:635`, ADL: `:675`).

- VWAP and typical/average ratios: after `vwap`/`typical_price`/`ohlc_average` (`feature_engineering/build_multi_timeframe_features.py:232`, `:249`, `:250`).

- Volatility z‑scores: reuse `calculate_zscore` for the vol series (`feature_engineering/multi_timeframe_features.py:194`), or implement small helpers in the builder.

- Time cyc encodings: extend `calculate_time_features` in `feature_engineering/multi_timeframe_features.py:956` or add post‑processing in builder.

## Modeling Guidance (keep vs drop)

- Keep: ratios, percentages, bounded oscillators (RSI/Stoch/UO), z‑scores, entropy, autocorr, position‑in‑range/Donchian, return‑based stats, normalized liquidity.
- Drop or strongly de‑emphasize: raw price levels (MA/VWAP/typical/ohlc avg), raw BB upper/lower/middle, raw rolling min/max, unnormalized MACD diff/line, raw OBV/ADL magnitudes, raw volume SMA.

## Monitoring & Validation

- Compute PSI/KL between train and rolling prod windows per feature; alert at PSI > 0.25.
- Use walk‑forward CV; fit any scalers on the training fold only and apply to validation/test.
- Log feature drift and toggle to normalized alternatives when drifted.

## Next Steps

1) Implement minimal new transforms in the builder (EMA ratios, MACD/BB/OBV/ADL normalizers, time cyc encodings) behind flags.
2) Add a pruning list to the modeling stage to drop raw level features listed above.
3) Re‑train and compare OOS; report drift metrics alongside performance.

