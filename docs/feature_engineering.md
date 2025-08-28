### Implementation Notes (Unified API)
All feature functions follow a unified style to simplify composition and downstream modeling:

- Inputs use individual `pd.Series` (never whole `DataFrame`), with explicit `column_name` where relevant.
- Outputs are `Dict[str, float]` with consistent, column-aware keys, e.g., `{col}_feature_{window}` or `{col}_lag_N`.
- Rolling windows include the current bar and exclude future data; multi-timeframe features must be right-aligned to avoid leakage.

### Basic Transformations and Lags
These involve direct manipulations of the OHLCV data to create time-shifted or normalized versions.

1. **Lagged Prices**: Create lagged versions of Open, High, Low, Close (e.g., Close_{t-1}, Close_{t-5}, Close_{t-20}) to capture historical price levels for comparison to current prices, helping identify deviations from recent means.

2. **Price Differences**: Compute differences like Close - Open (intraday range), High - Low (daily range), or Close - Close_{t-1} (daily change) to measure short-term movements away from equilibrium.

3. **Log Transformations**: Apply log(Open), log(High), log(Low), log(Close), log(Volume) to stabilize variance and make features more normally distributed, which is useful for mean reversion models sensitive to outliers.

4. **Percentage Changes**: Calculate percentage returns like (Close - Close_{t-1}) / Close_{t-1} or log returns log(Close / Close_{t-1}) over various windows (1-day, 5-day, 20-day) to quantify deviations from the mean return.

5. **Cumulative Returns**: Sum of log returns over rolling windows (e.g., 5-day cumulative return) to detect extended deviations that might signal reversion opportunities.

6. **Z-Scores of Prices**: Normalize prices by subtracting the rolling mean and dividing by rolling standard deviation (e.g., z-score of Close over 20 periods) to directly measure how far the price is from its historical mean.

7. **Lagged Volumes**: Similar to prices, lag Volume (e.g., Volume_{t-1}, Volume_{t-10}) to incorporate trading activity history, as unusual volume spikes might precede mean reversion.

8. **Volume Changes**: Compute Volume - Volume_{t-1} or percentage volume change to identify surges that could indicate overreactions leading to reversion.

### Moving Averages and Trend Features
These smooth out noise and highlight deviations from trends, core to mean reversion.

9. **Simple Moving Averages (SMA)**: SMA of Close over various windows (e.g., 5-day, 20-day, 50-day) to establish a "mean" level; features like Close - SMA_20 capture deviations.

10. **Exponential Moving Averages (EMA)**: EMA of Close or Open with different alphas (decay rates) for more weight on recent data; deviations like (Close - EMA_12) / EMA_12 for normalized distance.

11. **Weighted Moving Averages (WMA)**: WMA of High/Low/Close to emphasize recent periods; use as a baseline for reversion signals.

12. **Moving Average Crossovers**: Differences or ratios like SMA_5 - SMA_20, or binary indicators (1 if SMA_5 > SMA_20, else 0) to detect potential reversion points after trend shifts.

13. **Distance to Moving Averages**: Percentage deviation from MA, e.g., (Close - SMA_50) / SMA_50, to quantify overextension.

14. **Moving Average Convergence Divergence (MACD)**: MACD line (EMA_12 - EMA_26 of Close), Signal line (EMA_9 of MACD), and Histogram (MACD - Signal) to capture momentum shifts toward mean reversion.

15. **Moving Average of Volume**: SMA or EMA of Volume over windows like 10-day to normalize current volume and detect anomalies.

### Momentum and Oscillator Features
These identify overbought/oversold conditions, ideal for mean reversion.

16. **Relative Strength Index (RSI)**: RSI over 14 periods (based on average gain/loss from Close differences) to flag extremes (e.g., RSI > 70 for overbought, potential sell for reversion).

17. **Stochastic Oscillator**: %K = (Close - Low_14) / (High_14 - Low_14) * 100, and %D (SMA_3 of %K) over various lookback periods to measure position relative to recent range.

18. **Commodity Channel Index (CCI)**: (Typical Price - SMA_20 of Typical Price) / (0.015 * Mean Deviation), where Typical Price = (High + Low + Close)/3, to detect deviations from the mean.

19. **Rate of Change (ROC)**: (Close - Close_{t-n}) / Close_{t-n} * 100 for n=10,20, etc., to quantify momentum that might revert.

20. **Williams %R**: (High_n - Close) / (High_n - Low_n) * -100 over n periods, similar to stochastic for overextension signals.

21. **Ultimate Oscillator**: Weighted average of three RSIs (7,14,28 periods) to blend short- and long-term momentum for reversion timing.

22. **Money Flow Index (MFI)**: Like RSI but incorporates Volume; uses Typical Price * Volume to detect volume-weighted overbought/oversold.

### Volatility Features
Volatility can indicate uncertainty around the mean, affecting reversion probability.

23. **Historical Volatility**: Standard deviation of log returns over rolling windows (e.g., 20-day vol) to measure dispersion from the mean.

24. **Average True Range (ATR)**: SMA_14 of max(High-Low, |High-Close_{t-1}|, |Low-Close_{t-1}|) to capture volatility ranges for setting reversion thresholds.

25. **Bollinger Bands Features**: Upper/Lower Bands (SMA_20 ± 2*std_20 of Close), Bandwidth (Upper - Lower)/SMA, and %B ( (Close - Lower)/(Upper - Lower) ) to quantify squeezes or expansions signaling potential reversion.

26. **Volatility Ratios**: Ratio of short-term vol (5-day) to long-term vol (50-day) to detect volatility spikes that precede mean snaps back.

27. **Parkinson Volatility**: Based on High-Low ranges, sqrt( (1/(4*ln(2))) * mean( (ln(High/Low))^2 ) ) over windows for intraday vol estimates.

28. **Garman-Klass Volatility**: Incorporates Open, High, Low, Close: sqrt(0.5*(ln(High/Low))^2 - (2*ln(2)-1)*(ln(Close/Open))^2) for more accurate vol from OHLC.

### Volume-Integrated Features
Combine volume with price to add conviction to reversion signals.

29. **On-Balance Volume (OBV)**: Cumulative sum where OBV_t = OBV_{t-1} + Volume if Close > Close_{t-1}, else -Volume; deviations from price trend can signal reversion.

30. **Volume-Weighted Average Price (VWAP)**: Cumulative (Typical Price * Volume) / Cumulative Volume over intraday or multi-day; (Close - VWAP)/VWAP for deviation.

31. **Accumulation/Distribution Line (ADL)**: Cumulative ((Close - Low) - (High - Close)) / (High - Low) * Volume to measure buying/selling pressure relative to mean.

32. **Chaikin Oscillator**: EMA_3 - EMA_10 of ADL to detect momentum in volume flow for reversion.

33. **Volume Rate of Change (VROC)**: (Volume - Volume_{t-n}) / Volume_{t-n} * 100 to spot volume momentum that might lead to price reversion.

### Statistical and Distributional Features
Leverage stats to model the "mean" more robustly.

34. **Rolling Medians and Percentiles**: Median Close over 20 days as a robust mean; or 25th/75th percentiles to create interquartile ranges for deviation measurement.

35. **Kurtosis and Skewness**: Rolling kurtosis/skew of returns over 30 days to detect fat tails or asymmetry indicating higher reversion likelihood.

36. **Autocorrelation Features**: Lag-1 autocorrelation of returns to measure persistence (low autocorr suggests mean reversion).

37. **Hurst Exponent**: Computed via rescaled range analysis on log returns to quantify if the series is mean-reverting (Hurst < 0.5).

38. **Entropy Measures**: Approximate entropy of Close series over windows to gauge predictability and potential for reversion.

### Ratio and Hybrid Features
Combine elements for nuanced signals.

39. **Price-to-Volume Ratios**: Close / Volume or High / Volume_{avg} to normalize price moves by activity.

40. **Candle Body Ratios**: (Close - Open) / (High - Low) to capture intraday sentiment strength for reversion.

41. **Shadow Ratios**: Upper shadow (High - max(Open,Close)) / range, lower shadow similarly, to identify rejection candles signaling reversion.

42. **Typical Price Features**: Lags or MAs of (High + Low + Close)/3 as a smoothed price proxy.

43. **OHLC Averages**: (Open + High + Low + Close)/4 as another price estimator; deviations from its MA.

44. **Volatility-Adjusted Returns**: Log returns / sqrt(ATR) to normalize for risk in reversion contexts.

### Time-Based and Cyclical Features
Incorporate temporal patterns, assuming data has timestamps.

45. **Time of Day/Week Encodings**: If intraday OHLCV, include hour-of-day, day-of-week, day-of-month, month-of-year (optionally one-hot/sin-cos) to capture intra-period effects (e.g., end-of-day reversion).

46. **Rolling Window Statistics**: Min/Max Close over 10 days; ratio of current Close to that min/max for bounded deviation.

47. **Fourier Transforms**: Decompose Close series into frequencies to extract cyclical components (e.g., dominant cycle length) for periodic mean reversion.

48. **Wavelet Features**: Use wavelet decomposition on Close to separate noise from trends at different scales.

### Ensemble and Derived Features
Build on others for complexity.

49. **Principal Component Analysis (PCA) Features**: Apply PCA to a set of lagged prices/returns to reduce dimensionality while capturing variance for ML input.

50. **Interaction Terms**: Products like RSI * Volatility or Z-score * Volume to capture combined effects in ML models.

51. **Binary Threshold Features**: Indicators like 1 if RSI < 30 (oversold, potential buy for reversion), else 0; stack multiple for different indicators.

52. **Rolling Correlations**: Correlation between Close and Volume over windows to detect decoupling that might signal reversion.

These features can be computed using: 
1. rolling windows of various lengths (e.g., 5, 10, 20, 50 periods) to create even more variations. 
2. grouping windows of various timeframe (1h, 4h, 12h, 1d) to create even more variations.

### Advanced Additions (Implemented)
The core now includes several orthogonal features that complement classic TA and improve regime/context awareness for forward PnL prediction:

- **Trend/Breakout**
  - ADX / DMI: Trend strength and directional bias via Wilder smoothing; keys `{col}_adx_{w}`, `{col}_di_plus_{w}`, `{col}_di_minus_{w}`.
  - Aroon Up/Down/Oscillator: Recency of highs/lows over a window; `{col}_aroon_up_{w}`, `{col}_aroon_down_{w}`, `{col}_aroon_osc_{w}`.
  - Donchian Distance/Position: Price position within rolling high–low channel and normalized distances to edges; `{col}_donchian_pos_{w}`, `{col}_donchian_upper_dist_{w}`, `{col}_donchian_lower_dist_{w}`.

- **Volatility (drift-robust estimators)**
  - Rogers–Satchell Volatility: Intraday vol robust to drift using OHLC; `{col}_rs_{w}`.
  - Yang–Zhang Volatility: Combines overnight, intraday, and RS components; `{col}_yz_{w}`.

- **Volume & Liquidity**
  - Relative Volume (RVOL): Current volume vs rolling mean; `{col}_rvol_{w}`.
  - Amihud Illiquidity: Mean(|return| / dollar volume) over window; `{col}_amihud_{w}`.
  - Roll Spread: Bid–ask spread proxy from negative autocovariance of price changes; `{col}_roll_spread_{w}`.
  - Turnover Z-score: Z-score of `close*volume` over window; `turnover_z_{w}`.

- **Standardization & Risk**
  - Return Z-Score: Standardized latest log-return vs recent returns; `{col}_ret_zscore_{w}`.
  - ATR-Normalized Distance: `(current − reference) / ATR` for vol-aware overextension; `{col}_dist_{label}_atr`.
  - Historical VaR / CVaR: Quantile and tail mean of returns; `{col}_var_{pct}_{w}`, `{col}_cvar_{pct}_{w}`.

- **Serial Dependence & Complexity**
  - Ljung–Box p-value: Tests joint autocorrelation up to K lags on recent returns; `{col}_ljung_p_{lags}_{w}`.
  - Permutation Entropy: Ordinal-pattern entropy (0–1) of recent prices; `{col}_perm_entropy_{m}_{w}`.
  - Spectral Entropy: FFT power-spectrum entropy (0–1) of detrended recent prices; `{col}_spectral_entropy_{w}`.
  - OU Half-Life: Mean reversion speed proxy from OU fit; `{col}_ou_halflife_{w}`.

### Notes on Usage
- Prefer a curated subset to reduce collinearity (e.g., choose 1–2 from each family).
- Normalize features where appropriate (percent, z-scores, ATR-normalized) for regime robustness.
- Keep label construction leakage-safe (forward windows shifted, no overlapping if needed) and include fees/slippage when modeling forward PnL.

### Current Bar Features (t and t−1 only)

Only use information from t and t−1. To obtain deeper lags (t−1/t−2, t−2/t−3, …), self-join the table with row shifts and reuse these same base columns.

- Scope and downstream lags
  - Build base features using at most t and t−1.
  - For additional lags, join with `X.shift(k)` and reapply interactions as needed.

- Close log return (t vs t−1) — `close_logret_current_<TF>`
  - Inputs: Close C_t, Close C_{t−1}
  - Formula: `log(C_t / C_{t−1})`

- High–Low range (percent of Open) — `high_low_range_pct_current_<TF>`
  - Inputs: High H_t, Low L_t, Open O_t
  - Formula: `(H_t − L_t) / O_t`

- Close–Open percent (of Open) — `close_open_pct_current_<TF>`
  - Inputs: Close C_t, Open O_t
  - Formula: `(C_t − O_t) / O_t`  (equivalently `C_t / O_t − 1`)

- Log volume (current) — `log_volume_<TF>`
  - Inputs: Volume V_t
  - Formula: `log1p(V_t)`

- Log volume delta (t vs t−1) — `log_volume_delta_current_<TF>`
  - Inputs: Volume V_t, Volume V_{t−1}
  - Formula: `log1p(V_t) − log1p(V_{t−1})`  (≈ `log(V_t/V_{t−1})` if V>0)

- Sign of close log return — `sign_close_logret_current_<TF>`
  - Inputs: `close_logret_current_<TF>`
  - Formula: `sign(close_logret_current_<TF>)` in {−1, 0, 1}

- Interactions (elementwise; names are concatenations `feature1_x_feature2`)
  - Close log return × log volume — `close_logret_current_<TF>_x_log_volume_<TF>`
  - Close log return × log volume delta — `close_logret_current_<TF>_x_log_volume_delta_current_<TF>`
  - Sign(close log return) × log volume — `sign_close_logret_current_<TF>_x_log_volume_<TF>`
  - High–Low range pct × log volume — `high_low_range_pct_current_<TF>_x_log_volume_<TF>`
  - Close–Open pct × log volume — `close_open_pct_current_<TF>_x_log_volume_<TF>`

- Conventions
  - Timeframe suffix: `<TF> ∈ { _1H, _4H, _12H, _1D }` (right-closed resampling for higher TFs to avoid leakage)
  - Only t and t−1 are used in base definitions; deeper lags are produced by self-joins