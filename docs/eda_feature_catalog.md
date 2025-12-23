# EDA Feature Catalog

Complete documentation of the 46 features used in EDA and modeling.

---

## 1. Technical Indicators

### 1.1 Volatility Features

- **close_parkinson_20_1H** - Parkinson Volatility
  - **What it measures:** Estimates price volatility using the high-low range of each bar, which captures intrabar price swings more efficiently than just looking at closing prices.
  - **Step-by-step calculation:**
    1. For each bar, calculate `ln(High / Low)` — the natural log of the high-to-low ratio
    2. Square each value: `(ln(High / Low))²`
    3. Take the average of these squared values over the last 20 bars
    4. Multiply by the scaling factor `1 / (4 × ln(2))` ≈ 0.361
    5. Take the square root of the result
  - **Formula:** `sqrt((1 / (4 × ln(2))) × mean((ln(High/Low))², window=20))`
  - **Interpretation:** Higher values = more volatile market. This estimator is ~5x more efficient than close-to-close volatility because it uses the full price range of each bar.

- **close_hv_20_12H** - Historical Volatility
  - **What it measures:** Traditional volatility based on price changes between consecutive bars, annualized for comparison across timeframes.
  - **Step-by-step calculation:**
    1. Calculate log returns: `ln(Close_t / Close_{t-1})` for each bar
    2. Take the standard deviation of these returns over the last 20 bars
    3. Annualize by multiplying by `sqrt(365)` (since crypto trades 365 days/year)
  - **Formula:** `std(ln(Close_t / Close_{t-1}), window=20) × sqrt(365)`
  - **Interpretation:** A value of 0.50 means ~50% annualized volatility. Higher = more price uncertainty.

- **close_atr_14_1D** - Average True Range
  - **What it measures:** Average price movement per bar, accounting for gaps between bars (when today's open differs from yesterday's close).
  - **Step-by-step calculation:**
    1. For each bar, calculate "True Range" as the maximum of:
       - `High - Low` (intrabar range)
       - `|High - Previous Close|` (gap up captured)
       - `|Low - Previous Close|` (gap down captured)
    2. Average these True Range values over the last 14 bars
  - **Formula:** `mean(max(High-Low, |High-PrevClose|, |Low-PrevClose|), window=14)`
  - **Interpretation:** Measured in price units (e.g., $500 ATR means average daily movement is $500). Used to set stop-losses and position sizes.

**References:**
- [calculate_parkinson_volatility](../feature_engineering/multi_timeframe_features.py#L596-L610)
- [calculate_historical_volatility](../feature_engineering/multi_timeframe_features.py#L507-L520)
- [calculate_atr](../feature_engineering/multi_timeframe_features.py#L523-L545)

> **Regime vs. State Note:**
> Current volatility features (window=20) mix "Regime" (absolute level) and "State" (local fluctuation).
> - **Regime Feature:** Add a longer-term anchor, e.g., `close_parkinson_168_1H` (1 week) to define the baseline environment.
> - **State Feature:** Create a ratio `parkinson_20_1H / parkinson_168_1H` to capture relative spikes independent of the background regime.

---

### 1.2 Risk Metrics

- **close_var_5_50_12H** - Value at Risk (5%)
  - **What it measures:** The worst expected return at the 5% probability level — i.e., "95% of the time, losses won't exceed this value."
  - **Step-by-step calculation:**
    1. Calculate log returns for each bar: `ln(Close_t / Close_{t-1})`
    2. Collect the last 50 return values
    3. Find the 5th percentile (5% quantile) of these returns
  - **Formula:** `5th_percentile(log_returns, window=50)`
  - **Interpretation:** A VaR of -0.03 means "In 95% of periods, the return is better than -3%." More negative = higher downside risk.

- **close_cvar_5_50_4H** - Conditional VaR (Expected Shortfall)
  - **What it measures:** The average loss *given that* we're in the worst 5% of outcomes — answers "When things go bad, how bad do they get?"
  - **Step-by-step calculation:**
    1. Calculate log returns for each bar
    2. Find all returns that are at or below the VaR threshold (worst 5%)
    3. Calculate the mean of these "tail" returns
  - **Formula:** `mean(returns where return ≤ VaR_5%)`
  - **Interpretation:** CVaR is always worse (more negative) than VaR. If VaR is -3% and CVaR is -5%, extreme crashes average -5% loss.

**References:**
- [calculate_var_cvar](../feature_engineering/multi_timeframe_features.py#L1184-L1201)

> **Normalization / Tail Shape Note:**
> Raw VaR/CVaR scale with volatility. Use these unitless shape descriptors instead:
> - **Tail Thickness (Kurtosis Proxy):**
>   - *Downside:* `CVaR_05 / VaR_05` (How deep is the crash?)
>   - *Upside:* `CVaR_95 / VaR_95` (How explosive is the pump?)
> - **Robust Skewness:**
>   - Formula: `(VaR_95 + VaR_05) / (VaR_95 - VaR_05)`
>   - Standard "Bowley Skewness" using quantiles.
>   - Positive = Upside potential > Downside risk. Negative = Crash risk dominates.

---

### 1.3 Momentum Oscillators

- **close_rsi_14_12H** - Relative Strength Index
  - **What it measures:** The ratio of recent upward price movements to total price movement, expressed as a value from 0 to 100.
  - **Step-by-step calculation:**
    1. Calculate price changes: `Close_t - Close_{t-1}`
    2. Separate into gains (positive changes) and losses (negative changes as positive values)
    3. Calculate smoothed average gain and loss using Wilder's EMA (α = 1/14)
    4. Compute Relative Strength: `RS = AvgGain / AvgLoss`
    5. Convert to RSI: `RSI = 100 - (100 / (1 + RS))`
  - **Formula:** `100 - 100/(1 + AvgGain/AvgLoss)` where averages use Wilder's smoothing
  - **Interpretation:**
    - RSI > 70: "Overbought" — price has risen rapidly, may reverse
    - RSI < 30: "Oversold" — price has fallen rapidly, may bounce
    - RSI = 50: Balanced buying/selling pressure

- **close_adx_14_12H** - Average Directional Index
  - **What it measures:** The strength of a trend (regardless of direction). Does NOT tell you if the trend is up or down, only how strong it is.
  - **Key concepts:**
    - **+DM (Plus Directional Movement):** Measures upward price movement. Calculated as `High_today - High_yesterday` if positive and greater than downward movement, else 0.
    - **-DM (Minus Directional Movement):** Measures downward price movement. Calculated as `Low_yesterday - Low_today` if positive and greater than upward movement, else 0.
    - **TR (True Range):** The true price range for each bar (see ATR above).
    - **+DI (Plus Directional Indicator):** `(Smoothed +DM / Smoothed TR) × 100` — percentage of movement that is upward
    - **-DI (Minus Directional Indicator):** `(Smoothed -DM / Smoothed TR) × 100` — percentage of movement that is downward
    - **DX (Directional Index):** `|+DI - -DI| / (+DI + -DI) × 100` — normalized difference between up and down movement
  - **Step-by-step calculation:**
    1. Calculate +DM and -DM for each bar
    2. Calculate True Range (TR) for each bar
    3. Smooth +DM, -DM, and TR using Wilder's EMA (α = 1/14)
    4. Calculate +DI and -DI from smoothed values
    5. Calculate DX from +DI and -DI
    6. Smooth DX using Wilder's EMA to get final ADX
  - **Formula:** `ADX = EMA(|+DI - -DI| / (+DI + -DI) × 100, period=14)`
  - **Interpretation:**
    - ADX < 20: Weak/no trend (ranging market)
    - ADX 20-25: Trend may be emerging
    - ADX > 25: Strong trend in progress
    - ADX > 50: Very strong trend

- **close_cum_return_10_1D** - Cumulative Return
  - **What it measures:** Total price movement over the lookback window, expressed as the sum of log returns.
  - **Step-by-step calculation:**
    1. Calculate log returns: `ln(Close_t / Close_{t-1})` for each bar
    2. Sum the last 10 log returns
  - **Formula:** `sum(ln(Close_t / Close_{t-1}), window=10)`
  - **Interpretation:** A value of 0.05 means ~5% gain over the period. Positive = price went up, Negative = price went down.

**References:**
- [calculate_rsi](../feature_engineering/multi_timeframe_features.py#L316-L344)
- [calculate_adx](../feature_engineering/multi_timeframe_features.py#L1087-L1125)
- [calculate_cumulative_returns](../feature_engineering/multi_timeframe_features.py#L123-L143)

> **Interpretation Note:**
> - **RSI (Directional Balance):** Measures "Velocity relative to Volatility". `AvgGain / (AvgGain + AvgLoss)`. High RSI = Up-moves dominate recent volatility. **Stationary [0-100]**.
> - **ADX (Trend Magnitude):** Measures "Directionless Strength". `|+DI - -DI|`. High ADX = Strong trend (Up or Down). **Stationary [0-100]**.
> - **Cumulative Return:** Non-stationary (scales with volatility regime). **REMOVE** or normalize (e.g. `Return / Volatility`).

---

### 1.4 Trend Indicators

- **close_macd_histogram_12_26_9_1D** - MACD Histogram
  - **What it measures:** The momentum of price trends by comparing short-term and long-term moving averages.
  - **Key concepts:**
    - **EMA (Exponential Moving Average):** A weighted average that gives more importance to recent prices. EMA(12) reacts faster than EMA(26).
    - **MACD Line:** The difference between fast and slow EMAs: `EMA(12) - EMA(26)`
    - **Signal Line:** A smoothed version of the MACD Line: `EMA(MACD Line, 9)`
    - **MACD Histogram:** The difference between MACD Line and Signal Line
  - **Step-by-step calculation:**
    1. Calculate 12-period EMA of closing prices (fast EMA)
    2. Calculate 26-period EMA of closing prices (slow EMA)
    3. MACD Line = Fast EMA - Slow EMA
    4. Signal Line = 9-period EMA of the MACD Line
    5. Histogram = MACD Line - Signal Line
  - **Formula:** `Histogram = (EMA(close, 12) - EMA(close, 26)) - EMA(MACD_Line, 9)`
  - **Interpretation:**
    - Histogram > 0: Bullish momentum (MACD above signal)
    - Histogram < 0: Bearish momentum (MACD below signal)
    - Histogram growing: Momentum strengthening
    - Histogram shrinking: Momentum weakening

- **close_ma_cross_diff_5_20_12H** - MA Cross Difference
  - **What it measures:** The gap between short-term and long-term simple moving averages, indicating trend direction and strength.
  - **Step-by-step calculation:**
    1. Calculate 5-period Simple Moving Average (SMA): average of last 5 closing prices
    2. Calculate 20-period SMA: average of last 20 closing prices
    3. Subtract: SMA(5) - SMA(20)
  - **Formula:** `SMA(close, 5) - SMA(close, 20)`
  - **Interpretation:**
    - Positive: Short-term prices above long-term average (uptrend)
    - Negative: Short-term prices below long-term average (downtrend)
    - Crossing zero: Potential trend change (the classic "golden cross" or "death cross")

**References:**
- [calculate_macd](../feature_engineering/multi_timeframe_features.py#L274-L300)
- [calculate_ma_crossovers](../feature_engineering/multi_timeframe_features.py#L235-L256)

> **Recommendation: REMOVE & REPLACE**
> - **Remove:** Raw MACD/MA-Diff (Non-Stationary Dollar Units).
> - **Replace with Normalized Momentum:**
>   - Formula: `Price ROC / Volatility` (e.g., `Return_10 / HV_10`).
>   - *Why:* Acts as a rolling Sharpe Ratio or Z-Score. Measures "Significance of the Move" rather than just raw magnitude. Filters out high-vol noise to better predict genuine trend continuation.

---

### 1.5 Volume Indicators

- **volume_roc_10_12H** - Volume Rate of Change
  - **What it measures:** Percentage change in trading volume compared to N periods ago.
  - **Step-by-step calculation:**
    1. Get current volume: `Volume_t`
    2. Get volume from 10 periods ago: `Volume_{t-10}`
    3. Calculate percentage change: `(Volume_t - Volume_{t-10}) / Volume_{t-10} × 100`
  - **Formula:** `(Volume - Volume_{t-10}) / Volume_{t-10} × 100`
  - **Interpretation:**
    - Positive: Volume increasing vs 10 bars ago
    - Negative: Volume decreasing vs 10 bars ago
    - Large spikes may indicate institutional activity or news events

**References:**
- [calculate_roc](../feature_engineering/multi_timeframe_features.py#L409-L422)

> **Interpretation Note:**
> - **Volume ROC:** compares current volume to N bars ago (e.g., 2pm vs 4am). Inherits strong **intra-day seasonality** (noise).
> - **Recommendation:** Add **Relative Volume (RVOL):** `Volume / Mean(Volume at this time-of-day)`. This isolates abnormal institutional activity from expected daily cycles.

---

### 1.6 Time Features

- **time_day_of_week_1H** - Day of Week
  - **What it measures:** Which day of the week the bar falls on.
  - **Values:** 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
  - **Purpose:** Captures weekly seasonality patterns (e.g., weekends often have lower volume/volatility).

- **time_month_of_year_1H** - Month of Year
  - **What it measures:** Which month of the year the bar falls in.
  - **Values:** 1=January through 12=December
  - **Purpose:** Captures seasonal patterns (e.g., "Sell in May," year-end rallies).

- **time_week_of_month_1H** - Week of Month
  - **What it measures:** Which week within the current month.
  - **Calculation:** `(day_of_month - 1) // 7 + 1`
  - **Values:** 1 to 5
  - **Purpose:** Captures monthly patterns (e.g., options expiration effects, month-end rebalancing).

**References:**
- [calculate_time_features](../feature_engineering/multi_timeframe_features.py#L956-L967)

---

## 2. OI/Price Features

### 2.1 Raw Features

- **count_long_short_ratio** - Long/Short Account Ratio
  - **What it measures:** The ratio of trading accounts holding long positions vs short positions on Binance perpetual futures.
  - **Source:** Binance API endpoint for account position ratios.
  - **Example:** A value of 2.0 means there are twice as many accounts long as short.
  - **Interpretation:**
    - High ratio (>2.0): Crowd is bullish — can be contrarian signal if extreme
    - Low ratio (<1.0): Crowd is bearish
    - Near 1.0: Balanced positioning

- **premium_idx_close** - Premium Index Close
  - **What it measures:** The premium or discount of perpetual futures price vs spot price, used to calculate funding rates.
  - **Source:** Binance API premium index at hour close.
  - **Interpretation:**
    - Positive premium: Perp trading above spot (bullish pressure, longs pay shorts)
    - Negative premium: Perp trading below spot (bearish pressure, shorts pay longs)
    - Near zero: Fair value, balanced market

> **Interpretation Note:** 
> - **Long/Short Ratio:** Regime dependent. Baselines drift significantly (e.g., 2021 Bull Avg=2.0 vs 2022 Bear Avg=0.8). Raw value confuses "Bullish Sentiment" with "Bull Market Regime".
> - **Recommendation:** Use **Z-Score** (`(LS - Mean_168h)/Std_168h`) to measure "Relative Sentiment" (Are traders unusually long *right now*?).
> - **Premium:** Stationarity is generally fine (mean-reverting), but volatility scaling applies. Z-score is preferred (already consistent with Group 2.6).

---

### 2.2 OI Z-Score & Volatility

- **oi_zscore_168h** - OI Z-Score
  - **What it measures:** How unusual current Open Interest is compared to recent history, normalized for comparability.
  - **Key concepts:**
    - **Open Interest (OI):** Total number of outstanding futures contracts (each contract has a buyer and seller).
    - **Z-Score:** Measures how many standard deviations a value is from the mean: `(value - mean) / std`
  - **Step-by-step calculation:**
    1. Calculate rolling 168-hour (7-day) mean of OI
    2. Calculate rolling 168-hour standard deviation of OI
    3. Z-score = (Current OI - Rolling Mean) / Rolling Std
  - **Formula:** `(OI - rolling_mean(OI, 168h)) / rolling_std(OI, 168h)`
  - **Interpretation:**
    - Z > 2: Unusually high OI (heavy speculation, potential volatility ahead)
    - Z < -2: Unusually low OI (quiet market, reduced leverage)
    - Z ≈ 0: Normal OI levels

- **oi_volatility_24h** / **oi_volatility_168h** - OI Volatility
  - **What it measures:** How rapidly OI is changing — the "acceleration" of positioning changes.
  - **Key concepts:**
    - **ROC (Rate of Change):** Percentage change in OI: `(OI_t - OI_{t-1}) / OI_{t-1}`
    - **Acceleration:** Second derivative — rate of change of rate of change: `diff(ROC)`
    - **Signed Acceleration:** `sign(ROC) × diff(ROC)` — positive when acceleration matches direction
  - **Step-by-step calculation:**
    1. Calculate OI rate of change: `oi_roc = OI.pct_change(1)`
    2. Calculate acceleration: `diff(oi_roc)`
    3. Apply sign weighting: `sign(oi_roc) × diff(oi_roc)`
    4. Smooth with EMA: 24h for short-term, 168h for long-term
  - **Formula:** `EMA(sign(oi_roc) × diff(oi_roc), span=24 or 168)`
  - **Interpretation:** High volatility = OI is swinging rapidly (unstable positioning). 24h captures short-term spikes, 168h captures regime changes.

**References:**
- [add_oi_price_features L73-92](../feature_engineering/oneoff_data_work/add_features_to_unified.py#L73-L92)

> **Interpretation Note:**
> - **Regime vs. Spike:** `oi_volatility_168h` defines the "Regime" (Background Noise Level). `oi_volatility_24h` is the "Current State".
> - **Recommendation:** Explicitly engineer `OI_Vol_Ratio = oi_volatility_24h / oi_volatility_168h`.
> - **Why:** Tree models struggle to learn division. Providing the ratio directly allows the model to instantly identify "Spikes" (Ratio > 1.5) regardless of whether the background regime is low (2023) or high (2021).

---

### 2.3 Scaled Acceleration

- **oi_accel_scaled_24h** / **oi_accel_scaled_168h** - Scaled OI Acceleration
  - **What it measures:** OI acceleration normalized by volatility — a "signal-to-noise" ratio for OI momentum.
  - **Key concepts:**
    - **Acceleration:** How fast the rate of change is changing (second derivative)
    - **Scaling by volatility:** Dividing by volatility normalizes the acceleration so a +0.5 in a quiet market equals a +0.5 in a volatile market
  - **Step-by-step calculation:**
    1. Calculate acceleration: `diff(oi_roc)`
    2. Smooth with EMA: `EMA(diff(oi_roc), 24 or 168)`
    3. Divide by volatility: `smoothed_accel / oi_volatility`
  - **Formula:** `EMA(diff(oi_roc), span) / oi_volatility_span`
  - **Interpretation:** Acts like a Sharpe Ratio for OI momentum. Values > 1 = strong signal relative to noise.

- **price_accel_scaled_24h** / **price_accel_scaled_168h** - Scaled Price Acceleration
  - **Same logic as OI, applied to price returns instead.**
  - **Formula:** `EMA(diff(price_roc), span) / price_volatility_span`

**References:**
- [add_oi_price_features L118-135](../feature_engineering/oneoff_data_work/add_features_to_unified.py#L118-L135)

---

### 2.4 OI-Price Interaction Features

- **oi_price_accel_div_24h** / **oi_price_accel_div_168h** - Acceleration Divergence
  - **What it measures:** When OI is accelerating differently than price — potential leading indicator.
  - **Formula:** `oi_accel_scaled - price_accel_scaled`
  - **Interpretation:**
    - Positive: OI accelerating faster than price (traders building positions ahead of move)
    - Negative: Price accelerating faster than OI (price move on existing positions)

- **oi_price_accel_product_168h** - Acceleration Product
  - **What it measures:** Whether OI and price are accelerating in the same direction.
  - **Formula:** `oi_accel_scaled_168h × price_accel_scaled_168h`
  - **Interpretation:**
    - Positive: Both accelerating same direction (aligned momentum, trend strengthening)
    - Negative: Accelerating opposite directions (divergence, potential reversal)
    - Note: `(-)×(-) = (+)` so positive can mean both up or both down

- **oi_accel_vs_price_lag1h** / **price_accel_vs_oi_lag1h** - Lead/Lag Indicators
  - **What it measures:** Whether OI leads price (or vice versa) by comparing current acceleration to 1-hour-lagged acceleration.
  - **Formula:** `current_A_accel - lagged_B_accel`
  - **Interpretation:** If OI consistently leads price, `oi_accel_vs_price_lag1h` will be a good predictor.

**References:**
- [add_oi_price_features L144-161](../feature_engineering/oneoff_data_work/add_features_to_unified.py#L144-L161)

> **Interpretation Note:**
> - **Stationarity:** Scaled Accel `(Accel / Vol)` is a Sharpe Ratio (Signal-to-Noise). Strictly Stationary.
> - **Redundancy (Tree Models):**
>   - **Differences (`Acc_A - Acc_B`):** Redundant. Trees learn linear splits (`A > X` and `B < Y`) easily. Can generally remove Divergence/Lag-Diffs if base features are present.
>   - **Products (`Acc_A * Acc_B`):** **NOT Redundant**. Trees struggle to approximate multiplication. Keep the Product feature (Interaction Strength).
>   - **Product Ambiguity:** `(-)*(-) = (+)` and `(+)*(+) = (+)`. The product measures "Agreement Strength", not Direction. Use base features for direction.

---

### 2.5 EMA-Based Features

- **oi_ema_distance_168h** - OI EMA Distance
  - **What it measures:** How far current OI is from its 7-day exponential moving average, as a percentage.
  - **Key concepts:**
    - **EMA:** Exponential Moving Average — weighted average giving more importance to recent values
    - **Distance as percentage:** Comparing to EMA rather than raw value makes it comparable across time
  - **Formula:** `(OI - EMA(OI, 168)) / EMA(OI, 168)`
  - **Interpretation:**
    - +0.10: OI is 10% above its 7-day EMA (extended, possibly overbought)
    - -0.10: OI is 10% below its 7-day EMA (contracted, possibly oversold)

- **oi_price_momentum_168h** - OI-Price Momentum
  - **What it measures:** Whether OI and price are moving in the same direction (aligned momentum).
  - **Key concepts:**
    - **EMA Slope:** The rate of change of the EMA itself: `pct_change(EMA)`
    - **Product of slopes:** Positive when both rising or both falling
  - **Formula:** `oi_ema_slope × price_ema_slope`
  - **Interpretation:** Positive = OI and price trending together; Negative = diverging trends.

- **oi_price_divergence_168h** - OI-Price Divergence
  - **What it measures:** The difference in how far OI and price have deviated from their respective EMAs.
  - **Formula:** `oi_ema_distance - price_ema_distance`
  - **Interpretation:** Positive = OI is more extended than price; Negative = Price is more extended.

- **oi_price_ratio_spread** - Ratio Spread
  - **What it measures:** Compares the short-term vs long-term EMA ratios for OI and price.
  - **Formula:** `(EMA(OI,24) / EMA(OI,168)) - (EMA(Price,24) / EMA(Price,168))`
  - **Interpretation:** Captures relative momentum divergence between OI and price on different timeframes.

**References:**
- [add_oi_price_features L163-191](../feature_engineering/oneoff_data_work/add_features_to_unified.py#L163-L191)

> **Interpretation Note:**
> - **Stationarity:** All are properly normalized (Percentage dist, Slopes, Ratios). Robust.
> - **Redundancy:**
>   - **Momentum (Product):** **Keep**. Captures non-linear alignment.
>   - **Ratio Spread:** **Keep**. Complex ratio relationship.
>   - **Divergence (Linear Diff):** **Redundant**. Tree can infer `Distance(OI) - Distance(Price)` from base components.

---

### 2.6 Premium Features

- **premium_zscore_168h** - Premium Z-Score
  - **What it measures:** How unusual the current funding premium is compared to recent history.
  - **Step-by-step calculation:**
    1. Calculate rolling 168-hour mean of premium index
    2. Calculate rolling 168-hour standard deviation
    3. Z-score = (Current Premium - Rolling Mean) / Rolling Std
  - **Formula:** `(premium - rolling_mean(premium, 168h)) / rolling_std(premium, 168h)`
  - **Interpretation:**
    - Z > 2: Extremely high funding rate — crowded longs, potential squeeze
    - Z < -2: Extremely negative funding — crowded shorts, potential squeeze
    - Z ≈ 0: Normal funding conditions

- **premium_volatility_48h** - Premium Volatility
  - **What it measures:** How volatile the funding premium has been over the last 48 hours.
  - **Formula:** `rolling_std(premium, 48h)`
  - **Interpretation:** High volatility = funding rates are swinging wildly (market uncertainty, aggressive positioning).

**References:**
- [add_premium_features](../feature_engineering/oneoff_data_work/add_features_to_unified.py#L196-L211)
- [add_spot_features L309-312](../feature_engineering/oneoff_data_work/add_features_to_unified.py#L309-L312)

> **Interpretation Note:**
> - **Z-Score:** **Stationary** (Mean-reverting). Excellent for detecting "Crowded Positioning" extremes.
> - **Volatility:** Regime Dependent (Funding swings are larger in Bull/Bear runs). **Log-transform** or use Ratio (`Vol_48 / Vol_168`) to capture relative "Funding Turbulence".

---

## 3. Spot Features

### 3.1 Spot-Only Features

- **spot_vol_price_corr_168h** - Spot Volume-Price Correlation
  - **What it measures:** How correlated spot volume changes are with price changes over a 7-day window.
  - **Key concepts:**
    - **ROC (Rate of Change):** Percentage change period-over-period
    - **Rolling Correlation:** Correlation coefficient between two series over a moving window
  - **Formula:** `rolling_corr(spot_vol_roc, price_roc, 168h)`
  - **Interpretation:**
    - High positive (>0.5): Volume increases with price (healthy trend confirmation)
    - Negative: Volume increases when price falls (selling pressure, distribution)

- **spot_dom_vol_24h** - Spot Dominance Volatility
  - **What it measures:** How stable the ratio of spot-to-total volume is.
  - **Key concepts:**
    - **Spot Dominance:** `spot_volume / (spot_volume + perp_volume)` — what fraction of volume is spot
  - **Formula:** `rolling_std(spot_dominance, 24h)`
  - **Interpretation:** High volatility = rapid shifts between spot and perpetual trading (regime transitions).

- **spot_dom_roc_price_roc_corr_24h** - Spot Dominance-Price Correlation
  - **What it measures:** Whether changes in spot dominance correlate with price changes.
  - **Formula:** `rolling_corr(pct_change(spot_dominance), price_roc, 24h)`
  - **Interpretation:** Positive = price rises when spot dominance rises (spot buying drives price).

- **imbalance_price_corr_168h** - Imbalance-Price Correlation
  - **What it measures:** How taker buy/sell imbalance correlates with price over 7 days.
  - **Key concepts:**
    - **Imbalance:** `taker_buy_volume - taker_sell_volume` — net buying pressure
    - **Imbalance Z-Score:** Normalized imbalance for comparability
  - **Formula:** `rolling_corr(imbalance_zscore, price_roc, 168h)`
  - **Interpretation:** High positive = buy pressure successfully moves price (healthy market microstructure).

- **trade_size_price_corr_168h** - Trade Size-Price Correlation
  - **What it measures:** Whether average trade size changes correlate with price changes.
  - **Key concepts:**
    - **Average Trade Size:** `quote_volume / num_trades` — average USD per trade
  - **Formula:** `rolling_corr(avg_trade_size_roc, price_roc, 168h)`
  - **Interpretation:** Positive = larger trades when price rises (institutions buying); Negative = larger trades when selling.

- **trade_count_lead_price_corr_168h** - Trade Count Leading Price
  - **What it measures:** Whether trade count 24 hours ago predicts current price changes.
  - **Formula:** `rolling_corr(pct_change(trade_count.shift(24)), price_roc, 168h)`
  - **Interpretation:** Positive = increased trading activity leads to price moves (activity precedes volatility).

**References:**
- [add_spot_features L306-372](../feature_engineering/oneoff_data_work/add_features_to_unified.py#L306-L372)

---

### 3.2 OI-Dependent Spot Features

- **three_way_divergence** - Three-Way Divergence
  - **What it measures:** Combined divergence between spot volume, open interest, and funding premium signals.
  - **Formula:** `spot_vol_zscore - oi_zscore - premium_zscore`
  - **Interpretation:** Large positive = spot volume elevated but OI and premium are not (possible accumulation without leverage).

- **spot_dom_roc_oi_roc_corr_48h** / **spot_dom_roc_oi_roc_corr_168h** - Spot Dominance-OI Correlation
  - **What it measures:** Whether spot dominance changes correlate with OI changes.
  - **Formula:** `rolling_corr(spot_dom_roc, oi_roc, 48h or 168h)`
  - **Interpretation:** Negative = spot dominance rises as OI falls (deleveraging, move to spot).

- **spot_vs_oi_price_corr_48h** - Spot vs OI Price Correlation
  - **What it measures:** Whether spot volume or OI correlates more strongly with price.
  - **Formula:** `spot_vol_price_corr_48h - oi_price_corr_48h`
  - **Interpretation:** Positive = spot volume is the stronger price driver; Negative = OI drives price more.

- **trade_size_premium_divergence** - Trade Size-Premium Divergence
  - **What it measures:** Divergence between average trade size (institutional activity) and funding premium (leverage sentiment).
  - **Formula:** `avg_trade_size_zscore - premium_zscore_168h`
  - **Interpretation:** Positive = large trades without high premium (smart money accumulating quietly).

- **trade_count_oi_corr_168h** - Trade Count-OI Correlation
  - **What it measures:** Whether trade count changes correlate with OI changes.
  - **Formula:** `rolling_corr(trade_count_roc, oi_roc, 168h)`
  - **Interpretation:** Positive = more trades when OI is rising (active position building).

- **trade_count_spot_dom_corr_168h** - Trade Count-Spot Dominance Correlation
  - **What it measures:** Whether trade count changes correlate with spot dominance changes.
  - **Formula:** `rolling_corr(trade_count_roc, spot_dom_roc, 168h)`
  - **Interpretation:** Positive = more trades when spot dominance rises (spot activity driving trade count).

**References:**
- [add_spot_features L376-415](../feature_engineering/oneoff_data_work/add_features_to_unified.py#L376-L415)

---

## 4. Key Source Files

- [multi_timeframe_features.py](../feature_engineering/multi_timeframe_features.py) - 89 technical indicator functions
- [add_features_to_unified.py](../feature_engineering/oneoff_data_work/add_features_to_unified.py) - OI, Premium, and Spot feature derivation
- [merge_duckdb_to_unified.py](../feature_engineering/oneoff_data_work/merge_duckdb_to_unified.py) - Merges 5 DuckDB sources into raw unified file
- [merge_features_for_modeling.py](../feature_engineering/oneoff_data_work/merge_features_for_modeling.py) - Merges technical + OI features for modeling

