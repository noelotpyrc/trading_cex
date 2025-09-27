# Goal

We want to build a Hidden Markov Model for regime detection, before that, we need to do some essential feature engineering to transform the OHLCV data.

### Essential Feature Engineering for HMM on OHLCV Data

These are the core transformations and features recommended for most HMM applications in financial time series. They focus on achieving stationarity, capturing key market dynamics (e.g., returns, risk, liquidity), and preparing multivariate observations. Skipping these can lead to poor model fit or unstable regimes.

1. **Log Returns**: Compute the logarithmic difference in closing prices: `log_returns = np.log(df['Close'] / df['Close'].shift(1))`. This makes the series stationary by focusing on relative changes, which is crucial for HMM emission distributions. Handle the initial NaN by dropping it.

2. **Volatility Measures**: Estimate intra-period volatility to detect high/low volatility regimes.
   - Parkinson Volatility: `vol = np.sqrt((1 / (4 * np.log(2))) * (np.log(df['High'] / df['Low']) ** 2))`. Uses High/Low for a robust, bias-reduced estimate.

3. **Volume Transformations**: Normalize or difference volume to account for trends and capture liquidity shifts.
   - Log Volume Change: `vol_change = np.log(df['Volume'] / df['Volume'].shift(1))`. Stationarizes raw volume, highlighting spikes or drops.

4. **Standardization/Scaling**: Apply z-score normalization across features: `from sklearn.preprocessing import StandardScaler; scaled_obs = StandardScaler().fit_transform(observations)`. Ensures features have similar scales, preventing dominance by high-variance ones like volume in multivariate HMM.

5. **Multivariate Observation Matrix**: Stack essential features into a 2D array for HMM input, e.g., `observations = np.column_stack([log_returns, vol, vol_change])`. This allows modeling joint distributions of returns, volatility, and volume.

### Optional Feature Engineering

These enhance the model for specific use cases (e.g., more regimes or complex markets) but aren't always necessary. Add them iteratively if basic features yield poor log-likelihood or uninterpretable states. Test for multicollinearity.

1. **Percentage Returns**: As an alternative or complement to log returns: `pct_returns = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)`. Useful if you prefer arithmetic over geometric interpretations.

2. **True Range**: A broader volatility proxy: `tr = np.maximum(df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))`. Captures overnight gaps in hourly data.

3. **Normalized Volume**: Rolling z-score: `vol_normalized = (df['Volume'] - df['Volume'].rolling(window=20).mean()) / df['Volume'].rolling(window=20).std()`. Better for non-log transformations or when absolute levels matter.

4. **Technical Indicators**: Short-term signals for trend/momentum regimes.
   - Moving Average Crossover: Difference like `close_minus_ema = df['Close'] - df['Close'].ewm(span=10).mean()`.
   - RSI (Relative Strength Index): For overbought/oversold detection, using libraries like TA-Lib if available.

5. **Detrending or Differencing**: For persistent trends: Apply first differences to non-stationary features (e.g., `diff_close = df['Close'].diff()`) or use filters like Hodrick-Prescott (from statsmodels).

6. **OHLC-Derived Features**: Body/Shadow ratios, e.g., candle body: `body = abs(df['Close'] - df['Open'])`, or wick ratios for sentiment in regimes.

7. **Aggregation for High-Frequency Data**: For your 1h data, optionally resample to 4h/daily aggregates (e.g., via `df.resample('4H').agg({'Open': 'first', 'High': 'max', ...})`) to reduce noise before feature computation.

Always validate with stationarity tests (e.g., ADF from statsmodels) and monitor HMM fit metrics. For your 3-year dataset, start with essentials to avoid overfitting.