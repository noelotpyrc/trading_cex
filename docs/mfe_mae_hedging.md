# Use both MFE and MAE to hedge

## Strategy idea

1. For each 1h bar, we open two equal size long and short
2. For the next 24h, whenever price increase above the expected MFE or decrease below the expected MAE, we close the opposite position and let the winning position run with a trailing take profit.
3. If the price action is within the range of MFE and MAE for the whole 24h, then just close both of them at 24h close

## Simulate the results without backtesting

Since this requires more lower timeframe ohlvc data to check when and how price action cross the MFE or MAE line, we are just approximating the results here with our 1h bar data:

1. For each 1h bar row, check the next 24h bars/rows, record the first bar where ohlc cross MFE or MAE, and record our base cost price based on MFE or MAE.
2. Calculate the expected take profit speed based on how many bars left until 24h bar, for example, if there are 5 bars left, then we expect to reduce our size by 1/5 for each bar of the next 5 bars.
3. Use the next N bars' close price to calculate the expected profit, e.g., (close_n - base_cost_price) * 1/N, and sum them up to get the cumulative profit.
4. In the case of the price reverse back to our base cost price, we are supposed to immediately close all our positions. For this scenario, our profit would be the cumulative profits from the previous bars.

## Implementation

### Loader and alignment
1. Load hourly OHLCV and both MFE/MAE prediction CSVs (val/test).
2. Normalize timestamps to UTC-naive, left-join predictions to OHLCV by timestamp.
3. Validate required columns: pred_qXX bands, exp_ret_avg, y_true.
### Convert predicted returns to price levels
1. Interpret predictions as percent returns relative to entry close (default).
2. Sign conventions and clipping:
    - MFE uses non-negative return: `mfe_r = max(exp_ret_avg_mfe, 0)`
    - MAE uses non-positive return: `mae_r = min(exp_ret_avg_mae, 0)`
3. Price targets per row:
    - `mfe_price = entry_price * (1 + mfe_r)`
    - `mae_price = entry_price * (1 + mae_r)`
### First cross detection over next 24 bars
1. For each start index i, scan bars i+1..i+24:
    1. MFE hit if `high_t >= mfe_price_i` (first hit).
    2. MAE hit if `low_t <= mae_price_i` (first hit).
2. If both hit in the same bar, tie policy is `zero` (PnL = 0 for that entry).
3. Record: first-hit type (MFE/MAE/NONE), hit timestamp, base cost price, remaining bars `R = (i+24) - hit_index`.
### Trailing scale-out PnL approximation
1. If MFE hits first: short leg closed; keep long leg with scale-out over R bars.
    1. Profit path: average of next R closes relative to base cost price, evenly scaled 1/R per bar.
2. If MAE hits first: long leg closed; keep short leg similarly.
3. If neither hits: both closed at bar i+24 close; net PnL ~ 0 ignoring carry.
4. Early reversal: after base cost is set, if price crosses back through base cost, close immediately; PnL equals cumulative up until reversal.
    - After MFE (long): reverse if `low_t <= base_cost`
    - After MAE (short): reverse if `high_t >= base_cost`

### Early reversal missed-PnL bounds
- For early reversals, estimate missed PnL bounds over remaining bars (optimistic vs pessimistic):
    - After MFE (long):
        - Upper bound uses highs: sum of `1/R_total * (high_t - base_cost)`
        - Lower bound uses lows:  sum of `1/R_total * (low_t - base_cost)`
    - After MAE (short):
        - Upper bound uses lows:  sum of `1/R_total * (base_cost - low_t)`
        - Lower bound uses highs: sum of `1/R_total * (base_cost - high_t)`

### Sizing and capital estimate
- Per-leg dollar size `S` (optional; default per-unit if unspecified). Each entry opens 1 long and 1 short of size `S/entry_price` each.
- Peak concurrent entries `H = 24` (when trading every bar with 24-bar horizon).
- Gross peak notional ≈ `2 * S * H`. With leverage `L`, initial margin ≈ `(2 * S * H) / L`. Add buffer (e.g., 25%).
- Per-unit alternative: per-entry notional ≈ `2 * price`; peak ≈ `2 * price * H`; margin ≈ that divided by `L`.