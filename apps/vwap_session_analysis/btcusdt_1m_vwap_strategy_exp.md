# Context

Used binance 1m level from 2020 to 2025 to backtest different strategies based on vwap momentum.

VWAP is calculated using rolling window of 15 bars, and once a close crosses the vwap, the strategy started to decide on entry based on different filters, and then different exit conditions were checked.

Fees are set to 0.03% for entry and exit together.

The backtests are run on 2025 data for now.

# 100% Failed attempts

## Simple entry based on cross

The only entry filter used was the cross itself, and the strategy entered at close of the next bar if that next bar is still in the same direction as the cross.

### Different exit conditions

#### Exits based on previous bar's vwap and checkpoints' close 
- checkpoints close is used as hard stops while previous bar's vwap is dynamic stops
- also tried different buffer for stops and multiple touch criteria for dynamic stops
- none of them worked, most of them ended up with nearly 100% pnl loss
- the main issue is:
    - the entry filter is just a non-signal
    - the exit filters is very sensitive to the market noise
    - the trailing stop setup is very hard to implement correctly with a custom simulation environment

# Attempts with improvements (still unprofitable)

## Entry based on cross with momentum confirmation

The main rule is: At bar 5 of cross happens, check different entry filters

#### Filter A: Intrabar returns
- 3 out of 4 initial bars must agree with the cross (1st bar) direction

#### Filter B: Gap widening
- 2 or 3 out of 4 initial bars must show widening gap (gap = close - vwap, gap diff = gap[i] - gap[i-1])

#### Filter C: Cumulative move
- minimum cumulative move in bps from bar 1 open to bar 4 close

#### Filter D: weekend only
- only run on weekend

#### Filter E: cooldown after stop loss
- cooldown for N crosses after stop loss was triggered

## Exit based on checkpoints and stop loss

Checkpoints are used as progressive profit locks, and stop loss is used as hard exit.

### Checkpoints
- Checkpoints are defined as specific bars after entry to check exit conditions
- Exit rules at each checkpoint (check at CLOSE):
   - Rule #1: Not profitable vs entry (with adjustment) → EXIT
   - Rule #2: Price moved against previous checkpoint → EXIT (starts from 2nd checkpoint)
   - Rule #3: Max checkpoint reached → EXIT (forced exit)

### Stop loss
- Stop loss is used as hard exit
- If stop loss is triggered, the strategy will use the cooldown period to prevent quick re-entry

## The results so far

- None of the strategies is profitable
- A/B/C entry filters increase the win pnl percentage but also increase the loss percentage
- D/E filters don't help on win pnl percentage but reduce the loss percentage and overall trading frequency

### Strategy comparison report


| Experiment | Trades | Win Rate | Avg Win | Avg Loss | Avg PnL | Total ROI | Final Capital |
|------------|--------|----------|---------|----------|---------|-----------|---------------|
| **C=5bps, CP=[10,15,20]** | 9,922 | 25.06% | 0.2083% | -0.1096% | -0.0300% | -94.98% | $5,019.27 |
| **C=5bps, CP=[20,30,40]** | 8,832 | 22.17% | 0.2886% | -0.1192% | -0.0288% | -92.36% | $7,642.10 |
| **C=15bps, CP=[10,15,20]** | 3,307 | 22.04% | 0.3301% | -0.1211% | -0.0217% | -51.70% | $48,295.11 |
| **C=15bps, CP=[20,30,40]** | 3,160 | 18.35% | 0.4554% | -0.1255% | -0.0189% | -45.85% | $54,148.89 |
| **Base (Allday, 15bps, CP[20,30,40], Adj-0.5%)** | 3,157 | 18.75% | 0.4509% | -0.1273% | -0.0189% | -45.74% | $54,260.91 |
| **Base + Weekend Only** | 440 | 17.95% | 0.3563% | -0.1274% | -0.0406% | -16.47% | $83,527.14 |
| **Base + No Fee** | 3,157 | 19.73% | 0.4575% | -0.0986% | +0.0111% | **+39.90%** | $139,903.10 |
| **Base + CD 2** | 1,383 | 26.75% | 0.3849% | -0.1702% | -0.0217% | -26.52% | $73,478.19 |
| **Base + CD 5** | 739 | 25.30% | 0.4003% | -0.1710% | -0.0264% | -18.16% | $81,841.97 |
| **Base + CD 10** | 432 | 25.46% | 0.4463% | -0.1673% | -0.0110% | **-4.89%** | $95,108.68 |

### Observations

1. **Profitability**: Fees (0.03%) are the main killer. Only the No Fees setup is profitable (+39.90%).
2. **Loss Reduction**: 
    - **Cooldowns** (CD 10) work best realistically (-4.89%), significantly cutting overtrading.
    - **Weekend Only** is also relatively safe (-16.47%).
    - **Higher C-threshold**: C=15bps performs much better (-51%) than C=5bps (-95%), reducing overtrading significantly (3k trades vs 10k trades).
3. **Discrepancy Resolved**: Base and No Fee now show identical trade counts (3,157).
