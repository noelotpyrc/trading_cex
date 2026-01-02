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
- A/B/C entry filters increase the win rate but also increase the loss rate
- D/E filters don't help on win rate but reduce the loss rate and overall trading frequency

### Strategy comparison report


| Experiment | Trades | Win Rate | Avg Win | Avg Loss | Avg PnL | Total ROI | Final Capital |
|------------|--------|----------|---------|----------|---------|-----------|---------------|
| **C=5bps, CP=[10,15,20]** | 13,640 | 31.43% | 0.1947% | -0.1423% | -0.0364% | -99.33% | $673.84 |
| **C=5bps, CP=[20,30,40]** | 6,968 | 33.28% | 0.3014% | -0.2005% | -0.0335% | -90.72% | $9,282.06 |
| **C=15bps, CP=[20,30,40]** | 2,642 | 35.28% | 0.4333% | -0.2693% | -0.0214% | -45.03% | $54,967.79 |
| **Base (Allday, 15bps, CP[20,30,40], Adj-0.5%)** | 2,501 | 44.38% | 0.3913% | -0.3413% | -0.0161% | -35.51% | $64,488.36 |
| **Base + Weekend Only** | 363 | 40.77% | 0.3380% | -0.2844% | -0.0306% | **-10.83%** | $89,169.23 |
| **Base + No Fee** | 2,502 | 48.08% | 0.3899% | -0.3344% | +0.0139% | **+36.65%** | $136,650.17 |
| **Base + SL CD 5** | 739 | 25.30% | 0.4003% | -0.1710% | -0.0264% | -18.16% | $81,841.97 |
| **Base + SL CD 10** | 432 | 25.46% | 0.4463% | -0.1673% | -0.0110% | **-4.89%** | $95,108.68 |
| **Base + SL CD 20** | 235 | 25.96% | 0.3749% | -0.1678% | -0.0269% | -6.24% | $93,760.29 |
