# VWAP Momentum Strategy - Buffer & Touch Comparison

## Test Configuration
- **Data**: 2025 BTC/USDT 1-min data
- **VWAP Period**: 15 bars
- **Entry**: Bar 2 OPEN after cross
- **Trailing Stops**: Bars 4, 7, 12
- **Fee**: 0.03% per round-trip
- **Initial Capital**: $100,000
- **Position Size**: $100,000 (constant)

## Summary Results

| Method | Total Trades | Long | Short | Win Rate | Long WR | Short WR | Avg Win | Avg Loss | Total PnL | Final Capital | Avg Bars |
|--------|--------------|------|-------|----------|---------|----------|---------|----------|-----------|---------------|----------|
| 0.03% buffer, single touch | 42,703 | 21,285 | 21,418 | 44.83% | 44.64% | 45.02% | 0.0758% | -0.0681% | -78.67% | $21,330.56 | 4.5 |
| 0.08% buffer, single touch | 32,006 | 15,992 | 16,014 | 51.45% | 51.19% | 51.70% | 0.0811% | -0.0935% | -69.42% | $30,584.12 | 7.0 |
| 0.1% buffer, single touch | 28,566 | 14,273 | 14,293 | 53.53% | 53.30% | 53.75% | 0.0845% | -0.1051% | -65.24% | $34,762.84 | 8.5 |
| 0.1% buffer, double touch | 25,570 | 12,765 | 12,805 | 55.48% | 55.29% | 55.67% | 0.0861% | -0.1075% | -2.95% | $97,049.12 | 10.1 |
| 0.1% buffer, triple touch | 23,948 | 11,936 | 12,012 | 56.87% | 56.76% | 56.98% | 0.0883% | -0.1120% | 46.50% | $146,503.09 | 11.4 |
| No buffer, single touch | 51,360 | 25,595 | 25,765 | 37.93% | 37.92% | 37.93% | 0.0751% | -0.0507% | -78.64% | $21,360.97 | 3.5 |

## Exit Reason Breakdown

| Method | Trailing Exits | Trailing Avg PnL | VWAP Exits | VWAP Avg PnL |
|--------|----------------|------------------|------------|--------------|
| 0.03% buffer, single touch | 27,877 | 0.0467% | 14,826 | -0.0982% |
| 0.08% buffer, single touch | 23,125 | 0.0500% | 8,881 | -0.1434% |
| 0.1% buffer, single touch | 20,957 | 0.0525% | 7,609 | -0.1583% |
| 0.1% buffer, double touch | 19,167 | 0.0527% | 6,403 | -0.1581% |
| 0.1% buffer, triple touch | 18,184 | 0.0533% | 5,764 | -0.1602% |
| No buffer, single touch | 27,521 | 0.0485% | 23,839 | -0.0625% |

## Key Observations

### Best Total PnL
- **0.1% buffer, triple touch**: 46.50%

### Best Win Rate
- **0.1% buffer, triple touch**: 56.87%

### Shortest Average Hold Time
- **No buffer, single touch**: 3.5 bars

### Most Trades
- **No buffer, single touch**: 51,360 trades

## Analysis

### Buffer Impact
Higher buffer reduces false stops but may exit later on real reversals.

### Touch Count Impact
- Single touch: Exits immediately on first stop breach
- Double/Triple touch: Gives position more room to breathe, reduces exit frequency

### Long vs Short Performance
- **0.03% buffer, single touch**: Long 44.6% vs Short 45.0%
- **0.08% buffer, single touch**: Long 51.2% vs Short 51.7%
- **0.1% buffer, single touch**: Long 53.3% vs Short 53.7%
- **0.1% buffer, double touch**: Long 55.3% vs Short 55.7%
- **0.1% buffer, triple touch**: Long 56.8% vs Short 57.0%
- **No buffer, single touch**: Long 37.9% vs Short 37.9%

---
*Report generated: 2025-12-30 23:13:16*