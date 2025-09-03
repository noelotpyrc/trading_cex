# Context

We have a ML model generating predicted log returns for different quantiles: 
q05, q10, q15, q25, q50, q75, q85, q90, q95.

We want to use those predicted quantile returns to generate trading signals. This document describes how to calculate each of them.

# Signals

## Expected log returns based on all quantiles

We treat quantile as probability, and predicted log return as outcome, then we can proximate the expected log returns based on all quantiles in following ways:

1. Conservative calculation 
```
0.05*q95 + (0.1-0.05)*q90 + (0.15-0.1)*q85 + (0.25-0.15)*q75 + (0.5-0.25)*q50 + (0.75-0.5)*q25 + (0.85-0.75)*q15 + (0.9-0.85)*q10 + (0.95-0.9)*q05 + 0.05*2*q05
```

2. Average calculation
```
0.05*q95 + (0.1-0.05)*(q90+q95)/2 + (0.15-0.1)*(q85+q90)/2 + (0.25-0.15)*(q75+q85)/2 + (0.5-0.25)*(q50+q75)/2 + (0.75-0.5)*(q25+q50)/2 + (0.85-0.75)*(q15+q25)/2 + (0.9-0.85)*(q10+q15)/2 + (0.95-0.9)*(q05+q10)/2 + 0.05*q05
```

Before doing the calculation, we also need to clean the quantile predictions because sometimes a lower quantile prediction could be larger than higher quantile predictions, e.g., q85 > q90, in this case, we want to replace the q90 prediction with the q85. The algorithm should be, first rank them by quantiles, from 5 to 95, then check whether higher quantile prediction is larger than the previous one, if not, replace it with the previous one.

## Quantile prediction exceed certain thresholds

We can use training data's predicted value to determine the threshold, or use the moving average of this quantile prediction as threshold.

## Cross over of quantiles

Whenever lower quantile predicts higher value than higher quantiles

## Implementation Plan

### Inputs and assumptions
- Input CSV: merged quantile predictions from the existing merger (see `model/merge_quantile_predictions.py`).
  - Columns present: `timestamp` (UTC-naive), `y_true`, and `pred_q05, pred_q10, pred_q15, pred_q25, pred_q50, pred_q75, pred_q85, pred_q90, pred_q95`.
  - We will operate row-wise on these quantile columns.

### Preprocessing (monotonic quantiles)
- Goal: enforce non-decreasing quantile predictions per row so `q05 ≤ q10 ≤ ... ≤ q95`.
- Steps per row:
  - Record diagnostics before fixing: number of violations and the maximum downward gap.
  - Apply a cumulative-maximum pass left→right over `[q05, q10, ..., q95]`.
- Outputs added:
  - `cross_violations` (int), `max_cross_gap` (float) computed on the original values.
  - The fixed, monotone quantile columns are used for all downstream metrics.

### Expected log return estimators
We approximate the expectation E[y] by integrating the quantile function Q(p) over p ∈ [0, 1] using a fixed grid at the trained quantiles and duplicated endpoints:
- p-grid: `[0.00, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 1.00]`.
- q-grid: `[q05, q05, q10, q15, q25, q50, q75, q85, q90, q95, q95]` (endpoint duplication).

1) Average (trapezoidal) estimator:
```
E_avg ≈ Σ_i (p_{i+1} − p_i) * (q_i + q_{i+1}) / 2
```

2) Conservative (left-Riemann) estimator:
```
E_conservative ≈ Σ_i (p_{i+1} − p_i) * q_i
```

Outputs added:
- `exp_ret_avg`, `exp_ret_conservative`.

### Probability of positive return
Estimate P(y > 0) by piecewise-linear interpolation on (p, q) after monotonic fix.
- If `q05 ≥ 0` ⇒ `prob_up = 1.0`; if `q95 ≤ 0` ⇒ `prob_up = 0.0`.
- Else find the segment [i, i+1] where q crosses 0 and solve for p* on the line between `(p_i, q_i)` and `(p_{i+1}, q_{i+1})`. Then `prob_up = 1 − p*`.

Output added:
- `prob_up`.

### Threshold-based signals
Two thresholding modes, optionally both:
- Static (from training+validation union preferred): compute percentiles on the union of training and validation predictions for the chosen feature(s) (e.g., `pred_q90`, `pred_q10`, `exp_ret_avg`, `prob_up`).
  - Example: go long if `pred_q90 ≥ pct90_train_val`, go short if `pred_q10 ≤ pct10_train_val`.
  - Fallback: if training merged CSV is not yet available, use validation-only percentiles and note the narrower coverage.
- Dynamic (rolling): rolling mean/mean±k·std of a selected quantile (e.g., `pred_q90` for longs, `pred_q10` for shorts).
  - Example: signal when the series crosses above/below its rolling baseline.

Notes on thresholds data sources:
- We will support providing multiple CSVs for threshold estimation. The CLI will accept a list of files (e.g., merged training and merged validation predictions), concatenate them, and compute percentiles over the combined distribution.
- Training merged predictions are not produced yet. We plan to:
  - Extend the merger to support `--split train` (or generate merged train predictions via the same mechanism if `pred_train.csv` files exist per run).
  - Until then, you can proceed with validation-only thresholds; later re-run thresholds with the train+val union to refresh the signal cutoffs.

We will also expose signals based on expected return and probability:
- `signal_exp_avg`: sign of `exp_ret_avg` with optional band around zero.
- `signal_prob_up`: long if `prob_up ≥ τ_long`, short if `prob_up ≤ τ_short`.

### Position sizing (optional)
- Use tail spread `tail_spread = q95 − q05` as a volatility proxy to scale position size.
- Optionally combine with |expected return|, clip to `max_leverage`.
- Output: `position_size` in [-1, 1] (or scaled), derived from selected sizing rule.

### Module and CLI layout
- Implement reusable utilities in `strategies/quantile_signals.py`:
  - `apply_monotonic_fix(df, qcols) -> (df_fixed, diagnostics_df)`
  - `expected_return_trapezoid(row, p, qcols) -> float`
  - `expected_return_left(row, p, qcols) -> float`
  - `estimate_prob_up(row, p, qcols) -> float`
  - `compute_tail_spread(row, qcols) -> float`
  - `generate_signals(df, cfg) -> df_with_signals`
- Add a CLI wrapper `model/compute_signals_from_quantiles.py`:
  - Args:
    - `--pred-csv` path to merged `pred_val.csv` or `pred_test.csv`
    - `--method {avg,conservative}` default `avg`
    - `--threshold-csvs PATH [PATH ...]` optional list of CSVs (e.g., merged train and merged val) used to compute static thresholds over the union; if omitted, static thresholds are disabled or computed from `--pred-csv` only when explicitly requested.
    - `--rolling N` optional int window for dynamic thresholds
    - `--prob-thresholds τ_long τ_short` optional floats (defaults 0.55 / 0.45)
    - `--out` output CSV path (default: side-by-side next to input)
  - Writes the input CSV augmented with the new columns.

### Configuration defaults
- p-grid fixed at the trained quantiles plus endpoints as above.
- Default method: `avg` (trapezoid), as it is less biased than left-Riemann.
- Default thresholds: prefer computing from the union of training+validation when `--threshold-csvs` are supplied; otherwise thresholds are disabled by default (or may fall back to the single provided file if explicitly requested). Use a modest zero-band around expected return (e.g., ±1e-5) to reduce churn.

### Testing and validation
- Unit tests (synthetic):
  - Quantile crossing examples verify `apply_monotonic_fix` and diagnostics.
  - Closed-form checks: constant q across p ⇒ E equals that constant; linear q(p) ⇒ numerical E matches analytic integral within tolerance.
  - Probability interpolation tests with known brackets.
- Integration test:
  - Read a small sample merged CSV, compute outputs, verify columns exist and contain finite values; smoke-check distribution ranges.

### Outputs summary (added columns)
- `cross_violations`, `max_cross_gap`, `exp_ret_avg`, `exp_ret_conservative`, `prob_up`, `tail_spread`, optional `position_size`, and boolean/int signals depending on chosen strategy.

### Next steps
- Implement `strategies/quantile_signals.py` and the CLI wrapper.
- Extend `model/merge_quantile_predictions.py` to support `--split train` so we can produce a merged training predictions CSV (or document a temporary alternative if needed).
- Optionally add a quick plotting helper to visualize `prob_up`, expected returns, and generated signals against `y_true`.
