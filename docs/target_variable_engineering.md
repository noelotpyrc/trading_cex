## Target Variable Engineering (Forward-Window Labels)

This document specifies leakage-safe, forward-window target variables that pair with the existing lookback-window feature engineering. For the starter scope:
- Only forward-window OHLCV is used to compute labels (plus the entry price at t).
- No fees/slippage.
- No volatility/ATR scaling; use fixed percent thresholds for TP/SL.

## Principles

- No leakage: labels for timestamp t use only prices in (t, t + horizon], never [≤ t]. Use right-closed, right-labeled windows for aggregation.
- Forward-window OHLCV only: derive labels solely from the forward window's OHLCV and the entry price at t.
- Direction symmetry: definitions work for either long or short.
- Parameterized: horizons and fixed percent TP/SL thresholds only.
- Deterministic tie-breaking: define clear policy if TP and SL are breached within the same bar.
- Clear naming: consistent, machine-parsable keys so multiple horizons and configs can co-exist.

## Naming Convention

Labels follow:

- Fixed horizon: `y_{name}_{horizon}{unit}` → e.g., `y_logret_12h`, `y_mfe_24h`.
- Barrier-based (fixed percent): `y_{name}_u{up}_d{down}_{horizon}{unit}` → e.g., `y_tb_label_u1.5_d1.0_24h`.

Where:

- `unit` is bars (`b`) if bar-based or time (`h`, `d`) if time-based.

## Core Label Families

### 1) Fixed-Horizon Future Return (Regression)

- Definition (log): \(r_{t\to t+H} = \log(\text{close}_{t+H}) - \log(\text{close}_t)\)
- Long/Short symmetry: short return is simply `-r` if modeling short payoff separately.
- Variants:
  - `y_logret_{H}`: log return to horizon H
  - `y_ret_{H}`: simple return `(close_{t+H}/close_t - 1)`
- Notes: set to NaN when not enough future bars.

### 2) Extrema Within Window: MFE / MAE (Regression)

- Max Favorable/Adverse Excursion for a long entry at `close_t` over (t, t+H]:
  - `MFE_long = max(high_{t+1..t+H}) / close_t - 1`
  - `MAE_long = min(low_{t+1..t+H}) / close_t - 1`
- For short, flip via price inversion or emit separate keys.
- Variants:
  - `y_mfe_{H}`, `y_mae_{H}`
- Use to estimate achievable TP/SL or to train risk-aware models.

### 3) Barrier Outcomes (Classification)

Fixed-percent upper/lower barriers and a time barrier at H.

- Barriers for long at t:
  - Upper: `close_t * (1 + up_pct)`
  - Lower: `close_t * (1 - down_pct)`
  - Time barrier at t+H
- Ternary label: +1 if upper hit first, -1 if lower hit first, 0 if neither by t+H.
  - Key: `y_tb_label_u{up}_d{down}_{H}`
- Binary label (TP-before-SL): 1 if TP hit before SL within H, else 0 (timeouts counted as 0).
  - Key: `y_tp_before_sl_u{tp}_d{sl}_{H}`
- Tie-breaking within a bar (high ≥ upper and low ≤ lower): choose the one closest to open_{bar} or define `0` (conservative). Policy must be fixed globally.

### 4) Movement Magnitude Bins (Ordinal Classification)

- Discretize fixed-horizon returns or MFE/MAE into bins, e.g. `{-2, -1, 0, +1, +2}`.
- Keys: `y_ret_bin_{H}_k{num_bins}` or `y_mfe_bin_{H}_k{num_bins}`

## Execution Assumptions

- Entry price: default `close_t` or `open_{t+1}`; define globally.
- Intrabar evaluation:
  - Long TP check with highs, SL with lows. For short, TP with lows, SL with highs.
  - If both hit in same bar: use deterministic policy (e.g., assume worst for conservative labeling).

## Suggested Default Configs (1H data)

- Horizons H: {3h, 6h, 12h, 24h, 48h, 72h}
- Fixed TP/SL (percent): `tp_pct ∈ {0.5%, 1%, 2%}`, `sl_pct ∈ {0.5%, 1%, 2%}`

## API Sketch (Row-Oriented, Leakage-Safe)

These follow the project’s style: functions return per-row dictionaries of label keys to values.

```python
from typing import Dict
import pandas as pd
import numpy as np

def compute_forward_return(entry_price: float,
                           forward_close: pd.Series,
                           horizon_bars: int, *, log: bool = True,
                           column_name: str = 'close') -> Dict[str, float]:
    """Return y_logret_{H} or y_ret_{H} using entry price at t and forward close[t+H]."""


def compute_mfe_mae(forward_high: pd.Series, forward_low: pd.Series,
                    entry_price: float, horizon_bars: int) -> Dict[str, float]:
    """Return { 'y_mfe_{H}': ..., 'y_mae_{H}': ... } using forward OHLCV only."""


def compute_barrier_outcomes(forward_high: pd.Series, forward_low: pd.Series,
                             entry_price: float, up_pct: float, down_pct: float,
                             horizon_bars: int, *, mode: str = 'ternary',
                             tie_policy: str = 'conservative',
                             forward_open: pd.Series | None = None) -> Dict[str, float]:
    """Return ternary triple-barrier label and/or binary TP-before-SL label."""
```

## Edge Cases

- Insufficient future data: emit NaN for all labels requiring unavailable bars.
- Gaps/outliers: consider forward-filling timestamps; for spikes, labels reflect recorded OHLC.
- Equal-distance tie in triple-barrier: follow the fixed `tie_policy`.

## Quality and Evaluation

- Classification: balanced accuracy, ROC-AUC, PR-AUC.
- Regression: MAE/MSE; also evaluate directional accuracy vs sign(y).

## Recommended Label Sets to Start

- Short-horizon returns: `y_logret_{3h,6h,12h,24h}`
- Risk-aware: `y_mfe_{12h,24h}`, `y_mae_{12h,24h}`
- Decision/classification: `y_tb_label_u1.5_d1.5_24h`
- Execution-aware: `y_tp_before_sl_u1.0_d1.0_24h`
- Timing: `y_tth_tp_u1.5_24h`

These cover direction, magnitude, risk, execution, and timing and map well to classification and regression models while staying simple and OHLCV-only.
