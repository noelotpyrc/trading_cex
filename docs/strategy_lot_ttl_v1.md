**Purpose**
- Provide a minimal, repeatable way to evaluate signal + strategy combinations using lot-level returns only.
- No per-bar PnL, NAV, or costs; focus purely on entry→exit results per lot.

**Scope**
- Single-asset, bar-based OHLCV at any fixed timeframe (e.g., 1h, 4h, 1d).
- Signals are on the same timeframe as OHLCV.
- Entry executes at next_open by convention; shorts allowed by default.
- Exit mechanics (boundary choice and triggers) will be specified separately. This document only defines how the lot return is computed once an entry and an exit have been determined.

**Data Requirements**
- OHLCV table with columns: `timestamp, open, high, low, close, volume`.
  - `timestamp` is the bar open time, UTC-naive, strictly regular at the chosen frequency, deduped and sorted ascending.
- Signals table with columns: `timestamp, <signal columns...>` on the same timeframe.
- Join signals to OHLCV by inner-joining on `timestamp` after normalizing timestamps to UTC-naive at the chosen frequency.

**Lot Return**
- Decision bar index `t0` is where the enter rule evaluates to true.
- Entry executes at the next bar’s open: `start_idx = t0 + 1`, `px_entry = open[start_idx]`.
- Exit decision occurs at index `t_exit` according to the chosen exit rule. Exit executes at the configured boundary:
  - If `exit_at = "close"`: `px_exit = close[t_exit]` (exit fills at that bar’s close).
  - If `exit_at = "next_open"`: `px_exit = open[t_exit + 1]` (exit fills at the next bar’s open).
- Lot gross return is defined as: `gross_return = (px_exit / px_entry) - 1`.
- End-of-data handling: if the required exit boundary price is not available (e.g., `t_exit + 1` is out of range for `next_open`), exit at the last available boundary and flag as `EOD` during implementation.

**Return Semantics**
- TP/SL exits are instant at threshold: when TP/SL triggers, the lot fills at the threshold price.
  - Long: TP at `px_exit = px_entry * (1 + take_profit_pct)`; SL at `px_exit = px_entry * (1 - stop_loss_pct)`.
  - Short: TP at `px_exit = px_entry * (1 - take_profit_pct)`; SL at `px_exit = px_entry * (1 + stop_loss_pct)`.
  - Gross return for TP/SL equals the asset price change from entry to that threshold: `gross_return = px_exit / px_entry - 1` (e.g., +TP for long TP, −TP for short TP).
- Non‑TP/SL exits (e.g., time cap or end‑of‑data) fill at the configured boundary price:
  - `exit_at = "close"` → `px_exit = close[t_exit]`
  - `exit_at = "next_open"` → `px_exit = open[t_exit + 1]`
  - Gross return = `px_exit / px_entry - 1`.

**Entry Rule**
- Define enter conditions as boolean expressions over the joined OHLCV + signal columns.
  - `enter.long`: expression for long entries (e.g., `signal > 0.6`).
  - `enter.short`: expression for short entries (e.g., `signal < -0.6`).
- Evaluation and spawning:
  - Mode: continuous — evaluate at decision bar `t0`; spawn a new lot on every bar where the condition is true.
  - If both long and short are true on the same bar, ignore both by default (`allow_both_sides = false`).
  - NaN handling: if any referenced field is NaN at `t0`, do not spawn a lot.
- Execution timing: entries always fill at `next_open` → `start_idx = t0 + 1`, `px_entry = open[start_idx]`.

**Exit Boundary and Triggers**
- Boundary (`exec.exit_at ∈ {"close", "next_open"}`; default `"next_open"`) applies only to non‑TP/SL exits (time/EOD). TP/SL always fills instantly at threshold.
- TP/SL detection always uses intrabar high/low relative to the entry price (`px_entry`), with SL‑first tie breaking:
  - Thresholds: `tp_long = px_entry * (1 + take_profit_pct)`, `sl_long = px_entry * (1 - stop_loss_pct)`,
    `tp_short = px_entry * (1 - take_profit_pct)`, `sl_short = px_entry * (1 + stop_loss_pct)`.
  - On each bar `t ≥ start_idx`:
    - Long: if both `low[t] ≤ sl_long` and `high[t] ≥ tp_long` → SL (tie → SL first); else if `high[t] ≥ tp_long` → TP; else if `low[t] ≤ sl_long` → SL.
    - Short: if both `high[t] ≥ sl_short` and `low[t] ≤ tp_short` → SL (tie → SL first); else if `low[t] ≤ tp_short` → TP; else if `high[t] ≥ sl_short` → SL.
  - First bar that satisfies a condition sets `t_exit` and `exit_reason ∈ {TP, SL}`; TP/SL fill at threshold (see Return Semantics).
- Fallback when TP/SL does not hit:
  - If a time cap is configured (`exit.hold_bars = H`, integer ≥ 1), exit at `t_exit = start_idx + H - 1` at the chosen boundary.
  - Otherwise, exit at end of data (EOD) at the last available boundary price (`exit_reason = EOD`).
  - If TP/SL and the time cap both occur on the same decision bar, TP/SL takes precedence.

**Sizing**
- Parameter: `sizing.lot_notional` (required)
  - Positive number representing the fixed currency notional per lot (same currency as OHLC prices, e.g., USD/USDT).
  - Applies equally to long and short lots; shorts are represented by `side = -1`.
- Side and PnL per lot
  - Side `s ∈ {+1 (long), −1 (short)}` is determined by which enter rule fired.
  - With `gross_return = px_exit / px_entry − 1`, the lot PnL in currency units is:
    - `pnl_trade = s * lot_notional * gross_return`.
    - Examples:
      - Long TP of +2% with `lot_notional=10_000` → `pnl_trade = +200`.
      - Short TP of +2% implies `gross_return = -0.02` (price down 2%), side = −1 → `pnl_trade = (−1) * 10_000 * (−0.02) = +200`.
- Multiple entries
  - In continuous entry mode, each decision bar can spawn a new lot; evaluation is per-lot and independent.
  - Aggregations (e.g., total PnL) are computed by summing per‑lot `pnl_trade` across lots in the period of interest.

**Output Schema**
- Lots table (one row per lot):
  - `enter_decision_ts` (timestamp): timestamp of decision bar `t0` that spawned the lot.
  - `entry_ts` (timestamp): timestamp of the bar used for the fill at entry (next_open).
  - `exit_decision_ts` (timestamp): timestamp of the decision bar `t_exit` where the exit trigger was found.
  - `exit_ts` (timestamp): timestamp of the bar used for the fill at exit (per Return Semantics).
  - `exit_at` (string): `close` | `next_open` | `threshold` (TP/SL threshold fill).
  - `exit_reason` (string): `TP` | `SL` | `TIME` | `EOD`.
  - `side` (int): +1 for long, −1 for short.
  - `lot_notional` (float): currency notional per lot.
  - `px_entry` (float): entry price.
  - `px_exit` (float): exit price actually used to compute return.
  - `gross_return` (float): `(px_exit / px_entry) - 1`.
  - `pnl_trade` (float): `side * lot_notional * gross_return`.
  - Optional diagnostics (nullable):
    - `tp_pct` (float): configured take‑profit percent (if any).
    - `sl_pct` (float): configured stop‑loss percent (if any).
    - `hold_bars` (int): configured time cap (if any).
    - `bars_held` (int): realized executed bars from entry to exit decision index.

- Summary (scalar metrics over lots):
  - Counts: `n_lots`, `n_long`, `n_short`.
  - Outcome rates: `tp_rate`, `sl_rate`, `time_rate`, `eod_rate` (fractions of all lots).
  - Hit rate (by PnL): `hit_rate = mean(pnl_trade > 0)`.
  - Return stats: `mean_return`, `median_return`, `p25_return`, `p75_return`.
  - PnL stats: `mean_pnl`, `total_pnl` (currency units).
  - Optional per‑side breakdowns: the same metrics computed on long and short subsets.

Notes
- All timestamps are UTC‑naive and must align to the bar open times used in OHLCV.
- `exit_at = "threshold"` indicates TP/SL filled at the threshold price (instant hit). For TIME/EOD exits, `exit_at` equals the configured boundary (`close` or `next_open`).

**Edge Handling**
- Data quality
  - Timestamps must be strictly regular at the chosen bar frequency and deduplicated; otherwise behavior is undefined.
  - Any NaN in required OHLC (open/high/low/close) for a bar prevents TP/SL detection on that bar; the lot continues to the next bar. If prices needed for the exit fill are missing, exit EOD at the last available boundary.
- Entry at end of data
  - If `t0 + 1` is out of range (no next_open), skip spawning the lot.
- Detection window
  - TP/SL detection starts at `t = start_idx` (the first bar after entry), never earlier.
- TP/SL tie
  - If both TP and SL are reachable within the same bar (by `high/low`), resolve to SL (conservative) and treat the exit as instant at the SL threshold.
- Simultaneous long and short entry signals
  - Default is to ignore both (`allow_both_sides = false`). If enabled (`true`), spawn one long and one short lot independently on the same bar.
- Time cap ties
  - If TP/SL and `hold_bars` would exit on the same decision bar, use TP/SL.
- End-of-data
  - If no exit trigger occurs and no time cap is set, exit EOD at the last available boundary price and label `exit_reason = EOD`.

*** End of File
