# HMM Feature Engineering — Implementation Plan

This document lays out a concrete, reuse‑first plan to implement feature engineering for Hidden Markov Models (HMM) using existing modules. It defines the minimal observation set, optional extensions, APIs, data flow, tests, and a granular task list to deliver a reliable, leakage‑safe feature pipeline for regime detection.

## Goals
- Build a compact, stationary multivariate observation vector per bar suitable for Gaussian HMM.
- Reuse existing feature generators wherever possible; avoid duplicate logic.
- Provide both batch (vectorized DataFrame) and latest‑row (from lookbacks) builders.
- Standardize features via z‑scaling with a persistable scaler for inference.

## Baseline From HMM Doc (essentials)
- Log returns (stationarity)
- Intra‑period volatility (Parkinson)
- Volume stationarization (log change)
- Standardization (z‑score)
- Stack into multivariate observation matrix

## Proposed Feature Sets

### v1 (minimal, robust, 1H)
- `close_logret_current_1H` — log(C_t / C_{t−1})
  - Reuse current‑bar generator: `feature_engineering/current_bar_features.py:60`
  - Computation ref: `feature_engineering/current_bar_features.py:94`
- `close_parkinson_20_1H` — Parkinson volatility over rolling 20 bars
  - Vectorize with the same formula as: `feature_engineering/multi_timeframe_features.py:596`
- `log_volume_delta_current_1H` — log1p(V_t) − log1p(V_{t−1})
  - Reuse from current‑bar generator: `feature_engineering/current_bar_features.py:60`

Rationale: 3 orthogonal signals (return, volatility, liquidity) with low collinearity, strong regime signal, and minimal warmup.

### v2 (optional adds; add one per family)
- Return z‑score: `close_ret_zscore_20_1H`
  - Formula parity: `feature_engineering/multi_timeframe_features.py:1250`
- Relative volume: `volume_rvol_20_1H`
  - Parity: `feature_engineering/multi_timeframe_features.py:1188`
- Price vs VWAP ratio (log): `close_log_ratio_vwap_1H`
  - Parity: `feature_engineering/multi_timeframe_features.py:1600`
- One intrabar range/sentiment: `high_low_range_pct_current_1H` or `close_open_pct_current_1H`
  - From current‑bar generator: `feature_engineering/current_bar_features.py:60`

### Multi‑Timeframe Option (later)
- Duplicate the same compact set for 4H (and optionally 1D) via right‑closed resampling to avoid leakage.
- Resampling utility: `feature_engineering/utils.py:47`

## Module Design

Add a new module: `feature_engineering/hmm_features.py`

Public API:
- `build_hmm_observations_1h(ohlcv_1h: pd.DataFrame, config) -> tuple[pd.DataFrame, dict]`
  - Vectorized per‑row observations for 1H data using v1/v2 picks.
  - Drops warmup rows required by rolling windows.
  - Returns `(obs_df, meta)` where `meta` contains windows used, dropped rows count, and ordered column list.
- `scale_observations(obs_df: pd.DataFrame, scaler=None) -> tuple[pd.DataFrame, StandardScaler]`
  - Fit a `StandardScaler` when `scaler=None`, else apply the given scaler.
  - Caller persists the scaler for inference.
- `latest_hmm_observation_from_lookbacks(lookbacks_by_tf: dict[str, pd.DataFrame], scaler, config) -> tuple[np.ndarray, pd.Timestamp]`
  - Build features at the latest timestamp using 1H (and optional higher TFs via right‑closed resample).
  - Apply the prefit scaler; return a 1×F vector and the timestamp.
- Optional: `build_hmm_observations_multi_tf(lookbacks_by_tf, config) -> tuple[pd.DataFrame, dict]`
  - Concatenate the compact feature set across `['1H','4H']` for more context with bounded dimensionality.

Config object: `HMMFeatureConfig`
- Windows: `vol_window=20`, `ret_zscore_window=20`, `rvol_window=20`
- Feature toggles: `use_logret`, `use_parkinson`, `use_logvol_delta`, `use_ret_zscore`, `use_rvol`, `use_vwap_ratio`, `use_intrabar_range`
- Timeframes: default `['1H']`, optional `['1H','4H']`
- NaN policy: `drop_warmup=True`
- Naming: keep existing column naming + timeframe suffixes for consistency with current feature tables.

## Data Flow

Training (1H):
1. Input: OHLCV DataFrame (sorted, columns: open/high/low/close/volume; DatetimeIndex or a timestamp column).
2. Compute v1 (and optional v2) vectorized features.
3. Drop warmup rows for rolling windows; assert finite values.
4. Fit `StandardScaler` and transform to produce `X` (n_samples × n_features).
5. Train HMM on `X`.

Inference (latest bar):
1. Build lookbacks via existing utilities: `run/lookbacks_builder.py` (uses `feature_engineering/utils.py:76`).
2. Compute the same feature set at t only; if multi‑TF, resample via right‑closed bins.
3. Apply the persisted scaler; return a 1×F observation vector.

## Reuse Mapping (Key Functions)
- Current‑bar vectorized features: `feature_engineering/current_bar_features.py:60`
  - Log return logic: `feature_engineering/current_bar_features.py:94`
  - Intrabar range/close‑open pct and volume transforms are already included.
- Parkinson volatility formula: `feature_engineering/multi_timeframe_features.py:596`
- Return z‑score: `feature_engineering/multi_timeframe_features.py:1250`
- Relative volume: `feature_engineering/multi_timeframe_features.py:1188`
- VWAP ratios (log): `feature_engineering/multi_timeframe_features.py:1600`
- Right‑closed resampling (no leakage): `feature_engineering/utils.py:47`

## Naming, Leakage, and Alignment
- Keep existing suffix conventions: append `_1H`, `_4H`, etc.
- Avoid future leakage by right‑closing higher‑TF aggregations and only using data up to the current bar.
- Ensure observation DataFrame index equals the input index (or normalized timestamp column), sorted ascending.

## Tests and Validation

Deterministic & alignment:
- Index equality with input; strictly increasing timestamps.
- Column order equals config‑declared order.

Numerical integrity:
- After warmup drop, assert no NaNs/Infs.
- Sanity ranges: log returns finite; Parkinson > 0 when `high>low` within window.

Parity checks (vectorized vs scalar at last row):
- For a small slice, compare the last row of vectorized Parkinson(20) to scalar `calculate_parkinson_volatility` output.
- Same approach for return z‑score and relative volume, when enabled.

Leakage checks:
- Multi‑TF resampling tests use `resample_ohlcv_right_closed` and confirm last bin ends at the current 1H bar.

## Implementation Tasks

1) Module scaffold
- Create `feature_engineering/hmm_features.py` with `HMMFeatureConfig` and function stubs.

2) Vectorized feature builders (1H)
- Reuse `compute_current_bar_features` to extract: `close_logret_current_1H`, `log_volume_delta_current_1H`, and optional intrabar features.
- Implement vectorized Parkinson(20) with the same formula as the scalar reference and identical edge handling.
- Optional: vectorized return z‑score(20), relative volume(20), and VWAP log‑ratio.

3) Scaling
- Add `scale_observations` to fit/apply `StandardScaler`; ensure column order consistency is preserved in `meta`.

4) Latest‑row from lookbacks
- Implement `latest_hmm_observation_from_lookbacks` (1H first; optionally 4H) using right‑closed resampling.
- Apply prefit scaler; return 1×F vector + timestamp.

5) Tests
- Unit tests for parity: vectorized Parkinson vs scalar (last row).
- Test for NaN‑free outputs after warmup, fixed column order, and deterministic latest‑row building.

6) Docs & examples
- Short usage snippet in docs: build v1 observations from 1H OHLCV, fit scaler + GaussianHMM.

## Open Questions
- Should 4H be included in v1 by default, or keep 1H only to minimize dimension for initial runs?
- Preference for intrabar feature: `high_low_range_pct_current` vs `close_open_pct_current` (or both)?
- Any constraints on observation dimension for HMM training stability (e.g., <= 5)?

## Rollout Plan
- Phase 1: Implement v1 (1H only), tests, and a minimal example; evaluate HMM regimes.
- Phase 2: Add v2 options and optional 4H; re‑evaluate interpretability and stability.

