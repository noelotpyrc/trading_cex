# HMM Model Training Plan

This document outlines a lean, config‑driven pipeline to train and use a Gaussian Hidden Markov Model (HMM) for regime detection on the new HMM feature sets (v1/v2). The design mirrors our existing LightGBM runner (model/run_lgbm_pipeline.py): a single script that reads a JSON config, runs training end‑to‑end, and saves artifacts; plus a minimal inference helper.

## Goals
- Unsupervised regime detection with a compact, stationary observation vector.
- Support both v1 (minimal) and v2 (extended) features across 1H/4H/12H/1D.
- Provide a CLI to train, select state count, save artifacts, and apply the model for inference.

## Inputs
- Features CSVs (preferred for batch training)
  - v1: built via `run/build_hmm_v1_features_csv.py`
  - v2: built via `feature_engineering/build_hmm_v2_features_csv.py`
- Live lookbacks (preferred for latest-bar inference)
  - Built via `run/lookbacks_builder.py` + `feature_engineering/utils.py` (right-closed resampling)
  - Observation built via `feature_engineering/hmm_features.latest_hmm_observation_from_lookbacks`

## Dependencies
- `hmmlearn` for `GaussianHMM` (to be added to `requirements.txt`).
- `scikit-learn` for `StandardScaler` (already used).

## Configuration
Create `model/hmm_config.py` with a dataclass:
- Data: `feature_set` in {`v1`,`v2`}, `timeframes` (e.g., `['1H']`), `features_csv`, `train_start`, `train_end` (optional).
- Features: `parkinson_window` (default 20) for consistency with builders.
- Model: `n_states` or `state_grid` (e.g., `[2,3,4,5,6]`), `covariance_type` in {`diag`,`full`}, `n_iter` (200), `tol` (1e-3), `random_state`.
- Split: time-based fractions (e.g., `train=0.7`, `val=0.15`, `test=0.15`).
- Output: `out_dir` for run artifacts.

## Pipeline
1. Load features
   - Read CSV, normalize `timestamp`, select columns by `feature_set` and `timeframes`.
   - Drop NaN rows only in selected observation columns.
2. Split (time-based)
   - Train/Val/Test by timestamp proportions or date ranges.
3. Scale
   - Fit `StandardScaler` on train only; transform val/test.
4. Model selection (if `state_grid` provided)
   - For each `n_states`: fit on train, compute BIC/AIC and val log-likelihood.
   - Select by lowest BIC; break ties by higher val log-likelihood.
5. Final fit
   - Refit on (train + val) if desired; evaluate on test.
6. Evaluate
   - Log-likelihoods (train/val/test), BIC/AIC, transition matrix, stationary distribution, dwell times.
   - Per-state feature means/covariances; optional state labeling heuristic (e.g., sort by mean volatility).
7. Save artifacts
   - `model.joblib`, `scaler.joblib`, `config.json`, `metrics.json`.
   - `regimes.csv` with `timestamp`, `state`, and per-state posteriors (test or full range).
   - `diagnostics.csv` (transition matrix, dwell stats, state means).

## Modules & CLIs (lean)
- `model/run_hmm_pipeline.py`
  - Single entrypoint (like run_lgbm_pipeline.py) with `--config <json>` and optional `--log-level`.
  - Internals: load features → time split → fit/apply StandardScaler → select state count (if grid provided) → fit final model → save artifacts (model, scaler, metrics, regimes, diagnostics).
- `model/hmm_inference.py`
  - Minimal helpers for inference from features DataFrame or latest lookbacks (via `hmm_features.latest_hmm_observation_from_lookbacks`).

## Modeling Details
- Model: `GaussianHMM(n_components=k, covariance_type=diag|full, tol, n_iter, random_state)`.
- Initialization: k-means; set `reg_covar` if needed to avoid singular covariance.
- Selection criteria: prefer BIC; validate with holdout log-likelihood.

## Outputs
- `<out_dir>/model.joblib`, `<out_dir>/scaler.joblib`, `<out_dir>/config.json`.
- `<out_dir>/metrics.json` (AIC/BIC, LLs by split, selected `n_states`).
- `<out_dir>/regimes.csv`: `timestamp`, `state`, `p_state_0..K-1` for the evaluation range.
- `<out_dir>/diagnostics.csv`: transition matrix, stationary distribution, dwell times, per-state feature means.

## Inference Paths
- From features CSV: load, scale with saved scaler, call `predict_states`.
- From live data: build lookbacks, call `hmm_features.latest_hmm_observation_from_lookbacks`, scale, then predict.

## Config (example)
```
{
  "input_data": "/path/to/hmm_v1_features.csv",
  "output_dir": "runs/hmm_v1_1h",
  "features": {
    "feature_set": "v1",
    "timeframes": ["1H"],
    "parkinson_window": 20
  },
  "split": {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15
  },
  "model": {
    "state_grid": [2,3,4,5],
    "covariance_type": "diag",
    "n_iter": 200,
    "tol": 1e-3,
    "random_state": 42
  }
}
```

Columns-only mode (no v1/v2 needed)
```
{
  "input_data": "/path/to/hmm_v1v2_features.csv",   // a joined CSV with all desired columns
  "output_dir": "runs/hmm_columns_only_1h",
  "features": {
    // Ignore feature_set/timeframes; explicitly list the columns to use
    "columns": [
      "close_logret_current_1H",
      "log_volume_delta_current_1H",
      "close_parkinson_20_1H",
      "close_ret_zscore_20_1H",
      "volume_rvol_20_1H",
      "close_over_vwap_1H",
      "close_log_ratio_vwap_1H",
      "high_low_range_pct_current_1H",
      "close_open_pct_current_1H"
    ]
  },
  "split": { "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15 },
  "model": { "state_grid": [2,3,4,5], "covariance_type": "diag", "n_iter": 200, "tol": 1e-3, "random_state": 42 }
}
```

## Tests (essential)
- Unit: deterministic fit (fixed seed), shape checks, `select_n_states` returns a value within grid; posteriors sum to 1.
- E2E: train on `hmm_v1_features.csv` (1H), generate `regimes.csv`; quick spot-match with observations recomputed from `hmm_features` for the same timestamps.
- Live inference: verify 1×F observation from lookbacks matches CSV row at same timestamp for parity.

## Example Commands
- Build features
  - v1 (all TFs):
    - `python run/build_hmm_v1_features_csv.py --timeframes 1H 4H 12H 1D --output \
      "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v1_features.csv"`
  - v2 (all TFs):
    - `python feature_engineering/build_hmm_v2_features_csv.py --timeframes 1H 4H 12H 1D --output \
      "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_v2_features.csv"`
- Train (config‑driven runner):
  - `python model/run_hmm_pipeline.py --config configs/model_configs/hmm_v1_1h.json`
- Apply on latest lookbacks (simple CLI or small helper):
  - `python run/apply_hmm.py --model-dir runs/hmm_v1_1h --from-lookbacks`

## Milestones
1) Add `hmmlearn` to requirements; scaffold `model/run_hmm_pipeline.py` + `model/hmm_inference.py`.
2) Implement v1 1H training with fixed `n_states` or small `state_grid`; save artifacts.
3) Add BIC + val log-likelihood selection, diagnostics (transition matrix, per-state means).
4) Extend to v2 and multi‑TF column selection; keep pipeline unchanged.
5) Add apply helper for latest lookbacks; add E2E tests and brief docs.
