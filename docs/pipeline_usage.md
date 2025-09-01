## Unified LightGBM Pipeline - Usage Guide

This guide explains how to run the config-driven LightGBM training pipeline and what artifacts it produces.

### Overview
- Single command trains one model based on a JSON config.
- Steps: prepare splits -> hyperparameter tuning (optional) -> train with early stopping -> persist artifacts.
- Time-series CV uses sklearn TimeSeriesSplit via LightGBM `lgb.cv` during tuning.

### Command
```bash
/Users/noel/projects/trading_cex/venv/bin/python model/run_lgbm_pipeline.py --config <path/to/config.json> --log-level INFO
```

### Config file
- See `configs/model_configs/example_pipeline_config.jsonc` for a fully commented example.
- Key sections:
  - `input_data`: CSV with features and a target column (and optional `timestamp`).
  - `output_dir`: Folder for model run artifacts.
  - `training_splits_dir` (optional): Root folder to store/reuse splits. Defaults to `<output_dir>/training_splits`.
  - `target.variable`: Target column name.
  - `target.objective`: LightGBM objective (`regression`, `quantile` with `{alpha}`, etc.).
  - `split`: Either ratios/cutoffs to generate new splits, or `existing_dir` to reuse prepared splits.
  - `model`: LightGBM settings, CV for tuning, and params (including `num_boost_round`, `early_stopping_rounds`).

### Split options
- Generate new splits (ratio & time-ordered):
  ```json
  "split": { "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15 }
  ```
- With timestamp cutoffs (if `timestamp` exists):
  ```json
  "split": { "cutoff_start": "2024-12-31", "cutoff_mid": "2025-06-01" }
  ```
- Reuse existing splits:
  ```json
  "split": { "existing_dir": "/abs/path/to/prepared_YYYYMMDD_HHMMSS_<target>" }
  ```

### Hyperparameter tuning (optional)
- Grid search:
  ```json
  "model": {
    "hyperparameter_tuning_method": "grid",
    "hyperparameter_search_space": { "learning_rate": [0.05, 0.1], "num_leaves": [15, 31] },
    "cv": { "method": "expanding", "n_folds": 3, "fold_val_size": 0.2, "gap": 0 }
  }
  ```
- Bayesian (Optuna required):
  ```json
  "model": { "hyperparameter_tuning_method": "bayesian", "hyperparameter_search_space": { ... } }
  ```

### Outputs
- Model run directory (non-splits):
  - Location: `<output_dir>/run_<timestamp>_lgbm_<target>_<objective>/`
  - Files:
    - `model.txt`: LightGBM model
    - `pred_train.csv`, `pred_val.csv`, `pred_test.csv`
    - `feature_importance.csv`
    - `metrics.json`: primary metrics on train/val/test
    - `pipeline_config.json`: exact config used
    - `best_params.json`: chosen params (from tuning or provided)
    - `run_metadata.json`: summary metadata
    - `paths.json`: includes `input_data`, `prepared_data_dir`, `training_splits_dir`, `output_dir`
    - `prep_metadata.json`: copied from splits folder
    - `tuning_trials.csv`: CV trial summary (if tuning enabled)

- Training splits (reusable across runs):
  - Location: `<training_splits_dir>/prepared_<timestamp>_<target>/`
  - Files:
    - `X_train.csv`, `y_train.csv`, `X_val.csv`, `y_val.csv`, `X_test.csv`, `y_test.csv`
    - `prep_metadata.json` (cleaning/splitting details and timestamp ranges)
    - `tuning_trials.csv` (appended during tuning; also copied into run dir)

### Best practices
- Keep `training_splits_dir` outside volatile run folders to reuse splits.
- Use `val` only for early stopping in final training; rely on CV for tuning.
- For quantile objective, provide a single `alpha`, e.g. `{ "name": "quantile", "params": { "alpha": 0.05 } }`.

### Examples
- Minimal no-tuning run:
```bash
/Users/noel/projects/trading_cex/venv/bin/python model/run_lgbm_pipeline.py \
  --config configs/model_configs/synth_regression_rmse.json
```
- Run with existing splits:
```json
"split": { "existing_dir": "/Volumes/Extreme SSD/trading_data/cex/models/e2e/shared_splits/prepared_20250831_201724_y_logret_24h" }
```


