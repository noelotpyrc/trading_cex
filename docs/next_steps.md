# Next Steps Implementation Plan

Based on current TODO items, focusing on practical improvements without over-engineering.

## Priority 1: Engineering Tasks

### 1. Config-driven Model Training Pipeline

**Goal**: Replace hardcoded parameters with JSON config files

**Implementation**:
- Create `configs/model_configs/` directory
- Add config schema for:
  - Feature selection
  - Target selection
  - Split method parameters
  - Model hyperparameters
- Create a new run script to ingest the config and run the whole training data prep -> hyperparameter tuning -> traing and persisting process

**Example config structure**:
```json
{
  "input_data": "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv",
  "output_dir": "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P",
  "features": ["..."],
  "target": {
    "variable": "y_logret_24h",
    "objective": {"quantiles":0.05}
  },
  "split": {
    "method": "time_series",
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15
  },
  "model": {
    "type": "lgbm",
    "hyperparameter_tuning_method": "Bayesian",
    "hyperparameter_search_space": {"placeholder"},
    "params": {"placeholder"},
    "eval": ["rmse", "mae"]
  }
}
```

### 2. Document Complete Run Process

**Goal**: Clear documentation of end-to-end pipeline

**Update `docs/run_cmd.md`** with:

1. **Feature Engineering**:
   ```bash
   python feature_engineering/merge_features_targets.py --config configs/btc_features.json
   ```

2. **Data Preparation**:
   ```bash
   python model/prepare_training_data.py --config configs/model_configs/btc_24h_logret_basic.json
   ```

3. **Model Training**:
   ```bash
   python model/train_lgbm_quantiles.py --config configs/model_configs/btc_24h_logret_basic.json
   ```

4. **Evaluation & Visualization**:
   ```bash
   python model/plot_predictions_interactive.py --run-dir {model_output_dir}
   ```

## Priority 2: Modeling Tasks

### 3. Hyperparameter Tuning for Underfitting

**Goal**: Systematic hyperparameter search to improve model performance

**Implementation**:
- Add `model/tune_hyperparameters.py`
- Grid search over key parameters:
  ```python
  param_grid = {
      'learning_rate': [0.05, 0.1, 0.2],
      'num_leaves': [15, 31, 63],
      'max_depth': [3, 6, 9],
      'min_data_in_leaf': [50, 100, 200]
  }
  ```
- Different parameter sets for different objectives (regression vs quantiles)
- Use validation set for evaluation

### 4. Automatic Model Selection

**Goal**: Pick best model based on validation metrics

**Implementation**:
- Add evaluation logic to training scripts
- Metrics by model type:
  - **Regression**: RMSE, MAE, R²
  - **Quantiles**: Pinball loss per quantile
- Save best model + metadata:
  ```json
  {
    "best_config": {...},
    "validation_metrics": {...},
    "model_path": "best_model.pkl",
    "selected_on": "2024-08-29"
  }
  ```

## Priority 3: Strategy Tasks

### 5. Expected Value Signal Generation

**Goal**: Create trading signals from quantile predictions

**Implementation**:
- Add `strategies/quantile_strategy.py`
- Expected return calculation:
  ```python
  def calculate_expected_return(q05, q50, q95):
      # Simple weighted average
      return 0.1 * q05 + 0.8 * q50 + 0.1 * q95
  
  def generate_signal(expected_return, threshold=0.005):
      return 1 if expected_return > threshold else -1 if expected_return < -threshold else 0
  ```
- Integrate with existing backtesting framework

## Implementation Timeline

### Week 1: Engineering Foundation
- [ ] Create config system for model training
- [ ] Update documentation with complete run commands
- [ ] Test config-driven pipeline end-to-end

### Week 2: Model Improvement
- [ ] Implement hyperparameter tuning
- [ ] Add automatic model selection
- [ ] Test on multiple targets (logret, mfe)

### Week 3: Strategy Integration
- [ ] Build expected value signal generator
- [ ] Integrate with backtesting framework
- [ ] Compare against existing RSI strategy

## Success Metrics

- **Config System**: Can train models with different configs without code changes
- **Model Performance**: Improved validation metrics vs current baseline
- **Strategy Performance**: Expected value strategy shows promise in backtests

## Notes

- Keep existing code structure intact
- Focus on addressing current underfitting issues
- Maintain leakage-safe implementations
- Test each component before moving to next priority