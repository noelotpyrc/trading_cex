## Quantile signal flow

This documents the end-to-end workflow to go from trained quantile models to trading signals and inspection.

- Train quantile models (LightGBM)
- Merge quantile predictions across runs per split
- Generate signals from merged quantile predictions
- Inspect performance in a notebook

### 0) Setup handy variables (adjust to your paths)

```bash
export PROJECT_ROOT="/Users/noel/projects/trading_cex"
export INPUT_DATA="/abs/path/to/merged_features_targets.csv"          # same path used in your model configs
export TARGET="y_logret_24h"                                         # change to your target
export MODELS_ROOT="/abs/path/to/output/models"                       # parent directory containing run_* folders
export DIAG_DIR="$MODELS_ROOT/diagnosis/$TARGET"                      # where merged preds and signals will be written
mkdir -p "$DIAG_DIR"
```

### 1) Train quantile models

Use quantile configs (objective name: `quantile`, params.alpha: e.g., 0.05/0.10/0.50/0.90/0.95). You can run a single config or a batch.

- Single config:

```bash
python "$PROJECT_ROOT/model/run_lgbm_pipeline.py" \
  --config /abs/path/to/your_quantile_q05_config.json \
  --log-level INFO
```

- Batch by explicit config list (recommended to cover multiple alphas):

```bash
python "$PROJECT_ROOT/model/run_configs_batch.py" \
  --configs \
    /abs/path/to/your_quantile_q05_config.json \
    /abs/path/to/your_quantile_q10_config.json \
    /abs/path/to/your_quantile_q50_config.json \
    /abs/path/to/your_quantile_q90_config.json \
    /abs/path/to/your_quantile_q95_config.json \
  --log-level INFO
```

- Batch by filtering a directory of configs (must match `input_data` and `target.variable` inside each config):

```bash
python "$PROJECT_ROOT/model/run_configs_batch.py" \
  --config-dir "$PROJECT_ROOT/configs/model_configs" \
  --input-data "$INPUT_DATA" \
  --target "$TARGET" \
  --log-level INFO
```

Each run writes a `run_*` folder under the config's `output_dir` with `pred_train.csv`, `pred_val.csv`, `pred_test.csv` and `pipeline_config.json` (stores the quantile alpha).

### 2) Merge quantile predictions (per split)

Merge the predictions from multiple quantile runs (q05, q10, q50, q90, q95) into a single CSV per split.

- Auto-discover runs by `MODELS_ROOT`, `TARGET`, and `INPUT_DATA`:

```bash
for SPLIT in train val test; do
  python "$PROJECT_ROOT/model/merge_quantile_predictions.py" \
    --models-root "$MODELS_ROOT" \
    --target "$TARGET" \
    --input-data "$INPUT_DATA" \
    --split "$SPLIT" \
    --out-dir "$DIAG_DIR"
done
```

This writes: `"$DIAG_DIR/pred_train.csv"`, `pred_val.csv`, `pred_test.csv` with columns: `timestamp` (if available), `y_true`, `pred_q05..pred_q95`.

- Or merge by explicitly listing run directories:

```bash
python "$PROJECT_ROOT/model/merge_quantile_predictions.py" \
  --run-dirs \
    /abs/path/to/run_q05 \
    /abs/path/to/run_q10 \
    /abs/path/to/run_q50 \
    /abs/path/to/run_q90 \
    /abs/path/to/run_q95 \
  --split test \
  --out-dir "$DIAG_DIR"
```

### 3) Generate signals from merged quantile predictions

Compute diagnostics, expected returns, probability-of-positive, and basic signals.

- Minimal expected-returns only:

```bash
python "$PROJECT_ROOT/strategies/compute_signals_from_quantiles.py" \
  --pred-csv "$DIAG_DIR/pred_test.csv" \
  --exp-only \
  --out "$DIAG_DIR/pred_test_exp.csv"
```

- Full signals for each split (uses `avg` expected-return method by default):

```bash
# Train
python "$PROJECT_ROOT/strategies/compute_signals_from_quantiles.py" \
  --pred-csv "$DIAG_DIR/pred_train.csv" \
  --method avg \
  --out "$DIAG_DIR/pred_train_signals.csv"

# Val
python "$PROJECT_ROOT/strategies/compute_signals_from_quantiles.py" \
  --pred-csv "$DIAG_DIR/pred_val.csv" \
  --method avg \
  --out "$DIAG_DIR/pred_val_signals.csv"

# Test (optionally derive static thresholds from train+val; tune probabilities if desired)
python "$PROJECT_ROOT/strategies/compute_signals_from_quantiles.py" \
  --pred-csv "$DIAG_DIR/pred_test.csv" \
  --method avg \
  --threshold-csvs "$DIAG_DIR/pred_train.csv" "$DIAG_DIR/pred_val.csv" \
  --prob-thresholds 0.55 0.45 \
  --out "$DIAG_DIR/pred_test_signals.csv"
```

Notes:
- `--method` can be `avg` (trapezoid integral) or `conservative` (left Riemann).
- `--exp-zero-band` adds a deadband around 0 for expected-return signal (default 1e-6).
- When `--threshold-csvs` is provided, quantile-based signal uses 90th percentile of `pred_q90` for longs and 10th percentile of `pred_q10` for shorts over the union of supplied CSVs.

Outputs include columns like: `exp_ret_avg`, `exp_ret_conservative`, `prob_up`, `tail_spread`, `signal_exp`, `signal_prob`, and optionally `signal_quantile`.

### 4) Inspect performance in a notebook

Open either notebook and point paths to `"$DIAG_DIR"` files created above:

- `strategies/local_inspect.ipynb`
- `strategies/inspect_y_logret_performance.ipynb`

Example launch:

```bash
jupyter lab "$PROJECT_ROOT/strategies/local_inspect.ipynb"
```

You can visualize distributions, time series, and compare `signal_*` columns across splits.


