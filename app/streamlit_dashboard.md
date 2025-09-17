# Streamlit Dashboard Plan

## Objectives
- Visualize 1h BTCUSDT perp OHLCV history and model signals in an interactive, local-first dashboard.
- Provide quick inspection of signal correctness using 168h forward return and max drawdown metrics.
- Allow focused monitoring of high-value feature series alongside the market view.

## Data Inputs
- **OHLCV**: DuckDB table `ohlcv_btcusdt_1h` in `/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb` (timestamp, open, high, low, close, volume).
- **Predictions / Signals**: DuckDB table `predictions` in `/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb` (columns: `ts`, `y_pred`, `model_path`, `feature_key`).
- **Features**: DuckDB table `features` in `/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb` with JSON feature snapshots keyed by `feature_key` and timestamp.
- Metrics computed client-side after loading: 168h forward return and horizon max drawdown derived from OHLCV closes.

## UI Layout
- **Sidebar**
  - DuckDB path selectors (defaults to local production files).
  - Date range selector (default: trailing 90 days) applied to all panels.
- Feature dropdown listing high-importance features (top-N from `feature_importance.csv` in the model folder referenced by `model_path`).
- Checkboxes to toggle long (> threshold) and short (< threshold) signal overlays independently.
  - Manual refresh button that clears cached loaders and re-queries DuckDB.
  - Optional toggles: prediction threshold, show/hide volume, choose return metric (log vs pct).
- **Main Panel (top container)**
  - Plotly candlestick chart for 1h OHLCV with volume overlay.
  - Signal markers placed at prediction timestamps; marker color encodes correctness (green if forward 168h return meets direction, red otherwise), outline/opacity reflects max drawdown during the horizon.
  - Hover tooltip shows timestamp, prediction value, forward return %, max drawdown %, and metadata (`model_path`, `feature_key`).
  - Chart remains interactive (zoom/pan, range sliders enabled).
- **Bottom Panel (secondary container)**
  - Plotly line chart of the selected feature value over time.
  - Shares the same x-axis extent as the main panel; zooming the top chart updates the bottom chart via linked layout state.
  - Tooltip shows feature value, timestamp, and optionally standardized z-score.

## Interaction & State Management
- Utilize `st.cache_data(ttl=None)` for initial DuckDB pulls; manual refresh button calls `.clear()` on cached functions.
- Store sidebar selections in `st.session_state` to maintain choices across reruns.
- Use Plotly `relayoutData` events (via `streamlit-plotly-events` or native `st.plotly_chart` callbacks once available) to synchronize x-axis ranges between panels.
- Provide lightweight status indicators (`st.toast`/`st.status`) when queries are running or refresh is triggered.

## Implementation Notes
- Data loader module encapsulates DuckDB SQL; fetch OHLCV/predictions/features in a single call where practical to minimize I/O.
- Expand feature JSON to columns once per load; retain both wide DataFrame and melted long-form for plotting convenience.
- Precompute forward-return/drawdown metrics by joining predictions with OHLCV closes shifted by +168h and intrahorizon minima/maxima.
- Guard against missing future bars (e.g., most recent predictions) by flagging incomplete horizons and styling markers accordingly (grey).
- Package dependencies: `streamlit`, `plotly`, `duckdb`, `pandas`, optional `streamlit-plotly-events` (verify availability before committing).
- Keep all plotting logic modular so it can be unit tested outside Streamlit (e.g., helper functions returning Plotly figures given DataFrames).

## TODO
- [ ] Implement cached DuckDB loaders (`load_ohlcv`, `load_predictions`, `load_features`) with error messaging for missing files.
- [ ] Build helper to compute 168h forward return and drawdown metrics.
- [ ] Scaffold Streamlit app file with sidebar controls and placeholder charts.
- [ ] Connect real data loaders, overlay signals, and link charts.
- [ ] Add automated smoke test (e.g., CLI script) verifying figures render with synthetic data.
