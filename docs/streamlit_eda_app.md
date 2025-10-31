# Streamlit EDA App (In‑Repo)

## Objectives
- Explore premade feature tables alongside OHLCV with a no/low‑code UI.
- Compute targets on the fly from OHLCV (e.g., forward returns, direction, triple‑barrier) without leakage.
- Provide common EDA: overview, distributions, correlations, interactions, missingness, time‑series.
- Export a one‑click HTML EDA report for a chosen subset of columns.
- Reuse existing DuckDB stores and loaders in this repo.

## Scope and Placement
- This is a separate EDA app, complementary to the monitoring dashboard in `app/streamlit_app.py`.
- Proposed entry point: `app/streamlit_eda.py` (kept self‑contained and modular).
- Optional helpers under `app/eda/` if the file grows (loader, recipes, plots).

## Data Inputs
- Feature Store (CSV folder)
  - Source: a folder containing `ohlcv.csv`, `features.csv`, `targets.csv`.
  - Timestamp column: accepts `timestamp`, `time`, or first column fallback. Parsed as UTC and made tz‑naive.
  - Default folder: `/Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_tradingview/feature_store`.

## Configuration
- App reads a small YAML config for paths and defaults. Example:

```yaml
# configs/eda_config.yaml (copy from eda_config.example.yaml and edit)

dataset_id: binance_btcusdt_perp_1h

ohlcv:
  duckdb_path: "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
  table: "ohlcv_btcusdt_1h"

features:
  duckdb_path: "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb"
  table: "features"
  default_feature_key: "binance_btcusdt_perp_1h__v1"

feature_store:
  folder: "/Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_tradingview/feature_store"
  ohlcv_csv: "ohlcv.csv"
  features_csv: "features.csv"
  targets_csv: "targets.csv"

defaults:
  date_range_days: 90
  profile_minimal: false
  report_dir: "reports"
```

Notes
- The app currently uses the CSV feature store only (no DuckDB dependencies at runtime).
- You can override folder/filenames via the sidebar.
- Timestamps are normalized to UTC‑naive to match existing loaders.

## Target Recipes (No Leakage)
- forward_return(h)
  - Definition: `close.shift(-h) / close - 1`.
  - Drop last `h` rows introduced by the negative shift.
- direction_label(h, thr)
  - Definition: `1` if `forward_return(h) > thr` else `0`.
  - `thr` can be `0` (direction only) or a positive threshold.
- triple_barrier(h, tp, sl)
  - Uses path over next `h` bars: label `1` if `high` breaches `+tp`, `-1` if `low` breaches `-sl`, else `0` at horizon.
  - Ensure checks use only future bars relative to current `ts`.

All recipes operate on the OHLCV frame and return a target column aligned to current rows; the last h bars are dropped to avoid leakage.

## UI Flow
- Sidebar
  - Feature store folder and CSV filenames.
  - Date range filter (default trailing N days).
  - Feature columns multi‑select (from features.csv).
  - Target source: existing columns from targets.csv or recipe from OHLCV; parameters for the chosen recipe.
- Main Panels (separate; no joins by default)
  - OHLCV: rows/cols/time range, head, close time‑series, missingness.
  - Features: head, selected features time‑series, distributions, correlation, missingness.
  - Targets: head, selected/derived targets time‑series, distributions, correlation (if multiple), missingness.
  - Cross Analysis (optional): checkbox to enable on‑the‑fly join by timestamp for interactions and correlation between selected features and a chosen target.

## Report Export
- ydata‑profiling `ProfileReport` per panel (OHLCV, Features, Targets) and optional Joined view.
  - Saves to `reports/profile_<panel>_{dataset_id}_{YYYYMMDD_HHMM}.html` with a download button.

## Performance & UX
- Cache loaders with `st.cache_data`; cache static assets with `st.cache_resource`.
- Only generate the profiling report on explicit user action.
- Provide CSV/Parquet download of the filtered subset.
- Persist choices in `st.session_state`.

## Implementation Outline
- File layout
  - `app/streamlit_eda.py` — main UI and glue.
  - (optional) `app/eda/recipes.py` — target functions.
  - (optional) `app/eda/loader.py` — CSV loaders and helpers.
  - `configs/eda_config.yaml` — runtime config (local only; example committed as `.example`).
- Dependencies
  - `streamlit`, `ydata-profiling`, `plotly`, `pandas`, `numpy`, `pyyaml`.
- Run
  - `streamlit run app/streamlit_eda.py`.

## Guardrails
- Time leakage: always compute targets using only future bars and drop affected tail rows.
- Continuity: optionally validate hourly continuity over the selected window using utilities from `run/data_loader.py`.
- Robustness: defensive error messages for missing paths/tables/columns.

## TODO
- [x] CSV feature‑store loaders (pandas, chunked) and time range detection.
- [x] Target recipes with no leakage.
- [x] Separate OHLCV/Features/Targets panels; optional join for cross analysis.
- [x] ydata‑profiling export per panel and for joined view.
- [x] Caching and CSV downloads.

## Open Questions
- Exact DuckDB paths/tables to set as defaults for your environment?
- Preferred `feature_key` naming convention to display in the UI?
- Any must‑have targets beyond the three listed?
- Report export directory and naming pattern OK?
