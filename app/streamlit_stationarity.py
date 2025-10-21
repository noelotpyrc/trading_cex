from __future__ import annotations

"""
Streamlit Stationarity Lab (Placeholder)

This app is a placeholder scaffold for a dedicated stationarity analysis UI.
It intentionally does not implement the heavy tests yet. The comments below
outline what will be built, inputs/outputs, and UX flow.

Planned capabilities:

1) Data selection
   - Read from the existing CSV feature store (ohlcv.csv, features.csv, targets.csv).
   - Date range selector and timeframe filter (1H/4H/12H/1D) for feature columns.
   - Search/autocomplete for large feature/target lists (reuse patterns from EDA app).
   - No implicit joins by default; join on timestamp only when the user enables it.

2) Transformations (apply to a chosen series)
   - Differencing: diff(1), seasonal diff (e.g., diff(24), diff(24*7) for hourly).
   - Log/log1p, z-score standardization (global or rolling), winsorization/clipping.
   - Volatility standardization for returns: divide by rolling volatility.
   - Preview transformed series alongside original before running tests.

3) Visual diagnostics
   - Rolling mean/std plots (window selectable, e.g., 24/168/504).
   - ACF/PACF plots for original vs transformed.
   - Q–Q plot, histogram/KDE overlays by time-slice.
   - CUSUM/CUSUMSQ and change-point markers (placeholder if library not present).

4) Unit-root and stationarity tests (univariate)
   - ADF (Augmented Dickey–Fuller): H0 non-stationary (unit root).
   - KPSS: H0 stationary; complements ADF.
   - PP (Phillips–Perron) and DF-GLS for more power/robustness.
   - Zivot–Andrews for a single structural break in level/trend.
   - ARCH LM test for conditional heteroskedasticity (variance stationarity indication).
   - All tests will report p-values, selected regression terms (c/ct), lags,
     sample size used (post-transform), and simple pass/fail flags with caveats.

5) Multivariate checks (optional advanced)
   - Cointegration (Engle–Granger, Johansen) for pairs/sets of series.
   - Residual stationarity tests on spreads or linear combinations.

6) Targets-specific checks
   - For binary targets: drift tests on event rate (windowed proportions, chi-square),
     PSI across time bins, and rolling metrics.
   - For continuous targets (returns): ARCH effects, volatility clustering indicators.

7) Batch mode + export
   - Run tests across multiple selected columns (features/targets) in one go.
   - Render a concise result table with p-values and flags.
   - Export results as CSV and a light HTML report (no heavy dependencies required).

Dependencies (to be used when implemented):
   - statsmodels (adfuller, kpss, acf/pacf)
   - arch (PhillipsPerron, DFGLS, ZivotAndrews, ARCH LM)
   - plotly/matplotlib for ACF/PACF/rolling plots

"""

import sys
from pathlib import Path

import pandas as pd  # noqa: F401  # placeholder for future use
import streamlit as st

# Ensure local imports work when run via `streamlit run app/streamlit_stationarity.py`
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (THIS_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


st.set_page_config(page_title="Stationarity Lab (Placeholder)", layout="wide")
st.title("Stationarity Lab — Placeholder")

st.info(
    "This is a placeholder for a focused stationarity analysis UI."
    " It sketches the intended features via comments and section stubs."
)


with st.sidebar:
    st.header("Configuration")
    st.text_input(
        "Feature store folder",
        value=str(
            Path("/Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_tradingview/feature_store")
        ),
        help="Folder containing ohlcv.csv, features.csv, targets.csv",
    )
    st.date_input("Window (start, end)")
    st.radio("Timeframe", ["1H", "4H", "12H", "1D"], index=0, horizontal=True)
    st.text_input("Search (features/targets)", value="", help="Type to filter large lists")


st.subheader("1) Data Selection (Planned)")
st.markdown(
    "- Load ohlcv.csv, features.csv, targets.csv; no implicit joins by default.\n"
    "- Filter by date/timeframe; search/regex for column selection.\n"
    "- Join on timestamp only when user enables cross-series analysis."
)

st.subheader("2) Transformations (Planned)")
st.markdown(
    "- Differencing: diff(1), seasonal diff (diff 24/168).\n"
    "- Log/log1p; z-score; rolling-volatility standardization for returns.\n"
    "- Preview transformed series vs original."
)

st.subheader("3) Visual Diagnostics (Planned)")
st.markdown(
    "- Rolling mean/std; ACF/PACF; Q–Q and histogram/KDE overlays.\n"
    "- CUSUM/CUSUMSQ; basic change-point indicators (if available)."
)

st.subheader("4) Unit-Root/Stationarity Tests (Planned)")
st.markdown(
    "- ADF, KPSS, PP, DF-GLS, Zivot–Andrews; ARCH LM.\n"
    "- Configurable regression terms (c/ct), lags, window size; summary table with p-values."
)

st.subheader("5) Multivariate/Targets (Planned)")
st.markdown(
    "- Cointegration (EG/Johansen) for selected sets.\n"
    "- Binary target drift tests (proportion/CUSUM/PSI); continuous target volatility checks."
)

st.subheader("6) Export (Planned)")
st.markdown(
    "- Export test results CSV and a lightweight HTML report.\n"
    "- Optional saved configuration for reproducibility."
)

st.success("Placeholder ready. When you’re ready, I can implement step-by-step.")

