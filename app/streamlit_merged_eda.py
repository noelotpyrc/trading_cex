"""
Streamlit EDA for merged features + targets CSV.
Simplified single-file version for exploring the merged dataset.

Usage:
    streamlit run app/streamlit_merged_eda.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configuration
st.set_page_config(page_title="Merged Data EDA", layout="wide")
st.title("📊 Merged Features + Targets EDA")

# Default data path
DEFAULT_DATA_PATH = "/Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_original/feature_store/merged_features_targets.csv"


# ==================== Helper Functions ====================

@st.cache_data(show_spinner=False)
def load_merged_data(path: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Load and filter merged CSV data."""
    df = pd.read_csv(path)

    # Find timestamp column
    ts_col = None
    for col in ['timestamp', 'time', 'datetime']:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is None:
        st.error("No timestamp column found")
        return pd.DataFrame()

    # Convert to datetime
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col)

    # Filter by date range (convert date objects to pd.Timestamp)
    if start_date:
        start_ts = pd.Timestamp(start_date)
        df = df[df[ts_col] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df[ts_col] <= end_ts]

    return df


def categorize_columns(df: pd.DataFrame) -> dict:
    """Categorize columns into features, targets, and metadata."""
    cols = df.columns.tolist()

    # Metadata
    metadata = [c for c in cols if c in ['timestamp', 'time', 'datetime']]

    # Targets (usually start with 'y_')
    targets = [c for c in cols if c.startswith('y_') or c.startswith('target_')]

    # Features (everything else)
    features = [c for c in cols if c not in metadata and c not in targets]

    return {
        'metadata': metadata,
        'targets': targets,
        'features': features,
    }


def group_features_by_timeframe(features: list[str]) -> dict[str, list[str]]:
    """Group features by timeframe suffix (1H, 4H, 12H, 1D)."""
    timeframes = {'1H': [], '4H': [], '12H': [], '1D': [], 'other': []}
    suffix_pattern = re.compile(r'_(1H|4H|12H|1D)$')

    for feat in features:
        match = suffix_pattern.search(feat)
        if match:
            tf = match.group(1)
            timeframes[tf].append(feat)
        else:
            timeframes['other'].append(feat)

    return timeframes


def detect_binary_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Detect binary (0/1) columns."""
    binary = []
    for col in cols:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            binary.append(col)
    return binary


def categorize_targets(df: pd.DataFrame, target_cols: list[str]) -> dict:
    """Categorize targets into numerical, binary (0/1), and ternary (1/0/-1)."""
    numerical = []
    binary = []
    ternary = []

    for col in target_cols:
        unique_vals = set(df[col].dropna().unique())

        # Check if contains -1 (ternary indicator)
        has_negative = any(v in unique_vals for v in [-1, -1.0])

        # Check value ranges
        is_ternary_range = unique_vals.issubset({-1, 0, 1, -1.0, 0.0, 1.0})
        is_binary_range = unique_vals.issubset({0, 1, 0.0, 1.0})

        # Categorize: ternary must have -1, binary only has 0/1
        if has_negative and is_ternary_range:
            ternary.append(col)
        elif is_binary_range:
            binary.append(col)
        else:
            numerical.append(col)

    return {
        'numerical': numerical,
        'binary': binary,
        'ternary': ternary
    }


def plot_time_series(df: pd.DataFrame, cols: list[str], title: str = "Time Series") -> go.Figure:
    """Create time series plot."""
    ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]

    fig = go.Figure()
    for col in cols[:10]:  # Limit to 10 series
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[ts_col],
                y=df[col],
                mode='lines',
                name=col
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified'
    )
    return fig


def plot_categorical_rolling_percentage(df: pd.DataFrame, cols: list[str], window_days: int = 90, title: str = "Rolling Percentages") -> go.Figure:
    """Create rolling percentage plot for categorical targets."""
    ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]

    fig = go.Figure()

    for col in cols[:5]:  # Limit to 5 targets
        if col not in df.columns:
            continue

        # Determine unique values
        unique_vals = sorted(df[col].dropna().unique())

        # Create rolling windows for each category
        window_hours = window_days * 24
        df_sorted = df[[ts_col, col]].copy().sort_values(ts_col)

        for val in unique_vals:
            # Create binary indicator for this value
            df_sorted[f'{col}_is_{val}'] = (df_sorted[col] == val).astype(int)

            # Calculate rolling percentage
            rolling_sum = df_sorted[f'{col}_is_{val}'].rolling(window=window_hours, min_periods=1).sum()
            rolling_count = df_sorted[col].notna().rolling(window=window_hours, min_periods=1).sum()
            rolling_pct = (rolling_sum / rolling_count * 100).fillna(0)

            # Plot
            fig.add_trace(go.Scatter(
                x=df_sorted[ts_col],
                y=rolling_pct,
                mode='lines',
                name=f'{col}={val}',
                line=dict(width=2)
            ))

    fig.update_layout(
        title=f"{title} ({window_days}-day rolling window)",
        xaxis_title="Time",
        yaxis_title="Percentage (%)",
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    return fig


def plot_histograms(df: pd.DataFrame, cols: list[str], title: str = "Distributions") -> go.Figure:
    """Create histogram subplots."""
    n_cols = len(cols)
    n_rows = (n_cols + 2) // 3

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=n_rows,
        cols=min(3, n_cols),
        subplot_titles=cols[:9]  # Limit to 9 plots
    )

    for i, col in enumerate(cols[:9]):
        row = i // 3 + 1
        col_num = i % 3 + 1

        if col in df.columns:
            fig.add_trace(
                go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                row=row, col=col_num
            )

    fig.update_layout(title=title, height=300*n_rows)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str], method: str = 'pearson', title: str = "Correlation") -> go.Figure:
    """Create correlation heatmap."""
    if len(cols) < 2:
        return go.Figure()

    corr_df = df[cols].corr(method=method)

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_df.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))

    fig.update_layout(title=title, height=max(400, len(cols) * 30))
    return fig


def compute_feature_target_correlations(df: pd.DataFrame, target_col: str, feature_cols: list[str], method: str = 'pearson') -> pd.DataFrame:
    """Compute correlations between features and target."""
    results = []
    target = pd.to_numeric(df[target_col], errors='coerce')

    for feat in feature_cols:
        feat_series = pd.to_numeric(df[feat], errors='coerce')
        mask = target.notna() & feat_series.notna()
        n = mask.sum()

        if n < 10:
            continue

        try:
            corr = target[mask].corr(feat_series[mask], method=method)
            results.append({
                'feature': feat,
                'correlation': float(corr),
                'abs_correlation': abs(float(corr)),
                'n_obs': int(n)
            })
        except:
            pass

    return pd.DataFrame(results).sort_values('abs_correlation', ascending=False)


def plot_rolling_correlation(df: pd.DataFrame, target_col: str, feature_cols: list[str], window_days: int = 90, method: str = 'pearson') -> go.Figure:
    """Create rolling correlation plot between features and target."""
    ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]

    fig = go.Figure()
    window_hours = window_days * 24

    # Ensure data is sorted by timestamp
    df_work = df.sort_values(ts_col).reset_index(drop=True)

    # Ensure timestamp is datetime
    df_work[ts_col] = pd.to_datetime(df_work[ts_col])

    target = pd.to_numeric(df_work[target_col], errors='coerce')

    for feat in feature_cols:  # Process all selected features
        if feat not in df_work.columns:
            continue

        feat_series = pd.to_numeric(df_work[feat], errors='coerce')

        # Calculate rolling correlation using pandas rolling
        # Create a combined dataframe for cleaner rolling calculation
        temp_df = pd.DataFrame({
            'target': target,
            'feature': feat_series
        })

        # Use pandas rolling with min_periods
        rolling_corr = temp_df['target'].rolling(
            window=window_hours,
            min_periods=20
        ).corr(temp_df['feature'])

        fig.add_trace(go.Scatter(
            x=df_work[ts_col],
            y=rolling_corr,
            mode='lines',
            name=feat,
            line=dict(width=2)
        ))

    fig.update_layout(
        title=f"Rolling {method.capitalize()} Correlation ({window_days}-day window)",
        xaxis_title="Time",
        yaxis_title="Correlation",
        hovermode='x unified',
        yaxis=dict(range=[-1, 1]),
        xaxis=dict(type='date')
    )

    return fig


def compute_binary_separability(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    """Compute ROC AUC and PR AUC for binary targets."""
    results = []
    y = pd.to_numeric(df[target_col], errors='coerce')
    y = y.where(y.isin([0, 1]), np.nan)

    for feat in feature_cols:
        x = pd.to_numeric(df[feat], errors='coerce')
        mask = y.notna() & x.notna()
        n = mask.sum()

        if n < 20:
            continue

        yy = y[mask].astype(int)
        xx = x[mask]

        n_pos = (yy == 1).sum()
        n_neg = (yy == 0).sum()

        if n_pos < 5 or n_neg < 5:
            continue

        # Compute ROC AUC via rank formula
        ranks = xx.rank(method='average')
        sum_ranks_pos = ranks[yy == 1].sum()
        roc_auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        # Compute PR AUC (Precision-Recall AUC)
        # Sort by feature values in descending order
        sorted_indices = xx.argsort()[::-1]
        yy_sorted = yy.iloc[sorted_indices].values

        # Calculate precision at each threshold
        cumsum_pos = np.cumsum(yy_sorted)
        total_retrieved = np.arange(1, len(yy_sorted) + 1)
        precisions = cumsum_pos / total_retrieved
        recalls = cumsum_pos / n_pos

        # Calculate PR AUC using trapezoidal rule
        pr_auc = np.trapz(precisions, recalls)

        results.append({
            'feature': feat,
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'n_obs': int(n),
            'n_positive': int(n_pos),
            'n_negative': int(n_neg)
        })

    return pd.DataFrame(results).sort_values('roc_auc', ascending=False)


def plot_rolling_auc(df: pd.DataFrame, target_col: str, feature_cols: list[str], window_days: int = 90) -> go.Figure:
    """Create rolling ROC AUC plot for binary target."""
    ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]

    fig = go.Figure()
    window_hours = window_days * 24

    # Ensure data is sorted by timestamp
    df_work = df.sort_values(ts_col).reset_index(drop=True)

    # Ensure timestamp is datetime
    df_work[ts_col] = pd.to_datetime(df_work[ts_col])

    y = pd.to_numeric(df_work[target_col], errors='coerce')
    y = y.where(y.isin([0, 1]), np.nan)

    for feat in feature_cols:  # Process all selected features
        if feat not in df_work.columns:
            continue

        x = pd.to_numeric(df_work[feat], errors='coerce')

        # Calculate rolling ROC AUC using apply on rolling windows
        def calc_auc(idx):
            if len(idx) < 20:
                return np.nan

            y_window = y.iloc[idx]
            x_window = x.iloc[idx]

            mask = y_window.notna() & x_window.notna()
            n = mask.sum()

            if n < 20:
                return np.nan

            yy = y_window[mask].astype(int)
            xx = x_window[mask]

            n_pos = (yy == 1).sum()
            n_neg = (yy == 0).sum()

            if n_pos < 5 or n_neg < 5:
                return np.nan

            # Compute ROC AUC via rank formula
            ranks = xx.rank(method='average')
            sum_ranks_pos = ranks[yy == 1].sum()
            roc_auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            return float(roc_auc)

        # Use rolling apply
        rolling_auc = pd.Series(index=df_work.index, dtype=float)
        for i in range(len(df_work)):
            start_idx = max(0, i - window_hours + 1)
            rolling_auc.iloc[i] = calc_auc(range(start_idx, i + 1))

        fig.add_trace(go.Scatter(
            x=df_work[ts_col],
            y=rolling_auc,
            mode='lines',
            name=feat,
            line=dict(width=2)
        ))

    fig.update_layout(
        title=f"Rolling ROC AUC ({window_days}-day window)",
        xaxis_title="Time",
        yaxis_title="ROC AUC",
        hovermode='x unified',
        yaxis=dict(range=[0.0, 1.0]),
        xaxis=dict(type='date')
    )

    return fig


# ==================== Streamlit App ====================

@st.cache_data(show_spinner="Pre-computing correlations...")
def precompute_correlations(df: pd.DataFrame, numerical_targets: list[str], feature_cols: list[str]) -> dict:
    """Pre-compute all feature-target correlations."""
    results = {}
    for target in numerical_targets:
        for method in ['pearson', 'spearman']:
            key = f"{target}_{method}"
            results[key] = compute_feature_target_correlations(df, target, feature_cols, method)
    return results


@st.cache_data(show_spinner="Pre-computing ROC AUC and PR AUC...")
def precompute_auc(df: pd.DataFrame, binary_targets: list[str], feature_cols: list[str]) -> dict:
    """Pre-compute ROC AUC and PR AUC for all binary targets."""
    results = {}
    for target in binary_targets:
        results[target] = compute_binary_separability(df, target, feature_cols)
    return results


@st.cache_data(show_spinner="Computing rolling metrics...")
def precompute_rolling_correlation(df: pd.DataFrame, target_col: str, feature_cols: list[str], window_days: int, method: str) -> dict:
    """Pre-compute rolling correlations for selected features."""
    results = {}
    for feat in feature_cols:  # Process all selected features
        ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        df_work = df.sort_values(ts_col).reset_index(drop=True)
        df_work[ts_col] = pd.to_datetime(df_work[ts_col])

        window_hours = window_days * 24
        target = pd.to_numeric(df_work[target_col], errors='coerce')
        feat_series = pd.to_numeric(df_work[feat], errors='coerce')

        temp_df = pd.DataFrame({
            'target': target,
            'feature': feat_series
        })

        rolling_corr = temp_df['target'].rolling(
            window=window_hours,
            min_periods=20
        ).corr(temp_df['feature'])

        results[feat] = {
            'timestamps': df_work[ts_col].values,
            'values': rolling_corr.values
        }

    return results


@st.cache_data(show_spinner="Computing rolling ROC AUC...")
def precompute_rolling_auc(df: pd.DataFrame, target_col: str, feature_cols: list[str], window_days: int) -> dict:
    """Pre-compute rolling ROC AUC for selected features."""
    results = {}
    ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
    df_work = df.sort_values(ts_col).reset_index(drop=True)
    df_work[ts_col] = pd.to_datetime(df_work[ts_col])

    window_hours = window_days * 24
    y = pd.to_numeric(df_work[target_col], errors='coerce')
    y = y.where(y.isin([0, 1]), np.nan)

    for feat in feature_cols:  # Process all selected features
        if feat not in df_work.columns:
            continue

        x = pd.to_numeric(df_work[feat], errors='coerce')

        def calc_auc(idx):
            if len(idx) < 20:
                return np.nan
            y_window = y.iloc[idx]
            x_window = x.iloc[idx]
            mask = y_window.notna() & x_window.notna()
            n = mask.sum()
            if n < 20:
                return np.nan
            yy = y_window[mask].astype(int)
            xx = x_window[mask]
            n_pos = (yy == 1).sum()
            n_neg = (yy == 0).sum()
            if n_pos < 5 or n_neg < 5:
                return np.nan
            ranks = xx.rank(method='average')
            sum_ranks_pos = ranks[yy == 1].sum()
            roc_auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            return float(roc_auc)

        rolling_auc = pd.Series(index=df_work.index, dtype=float)
        for i in range(len(df_work)):
            start_idx = max(0, i - window_hours + 1)
            rolling_auc.iloc[i] = calc_auc(range(start_idx, i + 1))

        results[feat] = {
            'timestamps': df_work[ts_col].values,
            'values': rolling_auc.values
        }

    return results


def main():
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    data_path = st.sidebar.text_input("Data path", value=DEFAULT_DATA_PATH)

    # Load full dataset
    with st.spinner("Loading dataset..."):
        df_full = load_merged_data(data_path, start_date=None, end_date=None)

    if df_full.empty:
        st.error("No data loaded")
        st.stop()

    # Get date range info
    ts_col = 'timestamp' if 'timestamp' in df_full.columns else df_full.columns[0]
    df_full[ts_col] = pd.to_datetime(df_full[ts_col])
    min_date = df_full[ts_col].min().date()
    max_date = df_full[ts_col].max().date()

    # Categorize columns
    col_groups = categorize_columns(df_full)
    tf_groups = group_features_by_timeframe(col_groups['features'])
    target_categories = categorize_targets(df_full, col_groups['targets'])
    binary_targets = target_categories['binary']

    # Display summary
    st.sidebar.success(f"✅ Loaded: {len(df_full):,} rows")
    st.sidebar.info(f"Date range: {min_date} to {max_date}\nFeatures: {len(col_groups['features'])}\nTargets: {len(col_groups['targets'])}")

    # Pre-compute all analysis results
    st.sidebar.subheader("🔄 Pre-computing Analysis")

    with st.spinner("Pre-computing correlations and AUC..."):
        # Pre-compute correlations for numerical targets
        corr_cache = {}
        if target_categories['numerical']:
            corr_cache = precompute_correlations(df_full, target_categories['numerical'], col_groups['features'])

        # Pre-compute AUC for binary targets
        auc_cache = {}
        if binary_targets:
            auc_cache = precompute_auc(df_full, binary_targets, col_groups['features'])

    st.sidebar.success("✅ Analysis ready!")

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Overview", "📊 Features", "🎯 Targets", "🔗 Correlations", "🔍 Binary Analysis", "🔗 Feature Correlations"])

    # ==================== Tab 1: Overview ====================
    with tab1:
        st.header("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{len(df_full):,}")
        col2.metric("Features", len(col_groups['features']))
        col3.metric("Targets", len(col_groups['targets']))
        col4.metric("Date Range", f"{(max_date - min_date).days} days")

        st.subheader("Column Groups")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Timeframe Distribution**")
            tf_counts = {k: len(v) for k, v in tf_groups.items() if v}
            st.bar_chart(pd.DataFrame.from_dict(tf_counts, orient='index', columns=['count']))

        with col2:
            st.write("**Target Categories**")
            target_cat_data = {
                'Category': ['Numerical', 'Binary (0/1)', 'Ternary (-1/0/1)'],
                'Count': [
                    len(target_categories['numerical']),
                    len(target_categories['binary']),
                    len(target_categories['ternary'])
                ]
            }
            st.dataframe(pd.DataFrame(target_cat_data), hide_index=True)

        st.subheader("Data Preview")
        st.dataframe(df_full.head(100))

        st.subheader("Missing Values")
        missing = df_full.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            st.dataframe(missing.head(20))
        else:
            st.success("No missing values!")

    # ==================== Tab 2: Features ====================
    with tab2:
        st.header("Feature Exploration")

        # Timeframe selector
        tf_choice = st.radio("Select Timeframe", ["1H", "4H", "12H", "1D", "All"], horizontal=True)

        if tf_choice == "All":
            available_features = col_groups['features']
        else:
            available_features = tf_groups.get(tf_choice, [])

        st.caption(f"{len(available_features)} features available")

        # Feature search
        search_term = st.text_input("🔍 Search features", placeholder="e.g., rsi, macd, volume")
        if search_term:
            available_features = [f for f in available_features if search_term.lower() in f.lower()]

        # Feature selection (no default to avoid auto-selection on rerun)
        selected_features = st.multiselect("Select features", available_features, default=[])

        if selected_features:
            st.subheader("Time Series")
            st.plotly_chart(plot_time_series(df_full, selected_features, "Feature Time Series"), use_container_width=True)

            st.subheader("Distributions")
            st.plotly_chart(plot_histograms(df_full, selected_features, "Feature Distributions"), use_container_width=True)

            if len(selected_features) >= 2:
                st.subheader("Feature Correlations")
                st.plotly_chart(plot_correlation_heatmap(df_full, selected_features, title="Feature Correlation Matrix"), use_container_width=True)
        else:
            st.info("Select features to visualize")

    # ==================== Tab 3: Targets ====================
    with tab3:
        st.header("Target Exploration")

        # Display category info
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Numerical", len(target_categories['numerical']))
        col2.metric("🔵 Binary (0/1)", len(target_categories['binary']))
        col3.metric("🟣 Ternary (-1/0/1)", len(target_categories['ternary']))

        # --- Numerical Targets ---
        if target_categories['numerical']:
            st.subheader("📊 Numerical Targets")
            selected_numerical = st.multiselect(
                "Select numerical targets",
                target_categories['numerical'],
                default=target_categories['numerical'][:1],
                key='numerical_targets'
            )

            if selected_numerical:
                st.plotly_chart(plot_time_series(df_full, selected_numerical, "Numerical Targets - Time Series"), use_container_width=True)
                st.plotly_chart(plot_histograms(df_full, selected_numerical, "Numerical Targets - Distributions"), use_container_width=True)

                if len(selected_numerical) >= 2:
                    st.plotly_chart(plot_correlation_heatmap(df_full, selected_numerical, title="Numerical Targets - Correlation"), use_container_width=True)

        # --- Binary Targets (0/1) ---
        if target_categories['binary']:
            st.subheader("🔵 Binary Targets (0/1)")
            selected_binary = st.multiselect(
                "Select binary targets",
                target_categories['binary'],
                default=target_categories['binary'][:1],
                key='binary_targets'
            )

            if selected_binary:
                # Rolling window setting
                rolling_window_binary = st.slider(
                    "Rolling window (days)",
                    min_value=7,
                    max_value=365,
                    value=90,
                    step=7,
                    key='binary_rolling'
                )

                st.plotly_chart(
                    plot_categorical_rolling_percentage(df_full, selected_binary, rolling_window_binary, "Binary Targets - Rolling Percentage"),
                    use_container_width=True
                )
                st.plotly_chart(plot_histograms(df_full, selected_binary, "Binary Targets - Distributions"), use_container_width=True)

                # Show value counts
                for col in selected_binary[:3]:
                    counts = df_full[col].value_counts(dropna=False).sort_index()
                    pct = (counts / counts.sum() * 100).round(2)
                    st.caption(f"**{col}**: " + " | ".join([f"{int(k)}={v} ({pct[k]:.1f}%)" for k, v in counts.items()]))

        # --- Ternary Targets (-1/0/1) ---
        if target_categories['ternary']:
            st.subheader("🟣 Ternary Targets (-1/0/1)")
            selected_ternary = st.multiselect(
                "Select ternary targets",
                target_categories['ternary'],
                default=target_categories['ternary'][:1],
                key='ternary_targets'
            )

            if selected_ternary:
                # Rolling window setting
                rolling_window_ternary = st.slider(
                    "Rolling window (days)",
                    min_value=7,
                    max_value=365,
                    value=90,
                    step=7,
                    key='ternary_rolling'
                )

                st.plotly_chart(
                    plot_categorical_rolling_percentage(df_full, selected_ternary, rolling_window_ternary, "Ternary Targets - Rolling Percentage"),
                    use_container_width=True
                )
                st.plotly_chart(plot_histograms(df_full, selected_ternary, "Ternary Targets - Distributions"), use_container_width=True)

                # Show value counts
                for col in selected_ternary[:3]:
                    counts = df_full[col].value_counts(dropna=False).sort_index()
                    pct = (counts / counts.sum() * 100).round(2)
                    st.caption(f"**{col}**: " + " | ".join([f"{int(k)}={v} ({pct[k]:.1f}%)" for k, v in counts.items()]))

        if not any([target_categories['numerical'], target_categories['binary'], target_categories['ternary']]):
            st.info("No targets available")

    # ==================== Tab 4: Correlations ====================
    with tab4:
        st.header("Feature ↔ Target Correlations")

        # Filter to numerical targets only
        numerical_targets = target_categories['numerical']

        if not numerical_targets:
            st.warning("No numerical targets available for correlation analysis")
        else:
            target_col = st.selectbox("Select numerical target", numerical_targets)
            corr_method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)
            min_corr = st.slider("Min |correlation|", 0.0, 1.0, 0.1, 0.01)

            # Feature set selection
            tf_choice_corr = st.radio("Feature set", ["All", "1H", "4H", "12H", "1D"], horizontal=True, key='corr_tf')
            if tf_choice_corr == "All":
                feature_pool = col_groups['features']
            else:
                feature_pool = tf_groups.get(tf_choice_corr, [])

            # Get pre-computed correlations
            cache_key = f"{target_col}_{corr_method}"
            if cache_key in corr_cache:
                corr_df = corr_cache[cache_key]

                # Filter by feature pool
                corr_df = corr_df[corr_df['feature'].isin(feature_pool)]

                if not corr_df.empty:
                    st.dataframe(corr_df.head(50))

                    # Filter by threshold
                    filtered = corr_df[corr_df['abs_correlation'] >= min_corr]
                    st.caption(f"{len(filtered)} features above threshold")

                    if not filtered.empty:
                        # Use all features above threshold (not just top 20)
                        threshold_features = filtered['feature'].tolist()
                        st.plotly_chart(
                            plot_correlation_heatmap(df_full, [target_col] + threshold_features, method=corr_method, title=f"Features Above Threshold + Target ({len(threshold_features)} features)"),
                            use_container_width=True
                        )
                else:
                    st.warning("No valid correlations computed")
            else:
                st.info("Correlations not available for this target")

            # Rolling correlation section
            st.subheader("🔄 Rolling Correlation Analysis")
            st.caption("Analyze how feature-target correlations change over time (computed on-demand)")

            # Feature selection method
            rolling_corr_selection_method = st.radio(
                "Feature selection method",
                ["Manual selection", "Load from JSON file"],
                horizontal=True,
                key='rolling_corr_method'
            )

            rolling_features = []

            if rolling_corr_selection_method == "Manual selection":
                col1, col2 = st.columns([2, 1])
                with col1:
                    rolling_features = st.multiselect(
                        f"Select features for rolling correlation (all {len(feature_pool)} available)",
                        feature_pool,
                        default=[],
                        max_selections=10,
                        key='rolling_corr_features'
                    )
                with col2:
                    rolling_window_corr = st.slider(
                        "Rolling window (days)",
                        min_value=30,
                        max_value=365,
                        value=90,
                        step=7,
                        key='rolling_corr_window'
                    )
            else:
                # JSON file input
                default_corr_fl_path = "configs/feature_lists/binance_btcusdt_p60_from_corr_since2020_1.json"
                col1, col2 = st.columns([2, 1])
                with col1:
                    rolling_corr_fl_path = st.text_input(
                        "Feature list JSON file path",
                        value=default_corr_fl_path,
                        help="Path to JSON file containing list of features",
                        key='rolling_corr_fl_path'
                    )
                with col2:
                    rolling_window_corr = st.slider(
                        "Rolling window (days)",
                        min_value=30,
                        max_value=365,
                        value=90,
                        step=7,
                        key='rolling_corr_window_json'
                    )

                # Try to load the feature list
                try:
                    with open(rolling_corr_fl_path, 'r') as f:
                        rolling_corr_fl_features = json.load(f)

                    # Validate that it's a list
                    if not isinstance(rolling_corr_fl_features, list):
                        st.error("❌ JSON file must contain a list of feature names")
                    else:
                        # Filter to only features that exist in feature_pool
                        rolling_features = [f for f in rolling_corr_fl_features if f in feature_pool]
                        missing_features_rolling_corr = [f for f in rolling_corr_fl_features if f not in feature_pool]

                        st.success(f"✅ Loaded {len(rolling_corr_fl_features)} features from file")
                        st.info(f"📊 {len(rolling_features)} features available for rolling correlation analysis")

                        if missing_features_rolling_corr:
                            with st.expander(f"⚠️ {len(missing_features_rolling_corr)} features not available in feature pool"):
                                st.write(missing_features_rolling_corr[:50])
                                if len(missing_features_rolling_corr) > 50:
                                    st.caption(f"... and {len(missing_features_rolling_corr) - 50} more")

                        # Limit to reasonable number for performance
                        if len(rolling_features) > 20:
                            st.warning(f"⚠️ {len(rolling_features)} features selected. Computing rolling correlation for many features may take a while.")
                            max_rolling_corr_features = st.slider(
                                "Max features to plot (for performance)",
                                min_value=5,
                                max_value=min(50, len(rolling_features)),
                                value=min(20, len(rolling_features)),
                                step=5,
                                key='max_rolling_corr_features'
                            )
                            rolling_features = rolling_features[:max_rolling_corr_features]
                            st.caption(f"Plotting first {len(rolling_features)} features from the list")

                except FileNotFoundError:
                    st.error(f"❌ File not found: {rolling_corr_fl_path}")
                    st.info("Please check the file path and try again")
                except json.JSONDecodeError:
                    st.error(f"❌ Invalid JSON file: {rolling_corr_fl_path}")
                    st.info("Please ensure the file contains a valid JSON array of feature names")
                except Exception as e:
                    st.error(f"❌ Error loading feature list: {e}")

            if rolling_features:
                # Compute on demand (cached per selection) - always use df_full
                rolling_corr_data = precompute_rolling_correlation(
                    df_full, target_col, rolling_features, rolling_window_corr, corr_method
                )

                # Plot using cached data
                fig = go.Figure()
                all_corr_values = []
                for feat, data in rolling_corr_data.items():
                    fig.add_trace(go.Scatter(
                        x=data['timestamps'],
                        y=data['values'],
                        mode='lines',
                        name=feat,
                        line=dict(width=2)
                    ))
                    all_corr_values.extend([v for v in data['values'] if not np.isnan(v)])

                # Calculate appropriate y-axis range
                if all_corr_values:
                    min_corr = max(-1.0, min(all_corr_values) - 0.05)
                    max_corr = min(1.0, max(all_corr_values) + 0.05)
                    # Ensure symmetry around 0 for better visualization
                    abs_max = max(abs(min_corr), abs(max_corr))
                    y_range = [-abs_max, abs_max]
                else:
                    y_range = [-1, 1]

                fig.update_layout(
                    title=f"Rolling {corr_method.capitalize()} Correlation ({rolling_window_corr}-day window)",
                    xaxis_title="Time",
                    yaxis_title="Correlation",
                    hovermode='x unified',
                    yaxis=dict(range=y_range),
                    xaxis=dict(type='date')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select features to see rolling correlation over time")

    # ==================== Tab 5: Binary Analysis ====================
    with tab5:
        st.header("Binary Target Analysis")

        if not binary_targets:
            st.warning("No binary targets detected")
        else:
            binary_target = st.selectbox("Select binary target", binary_targets)
            min_auc = st.slider("Min ROC AUC", 0.5, 1.0, 0.55, 0.01)

            # Feature set selection
            tf_choice_bin = st.radio("Feature set for analysis", ["All", "1H", "4H", "12H", "1D"], horizontal=True, key="binary_tf")
            if tf_choice_bin == "All":
                feature_pool_bin = col_groups['features']
            else:
                feature_pool_bin = tf_groups.get(tf_choice_bin, [])

            # Get pre-computed AUC
            if binary_target in auc_cache:
                sep_df = auc_cache[binary_target]

                # Filter by feature pool
                sep_df = sep_df[sep_df['feature'].isin(feature_pool_bin)]

                if not sep_df.empty:
                    st.dataframe(sep_df.head(100))

                    # Filter by threshold
                    filtered_sep = sep_df[sep_df['roc_auc'] >= min_auc]
                    st.caption(f"{len(filtered_sep)} features above ROC AUC threshold")

                    if not filtered_sep.empty:
                        # Bar chart
                        top_n = min(20, len(filtered_sep))
                        fig_bar = px.bar(
                            filtered_sep.head(top_n),
                            x='feature',
                            y='roc_auc',
                            title=f"Top {top_n} Features by ROC AUC",
                            range_y=[0.5, 1.0]
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("No valid AUC scores computed")
            else:
                st.info("AUC not available for this target")

            # Feature list analysis section
            st.subheader("📋 Feature List AUC Analysis")
            st.caption("Load a feature list JSON and analyze AUC metrics for those features")

            # Feature list file selector
            default_fl_path = "configs/feature_lists/binance_btcusdt_p60_from_corr_since2020_1.json"
            feature_list_path_auc = st.text_input(
                "Feature list JSON file path",
                value=default_fl_path,
                help="Path to JSON file containing list of features to analyze",
                key='feature_list_auc_path'
            )

            # Try to load the feature list
            try:
                with open(feature_list_path_auc, 'r') as f:
                    fl_features = json.load(f)

                # Validate that it's a list
                if not isinstance(fl_features, list):
                    st.error("❌ JSON file must contain a list of feature names")
                else:
                    st.success(f"✅ Loaded {len(fl_features)} features from file")

                    # Get pre-computed AUC for this binary target
                    if binary_target in auc_cache:
                        sep_df_fl = auc_cache[binary_target]

                        # Filter to only features in the loaded list
                        sep_df_fl_filtered = sep_df_fl[sep_df_fl['feature'].isin(fl_features)]

                        # Check which features are missing
                        features_with_auc = set(sep_df_fl_filtered['feature'].tolist())
                        missing_from_data = [f for f in fl_features if f not in features_with_auc]

                        if not sep_df_fl_filtered.empty:
                            # Display summary statistics
                            st.subheader("Summary Statistics")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Features", len(sep_df_fl_filtered))
                            col2.metric("Mean AUC", f"{sep_df_fl_filtered['roc_auc'].mean():.4f}")
                            col3.metric("Median AUC", f"{sep_df_fl_filtered['roc_auc'].median():.4f}")
                            col4.metric("Min AUC", f"{sep_df_fl_filtered['roc_auc'].min():.4f}")
                            col5.metric("Max AUC", f"{sep_df_fl_filtered['roc_auc'].max():.4f}")

                            # Show distribution of AUC scores
                            st.subheader("AUC Distribution")
                            fig_hist = go.Figure()
                            fig_hist.add_trace(go.Histogram(
                                x=sep_df_fl_filtered['roc_auc'],
                                nbinsx=20,
                                name='ROC AUC',
                                marker_color='steelblue'
                            ))
                            fig_hist.update_layout(
                                title="Distribution of ROC AUC Scores",
                                xaxis_title="ROC AUC",
                                yaxis_title="Count",
                                showlegend=False
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                            # Show full table
                            st.subheader("Detailed Results")
                            st.dataframe(sep_df_fl_filtered, height=400)

                            # Download button for results
                            csv = sep_df_fl_filtered.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"feature_auc_{binary_target}_{Path(feature_list_path_auc).stem}.csv",
                                mime="text/csv"
                            )

                            # Bar chart of all features
                            if len(sep_df_fl_filtered) <= 50:
                                st.subheader("ROC AUC by Feature")
                                fig_bar_fl = px.bar(
                                    sep_df_fl_filtered.sort_values('roc_auc', ascending=True),
                                    x='roc_auc',
                                    y='feature',
                                    orientation='h',
                                    title=f"ROC AUC for {len(sep_df_fl_filtered)} Features",
                                    height=max(400, len(sep_df_fl_filtered) * 20)
                                )
                                fig_bar_fl.update_layout(
                                    xaxis_range=[0.45, max(0.65, sep_df_fl_filtered['roc_auc'].max() + 0.02)]
                                )
                                st.plotly_chart(fig_bar_fl, use_container_width=True)
                            else:
                                st.info(f"Too many features ({len(sep_df_fl_filtered)}) to display bar chart. Showing table only.")

                            # Show missing features
                            if missing_from_data:
                                with st.expander(f"⚠️ {len(missing_from_data)} features not found in dataset or have no AUC"):
                                    st.write(missing_from_data[:50])
                                    if len(missing_from_data) > 50:
                                        st.caption(f"... and {len(missing_from_data) - 50} more")
                        else:
                            st.warning("No features from the list have AUC scores for this target")
                            if missing_from_data:
                                st.info(f"{len(missing_from_data)} features from the list are not available")
                    else:
                        st.warning("No AUC data available for this binary target")

            except FileNotFoundError:
                st.error(f"❌ File not found: {feature_list_path_auc}")
                st.info("Please check the file path and try again")
            except json.JSONDecodeError:
                st.error(f"❌ Invalid JSON file: {feature_list_path_auc}")
                st.info("Please ensure the file contains a valid JSON array of feature names")
            except Exception as e:
                st.error(f"❌ Error loading feature list: {e}")

            st.divider()

            # Rolling AUC section
            st.subheader("🔄 Rolling AUC Analysis")
            st.caption("Analyze how ROC AUC changes over time (computed on-demand)")

            # Feature selection method
            rolling_selection_method = st.radio(
                "Feature selection method",
                ["Manual selection", "Load from JSON file"],
                horizontal=True,
                key='rolling_auc_method'
            )

            rolling_features_auc = []

            if rolling_selection_method == "Manual selection":
                col1, col2 = st.columns([2, 1])
                with col1:
                    rolling_features_auc = st.multiselect(
                        f"Select features for rolling AUC (all {len(feature_pool_bin)} available)",
                        feature_pool_bin,
                        default=[],
                        max_selections=10,
                        key='rolling_auc_features'
                    )
                with col2:
                    rolling_window_auc = st.slider(
                        "Rolling window (days)",
                        min_value=30,
                        max_value=365,
                        value=90,
                        step=7,
                        key='rolling_auc_window'
                    )
            else:
                # JSON file input
                col1, col2 = st.columns([2, 1])
                with col1:
                    rolling_fl_path = st.text_input(
                        "Feature list JSON file path",
                        value=default_fl_path,
                        help="Path to JSON file containing list of features",
                        key='rolling_auc_fl_path'
                    )
                with col2:
                    rolling_window_auc = st.slider(
                        "Rolling window (days)",
                        min_value=30,
                        max_value=365,
                        value=90,
                        step=7,
                        key='rolling_auc_window_json'
                    )

                # Try to load the feature list
                try:
                    with open(rolling_fl_path, 'r') as f:
                        rolling_fl_features = json.load(f)

                    # Validate that it's a list
                    if not isinstance(rolling_fl_features, list):
                        st.error("❌ JSON file must contain a list of feature names")
                    else:
                        # Filter to only features that exist in feature_pool_bin
                        rolling_features_auc = [f for f in rolling_fl_features if f in feature_pool_bin]
                        missing_features_rolling = [f for f in rolling_fl_features if f not in feature_pool_bin]

                        st.success(f"✅ Loaded {len(rolling_fl_features)} features from file")
                        st.info(f"📊 {len(rolling_features_auc)} features available for rolling AUC analysis")

                        if missing_features_rolling:
                            with st.expander(f"⚠️ {len(missing_features_rolling)} features not available in feature pool"):
                                st.write(missing_features_rolling[:50])
                                if len(missing_features_rolling) > 50:
                                    st.caption(f"... and {len(missing_features_rolling) - 50} more")

                        # Limit to reasonable number for performance
                        if len(rolling_features_auc) > 20:
                            st.warning(f"⚠️ {len(rolling_features_auc)} features selected. Computing rolling AUC for many features may take a while.")
                            max_rolling_features = st.slider(
                                "Max features to plot (for performance)",
                                min_value=5,
                                max_value=min(50, len(rolling_features_auc)),
                                value=min(20, len(rolling_features_auc)),
                                step=5,
                                key='max_rolling_auc_features'
                            )
                            rolling_features_auc = rolling_features_auc[:max_rolling_features]
                            st.caption(f"Plotting first {len(rolling_features_auc)} features from the list")

                except FileNotFoundError:
                    st.error(f"❌ File not found: {rolling_fl_path}")
                    st.info("Please check the file path and try again")
                except json.JSONDecodeError:
                    st.error(f"❌ Invalid JSON file: {rolling_fl_path}")
                    st.info("Please ensure the file contains a valid JSON array of feature names")
                except Exception as e:
                    st.error(f"❌ Error loading feature list: {e}")

            if rolling_features_auc:
                # Compute on demand (cached per selection) - always use df_full
                rolling_auc_data = precompute_rolling_auc(
                    df_full, binary_target, rolling_features_auc, rolling_window_auc
                )

                # Plot using cached data
                fig = go.Figure()
                all_values = []
                for feat, data in rolling_auc_data.items():
                    fig.add_trace(go.Scatter(
                        x=data['timestamps'],
                        y=data['values'],
                        mode='lines',
                        name=feat,
                        line=dict(width=2)
                    ))
                    all_values.extend([v for v in data['values'] if not np.isnan(v)])

                # Calculate appropriate y-axis range
                if all_values:
                    min_val = max(0.0, min(all_values) - 0.02)
                    max_val = min(1.0, max(all_values) + 0.02)
                else:
                    min_val, max_val = 0.0, 1.0

                fig.update_layout(
                    title=f"Rolling ROC AUC ({rolling_window_auc}-day window)",
                    xaxis_title="Time",
                    yaxis_title="ROC AUC",
                    hovermode='x unified',
                    yaxis=dict(range=[min_val, max_val]),
                    xaxis=dict(type='date')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select features to see rolling ROC AUC over time")

    # ==================== Tab 6: Feature Correlations ====================
    with tab6:
        st.header("Feature-Feature Correlations")
        st.caption("Explore correlations between features using a predefined feature list")

        # Feature list file selector
        default_feature_file = "configs/feature_lists/binance_btcusdt_p60_default.json"
        feature_list_path = st.text_input(
            "Feature list JSON file path",
            value=default_feature_file,
            help="Path to JSON file containing list of features to analyze"
        )

        # Try to load the feature list
        try:
            with open(feature_list_path, 'r') as f:
                selected_feature_list = json.load(f)

            # Filter to only features that exist in the dataset
            available_in_data = [f for f in selected_feature_list if f in df_full.columns]
            missing_features = [f for f in selected_feature_list if f not in df_full.columns]

            st.success(f"✅ Loaded {len(selected_feature_list)} features from file")
            st.info(f"📊 {len(available_in_data)} features available in dataset, {len(missing_features)} missing")

            if missing_features and len(missing_features) <= 10:
                with st.expander("Show missing features"):
                    st.write(missing_features)
            elif missing_features:
                with st.expander(f"Show missing features ({len(missing_features)} total)"):
                    st.write(missing_features[:50])
                    if len(missing_features) > 50:
                        st.caption(f"... and {len(missing_features) - 50} more")

            if not available_in_data:
                st.warning("No features from the list are available in the dataset")
            else:
                # Correlation method
                corr_method_feat = st.radio(
                    "Correlation method",
                    ["pearson", "spearman"],
                    horizontal=True,
                    key='feat_corr_method'
                )

                # Subsample slider (for performance with many features)
                if len(available_in_data) > 50:
                    st.warning(f"⚠️ {len(available_in_data)} features selected. Large correlation matrices may be slow to compute and visualize.")
                    max_features_to_show = st.slider(
                        "Max features to display (for performance)",
                        min_value=10,
                        max_value=min(200, len(available_in_data)),
                        value=min(50, len(available_in_data)),
                        step=10
                    )
                    features_to_plot = available_in_data[:max_features_to_show]
                    st.caption(f"Showing first {len(features_to_plot)} features from the list")
                else:
                    features_to_plot = available_in_data

                # Compute correlation matrix
                with st.spinner(f"Computing {corr_method_feat} correlation matrix for {len(features_to_plot)} features..."):
                    corr_matrix = df_full[features_to_plot].corr(method=corr_method_feat)

                # Display correlation heatmap
                st.subheader(f"Correlation Heatmap ({len(features_to_plot)} features)")

                # Create heatmap with better sizing
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}' if len(features_to_plot) <= 20 else None,
                    textfont={"size": 8},
                    hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>'
                ))

                # Adjust height based on number of features
                height = max(600, len(features_to_plot) * 15)
                fig.update_layout(
                    title=f"{corr_method_feat.capitalize()} Correlation Matrix",
                    height=height,
                    xaxis={'side': 'bottom', 'tickangle': 45},
                    yaxis={'side': 'left'}
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show highly correlated pairs
                st.subheader("Highly Correlated Feature Pairs")
                threshold = st.slider(
                    "Min |correlation| to display",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key='feat_corr_threshold'
                )

                # Extract upper triangle (avoid duplicates and self-correlation)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                corr_pairs = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) >= threshold:
                            corr_pairs.append({
                                'feature_1': corr_matrix.index[i],
                                'feature_2': corr_matrix.columns[j],
                                'correlation': float(corr_val),
                                'abs_correlation': abs(float(corr_val))
                            })

                if corr_pairs:
                    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('abs_correlation', ascending=False)
                    st.dataframe(corr_pairs_df, height=400)
                    st.caption(f"Found {len(corr_pairs_df)} feature pairs with |correlation| >= {threshold}")
                else:
                    st.info(f"No feature pairs found with |correlation| >= {threshold}")

        except FileNotFoundError:
            st.error(f"❌ File not found: {feature_list_path}")
            st.info("Please check the file path and try again")
        except json.JSONDecodeError:
            st.error(f"❌ Invalid JSON file: {feature_list_path}")
            st.info("Please ensure the file contains a valid JSON array of feature names")
        except Exception as e:
            st.error(f"❌ Error loading feature list: {e}")


if __name__ == "__main__":
    main()
