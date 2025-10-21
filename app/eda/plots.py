from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def time_series(df: pd.DataFrame, cols: Iterable[str], *, title: str = "Time Series") -> go.Figure:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = []
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[c], mode="lines", name=c))
    fig.update_layout(title=title, xaxis_title="timestamp", yaxis_title="value", height=420)
    return fig


def histograms(df: pd.DataFrame, cols: Iterable[str], *, nbins: int = 50, title: str = "Distributions") -> go.Figure:
    cols = [c for c in cols if c in df.columns]
    if len(cols) == 1:
        return px.histogram(df, x=cols[0], nbins=nbins, title=title)
    # Facet by column for multiple
    melted = df[cols].melt(var_name="variable", value_name="value")
    fig = px.histogram(melted, x="value", facet_col="variable", nbins=nbins, title=title)
    fig.update_layout(height=340 + 120 * max(0, (len(cols) - 1) // 3))
    return fig


def corr_heatmap(df: pd.DataFrame, cols: Iterable[str], *, method: str = "pearson", title: str = "Correlation") -> go.Figure:
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return go.Figure()
    corr = df[cols].corr(method=method).round(3)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title=title, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=500)
    return fig


def scatter_xy(df: pd.DataFrame, x: str, y: str, *, title: str = "Scatter") -> go.Figure:
    if x not in df.columns or y not in df.columns:
        return go.Figure()
    fig = px.scatter(df, x=x, y=y, opacity=0.6, title=title)
    fig.update_layout(height=420)
    return fig


def missingness(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame({"column": [], "missing_pct": []})
    pct = df[cols].isna().mean().sort_values(ascending=False)
    out = pct.reset_index()
    # For pandas <2.1, reset_index doesn't support 'names'; set columns manually
    out.columns = ["column", "missing_pct"]
    out["missing_pct"] = (out["missing_pct"] * 100).round(2)
    return out
