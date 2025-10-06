#!/usr/bin/env python3
"""Generate global SHAP contribution summary report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from model.shap.shap_utils import (
    FeatureSourceConfig,
    ensure_timestamp_alignment,
    load_feature_dataframe,
    load_shap_dataframe,
)


META_COLUMNS = {"timestamp", "y_true", "y_pred", "base_value", "prediction"}


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _feature_columns(shap_df) -> List[str]:
    cols: List[str] = []
    for col in shap_df.columns:
        if col in META_COLUMNS:
            continue
        if np.issubdtype(shap_df[col].dtype, np.number):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns found in SHAP dataframe")
    return cols


def _build_bar_figure(feature_rank) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=feature_rank.index.tolist(),
                y=feature_rank.values,
                marker_color="#1f77b4",
            )
        ]
    )
    fig.update_layout(
        title="Top Features by Mean |SHAP|",
        xaxis_title="Feature",
        yaxis_title="Mean |SHAP|",
        xaxis_tickangle=-45,
        template="plotly_white",
    )
    return fig


def _build_scatter_figure(merged_df, top_features: List[str]) -> go.Figure:
    fig = go.Figure()
    buttons = []
    timestamp_series = merged_df["timestamp"].astype(str)
    y_pred_series = merged_df.get("y_pred")

    for idx, feature in enumerate(top_features):
        value_col = f"{feature}_feat"
        if value_col not in merged_df.columns:
            raise KeyError(
                f"Feature values column '{value_col}' missing. Ensure feature data contains '{feature}'."
            )
        if y_pred_series is not None:
            customdata = np.column_stack((timestamp_series.to_numpy(), y_pred_series.to_numpy()))
            hovertemplate = (
                "Feature value=%{x:.4f}<br>SHAP=%{y:.4f}<br>"
                "timestamp=%{customdata[0]}<br>prediction=%{customdata[1]:.4f}<extra></extra>"
            )
        else:
            customdata = timestamp_series.to_numpy().reshape(-1, 1)
            hovertemplate = (
                "Feature value=%{x:.4f}<br>SHAP=%{y:.4f}<br>"
                "timestamp=%{customdata[0]}<extra></extra>"
            )
        fig.add_trace(
            go.Scattergl(
                x=merged_df[value_col],
                y=merged_df[feature],
                mode="markers",
                name=feature,
                marker=dict(size=6, opacity=0.6),
                visible=(idx == 0),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

        visibility = [False] * len(top_features)
        visibility[idx] = True
        buttons.append(
            dict(
                label=feature,
                method="update",
                args=[{"visible": visibility}, {"title": f"SHAP Contribution vs Value: {feature}"}],
            )
        )

    fig.update_layout(
        title=f"SHAP Contribution vs Feature Value: {top_features[0]}",
        xaxis_title="Feature value",
        yaxis_title="SHAP contribution",
        template="plotly_white",
        updatemenus=[
            dict(
                type="dropdown",
                showactive=True,
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.2,
            )
        ],
    )
    return fig


def _render_html(bar_fig: go.Figure, scatter_fig: go.Figure, title: str) -> str:
    bar_html = pio.to_html(bar_fig, include_plotlyjs="cdn", full_html=False)
    scatter_html = pio.to_html(scatter_fig, include_plotlyjs=False, full_html=False)
    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    h1 {{ margin-bottom: 0.5rem; }}
    section {{ margin-bottom: 3rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <section>
    <h2>Top Feature Contributions</h2>
    {bar_html}
  </section>
  <section>
    <h2>Contribution vs Feature Value</h2>
    <p>Select a feature from the dropdown to explore how its value relates to its SHAP contribution.</p>
    {scatter_html}
  </section>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create global SHAP contribution report")
    parser.add_argument("--shap-values", type=Path, required=True, help="Path to shap_values_<tag>.parquet")

    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--features-csv", type=Path, help="CSV containing feature values with timestamp column")
    feature_group.add_argument("--duckdb", type=Path, help="DuckDB database storing feature snapshots")

    parser.add_argument("--feature-key", type=str, help="Feature key when reading from DuckDB")
    parser.add_argument("--duckdb-table", type=str, default="features", help="DuckDB table name")
    parser.add_argument("--duckdb-timestamp-col", type=str, default="ts", help="DuckDB timestamp column")
    parser.add_argument("--duckdb-features-col", type=str, default="features", help="DuckDB features JSON column")
    parser.add_argument("--timestamp-column", type=str, default="timestamp", help="Timestamp column name in CSV/merged data")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to plot")
    parser.add_argument("--output", type=Path, help="Output HTML path (defaults to shap-values directory)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)

    shap_path = args.shap_values
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP values file not found: {shap_path}")

    shap_df = load_shap_dataframe(shap_path)
    feature_cols = _feature_columns(shap_df)

    cfg = FeatureSourceConfig(
        csv_path=args.features_csv,
        timestamp_column=args.timestamp_column,
        duckdb_path=args.duckdb,
        duckdb_table=args.duckdb_table,
        duckdb_feature_key=args.feature_key,
        duckdb_timestamp_column=args.duckdb_timestamp_col,
        duckdb_features_column=args.duckdb_features_col,
    )
    features_df = load_feature_dataframe(cfg)

    merged_df = ensure_timestamp_alignment(shap_df, features_df, timestamp_column=args.timestamp_column)

    mean_abs = merged_df[feature_cols].abs().mean().sort_values(ascending=False)
    top_features = mean_abs.head(args.top_n)
    if top_features.empty:
        raise ValueError("No features available after ranking computation")

    logging.info("Top features: %s", ", ".join(top_features.index.tolist()))

    bar_fig = _build_bar_figure(top_features)
    scatter_fig = _build_scatter_figure(merged_df, top_features.index.tolist())

    title = f"Global SHAP Contribution Report ({shap_path.stem})"
    html = _render_html(bar_fig, scatter_fig, title)

    output_path = args.output or shap_path.with_name(f"{shap_path.stem}_global_report.html")
    output_path.write_text(html, encoding="utf-8")
    logging.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
