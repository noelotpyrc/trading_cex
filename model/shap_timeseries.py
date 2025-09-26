#!/usr/bin/env python3
"""Produce time-series SHAP analysis with interactive plots."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from model.shap_utils import (
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


def _build_prediction_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "y_pred" in df.columns:
        y_pred_series = df["y_pred"]
    elif "prediction" in df.columns:
        y_pred_series = df["prediction"]
    else:
        raise KeyError("SHAP data must contain a 'y_pred' or 'prediction' column for time-series plot")

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=y_pred_series,
            mode="lines",
            name="Prediction",
            line=dict(color="#1f77b4"),
        )
    )

    if "y_true" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["y_true"],
                mode="lines",
                name="Target",
                line=dict(color="#ff7f0e", dash="dash"),
            )
        )

    if "base_value" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=np.full(len(df), df["base_value"].iloc[0]),
                mode="lines",
                name="Base value",
                line=dict(color="#2ca02c", dash="dot"),
            )
        )

    fig.update_layout(
        title="Predictions over Time",
        xaxis_title="Timestamp",
        yaxis_title="Prediction",
        template="plotly_white",
    )
    return fig


def _load_targets_from_csv(path: Path, timestamp_col: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise KeyError(f"Targets CSV missing timestamp column '{timestamp_col}'")
    if value_col not in df.columns:
        raise KeyError(f"Targets CSV missing value column '{value_col}'")
    targets = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[timestamp_col], errors="coerce", utc=True),
            "y_true": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    targets = targets.dropna(subset=["timestamp"])
    return targets


def _load_targets_from_duckdb(
    db_path: Path,
    table: str,
    target_name: Optional[str],
    feature_key: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb must be installed to fetch targets from database") from exc

    clauses = []
    params: list = []
    if target_name:
        clauses.append("target_name = ?")
        params.append(target_name)
    if feature_key:
        clauses.append("feature_key = ?")
        params.append(feature_key)
    if start:
        clauses.append("ts >= ?")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end:
        clauses.append("ts <= ?")
        params.append(pd.Timestamp(end).to_pydatetime())

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        f"SELECT ts AS timestamp, y_true FROM {table}{where} ORDER BY ts ASC"
    )
    with duckdb.connect(str(db_path)) as con:
        con.execute("SET TimeZone='UTC';")
        df = con.execute(sql, params).fetch_df()

    if df.empty:
        logging.warning("Targets query returned no rows from %s", table)
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df.rename(columns={"y_true": "y_true"}, inplace=True)
    return df


def _build_contribution_figure(df: pd.DataFrame, top_features: List[str]) -> go.Figure:
    fig = go.Figure()
    buttons = []

    for idx, feature in enumerate(top_features):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[feature],
                mode="lines",
                name=feature,
                visible=(idx == 0),
            )
        )
        visibility = [False] * len(top_features)
        visibility[idx] = True
        buttons.append(
            dict(
                label=feature,
                method="update",
                args=[{"visible": visibility}, {"title": f"SHAP Contribution over Time: {feature}"}],
            )
        )

    fig.update_layout(
        title=f"SHAP Contribution over Time: {top_features[0]}",
        xaxis_title="Timestamp",
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


def _build_contribution_table_htmls(
    merged_df: pd.DataFrame,
    feature_cols: List[str],
    top_n: int,
) -> Dict[str, str]:
    tables: Dict[str, str] = {}
    grouped = merged_df.set_index("timestamp")
    for ts, row in grouped.iterrows():
        records = []
        for feature in feature_cols:
            value_col = f"{feature}_feat"
            value = row.get(value_col, np.nan)
            shap_val = row[feature]
            records.append(
                {
                    "feature": feature,
                    "shap": shap_val,
                    "abs_shap": abs(shap_val),
                    "value": value,
                }
            )

        df = (
            pd.DataFrame(records)
            .sort_values("abs_shap", ascending=False)
            .head(top_n)
            .drop(columns=["abs_shap"])
        )
        tables[ts.isoformat()] = df.to_html(index=False, float_format=lambda x: f"{x:.6f}")
    return tables


def _render_html(
    pred_fig: go.Figure,
    contrib_fig: go.Figure,
    tables: Dict[str, str],
    title: str,
    timestamps: List[str],
    initial_ts: str,
) -> str:
    pred_html = pio.to_html(pred_fig, include_plotlyjs="cdn", full_html=False)
    contrib_html = pio.to_html(contrib_fig, include_plotlyjs=False, full_html=False)
    select_options = "\n".join(
        f"<option value=\"{ts}\"{' selected' if ts == initial_ts else ''}>{ts}</option>"
        for ts in timestamps
    )
    tables_json = json.dumps(tables)
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
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th {{ background-color: #f2f2f2; text-align: left; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <section>
    <h2>Predictions vs Timestamp</h2>
    {pred_html}
  </section>
  <section>
    <h2>Feature Contribution over Time</h2>
    <p>Use the dropdown to focus on individual features.</p>
    {contrib_html}
  </section>
  <section>
    <h2>Top Contributors</h2>
    <label for=\"timestamp-select\">Select timestamp:</label>
    <select id=\"timestamp-select\">
      {select_options}
    </select>
    <div id=\"contrib-table\"></div>
  </section>
  <script>
    const tables = {tables_json};
    const selectEl = document.getElementById('timestamp-select');
    const tableEl = document.getElementById('contrib-table');
    function renderTable(ts) {{
      tableEl.innerHTML = tables[ts] || '<p>No data for selected timestamp.</p>';
    }}
    selectEl.addEventListener('change', (event) => renderTable(event.target.value));
    renderTable('{initial_ts}');
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create time-series SHAP analysis report")
    parser.add_argument("--shap-values", type=Path, required=True, help="Path to shap_values_<tag>.parquet")
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--features-csv", type=Path, help="CSV containing feature values with timestamp column")
    feature_group.add_argument("--duckdb", type=Path, help="DuckDB database storing feature snapshots")

    parser.add_argument("--feature-key", type=str, help="Feature key when reading from DuckDB")
    parser.add_argument("--duckdb-table", type=str, default="features", help="DuckDB table name")
    parser.add_argument("--duckdb-timestamp-col", type=str, default="ts", help="DuckDB timestamp column")
    parser.add_argument("--duckdb-features-col", type=str, default="features", help="DuckDB features JSON column")
    parser.add_argument("--targets-csv", type=Path, help="Optional CSV providing y_true values")
    parser.add_argument("--targets-timestamp-col", type=str, default="timestamp")
    parser.add_argument("--targets-value-col", type=str, default="y_true")
    parser.add_argument("--targets-duckdb", type=Path, help="DuckDB database containing inference targets")
    parser.add_argument("--targets-table", type=str, default="inference_targets")
    parser.add_argument("--targets-name", type=str, help="Filter DuckDB targets by target_name")
    parser.add_argument("--targets-feature-key", type=str, help="Filter DuckDB targets by feature_key")
    parser.add_argument("--targets-start", type=str, help="Lower bound timestamp for DuckDB targets")
    parser.add_argument("--targets-end", type=str, help="Upper bound timestamp for DuckDB targets")
    parser.add_argument("--timestamp-column", type=str, default="timestamp", help="Timestamp column name in CSV/merged data")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top features to include")
    parser.add_argument("--focus-timestamp", type=str, help="Timestamp (ISO) to display initially")
    parser.add_argument("--output", type=Path, help="Output HTML path (defaults near shap-values)")
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

    if args.targets_csv and args.targets_duckdb:
        raise ValueError("Specify either --targets-csv or --targets-duckdb, not both")

    targets_df: Optional[pd.DataFrame] = None
    if args.targets_csv:
        if not args.targets_csv.exists():
            raise FileNotFoundError(f"Targets CSV not found: {args.targets_csv}")
        targets_df = _load_targets_from_csv(
            args.targets_csv,
            timestamp_col=args.targets_timestamp_col,
            value_col=args.targets_value_col,
        )
    elif args.targets_duckdb:
        if not args.targets_duckdb.exists():
            raise FileNotFoundError(f"Targets DuckDB not found: {args.targets_duckdb}")
        targets_df = _load_targets_from_duckdb(
            args.targets_duckdb,
            table=args.targets_table,
            target_name=args.targets_name,
            feature_key=args.targets_feature_key,
            start=args.targets_start,
            end=args.targets_end,
        )

    if targets_df is not None and not targets_df.empty:
        merged_df = merged_df.merge(targets_df, on="timestamp", how="left", suffixes=("", "_target"))
        if "y_true_target" in merged_df.columns:
            if "y_true" not in merged_df.columns:
                merged_df.rename(columns={"y_true_target": "y_true"}, inplace=True)
            else:
                merged_df["y_true"] = merged_df["y_true"].fillna(merged_df["y_true_target"])
                merged_df.drop(columns=["y_true_target"], inplace=True)

    mean_abs = merged_df[feature_cols].abs().mean().sort_values(ascending=False)
    top_features = mean_abs.head(args.top_n).index.tolist()
    if not top_features:
        raise ValueError("No features available for time-series plot")

    pred_fig = _build_prediction_figure(merged_df)
    contrib_fig = _build_contribution_figure(merged_df, top_features)

    timestamps = sorted({ts.isoformat() for ts in merged_df["timestamp"].unique()})
    if not timestamps:
        raise ValueError("No timestamps available in SHAP data")

    tables = _build_contribution_table_htmls(merged_df, top_features, args.top_n)
    initial_ts = timestamps[-1]
    if args.focus_timestamp:
        candidate = pd.to_datetime(args.focus_timestamp, utc=True).isoformat()
        if candidate in tables:
            initial_ts = candidate
        else:
            logging.warning("Specified focus timestamp %s not found; using %s", candidate, initial_ts)

    title = f"SHAP Time-Series Report ({shap_path.stem})"
    html = _render_html(pred_fig, contrib_fig, tables, title, timestamps, initial_ts)

    output_path = args.output or shap_path.with_name(f"{shap_path.stem}_timeseries_report.html")
    output_path.write_text(html, encoding="utf-8")
    logging.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
