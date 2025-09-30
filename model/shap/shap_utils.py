"""Shared helpers for SHAP reporting scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass
class FeatureSourceConfig:
    """Configuration describing how to fetch feature values."""

    csv_path: Optional[Path] = None
    timestamp_column: str = "timestamp"
    duckdb_path: Optional[Path] = None
    duckdb_table: str = "features"
    duckdb_feature_key: Optional[str] = None
    duckdb_timestamp_column: str = "ts"
    duckdb_features_column: str = "features"

    def is_csv(self) -> bool:
        return self.csv_path is not None

    def is_duckdb(self) -> bool:
        return self.duckdb_path is not None


def load_shap_dataframe(shap_path: Path) -> pd.DataFrame:
    """Load SHAP values parquet and normalise timestamp column if present."""

    df = pd.read_parquet(shap_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_feature_dataframe(cfg: FeatureSourceConfig) -> pd.DataFrame:
    """Load feature values from either CSV or DuckDB source."""

    if cfg.is_csv() and cfg.is_duckdb():
        raise ValueError("Provide either CSV or DuckDB parameters, not both")
    if not cfg.is_csv() and not cfg.is_duckdb():
        raise ValueError("One of csv_path or duckdb_path must be supplied")

    if cfg.is_csv():
        if not cfg.csv_path:
            raise ValueError("csv_path must be provided for CSV source")
        if not cfg.csv_path.exists():
            raise FileNotFoundError(f"Feature CSV not found: {cfg.csv_path}")
        df = pd.read_csv(cfg.csv_path)
        if cfg.timestamp_column not in df.columns:
            raise KeyError(
                f"Timestamp column '{cfg.timestamp_column}' not present in feature CSV"
            )
        df = df.copy()
        df[cfg.timestamp_column] = pd.to_datetime(df[cfg.timestamp_column], utc=True)
        return df

    # DuckDB source
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guarded by tests
        raise RuntimeError("duckdb must be installed to query feature table") from exc

    if not cfg.duckdb_path:
        raise ValueError("duckdb_path must be provided for DuckDB source")
    if not cfg.duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {cfg.duckdb_path}")
    if not cfg.duckdb_feature_key:
        raise ValueError("duckdb_feature_key must be supplied for DuckDB source")

    query = (
        f"SELECT {cfg.duckdb_timestamp_column} AS timestamp_col, {cfg.duckdb_features_column} "
        f"FROM {cfg.duckdb_table} WHERE feature_key = ? ORDER BY {cfg.duckdb_timestamp_column}"
    )

    with duckdb.connect(str(cfg.duckdb_path)) as con:
        con.execute("SET TimeZone='UTC';")
        df = con.execute(query, [cfg.duckdb_feature_key]).fetch_df()

    if df.empty:
        raise ValueError(
            f"No rows found in DuckDB table '{cfg.duckdb_table}' for feature_key="
            f"{cfg.duckdb_feature_key!r}"
        )

    if cfg.duckdb_features_column not in df.columns:
        raise KeyError(
            f"Column '{cfg.duckdb_features_column}' missing from DuckDB result"
        )

    feature_rows: Sequence[dict[str, float]] = []
    feature_names: Optional[Iterable[str]] = None
    for payload in df[cfg.duckdb_features_column]:
        data = json.loads(payload) if isinstance(payload, str) else payload
        if feature_names is None:
            feature_names = list(data.keys())
        feature_rows.append({k: float(v) for k, v in data.items()})

    features_df = pd.DataFrame(feature_rows)
    features_df.insert(0, cfg.timestamp_column, pd.to_datetime(df["timestamp_col"], utc=True))
    if "timestamp_col" in features_df.columns:
        features_df = features_df.drop(columns=["timestamp_col"])
    return features_df


def ensure_timestamp_alignment(
    shap_df: pd.DataFrame,
    features_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Join SHAP and feature frames on timestamp ensuring 1-1 mapping."""

    if timestamp_column not in shap_df.columns:
        raise KeyError(
            f"SHAP data must include a '{timestamp_column}' column for alignment"
        )
    if timestamp_column not in features_df.columns:
        raise KeyError(
            f"Feature data must include a '{timestamp_column}' column for alignment"
        )

    merged = shap_df.merge(features_df, on=timestamp_column, suffixes=("", "_feat"))
    if merged.empty:
        raise ValueError("No overlapping timestamps between SHAP and feature data")
    return merged
