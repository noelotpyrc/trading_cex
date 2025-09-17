#!/usr/bin/env python3
"""Utilities for persisting feature snapshots into DuckDB."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import duckdb  # type: ignore
import pandas as pd

TABLE_NAME = "features"
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    feature_key VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    features JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(feature_key, ts)
)
"""


@contextmanager
def connect(db_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET TimeZone='UTC';")
        yield con
    finally:
        con.close()


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(CREATE_TABLE_SQL)


@dataclass
class FeatureRow:
    feature_key: str
    ts: pd.Timestamp
    features: Mapping[str, float]

    @classmethod
    def from_dataframe(
        cls,
        feature_key: str,
        features_df: pd.DataFrame,
        *,
        feature_columns: Iterable[str] | None = None,
    ) -> "FeatureRow":
        if 'timestamp' not in features_df.columns:
            raise ValueError("features_df must include a 'timestamp' column")
        if len(features_df) != 1:
            raise ValueError("features_df must contain exactly one row")

        ts_raw = pd.Timestamp(features_df.iloc[0]['timestamp'])
        ts = ts_raw.tz_convert(None) if ts_raw.tzinfo is not None else ts_raw.tz_localize(None)

        if feature_columns is None:
            feature_columns = [c for c in features_df.columns if c != 'timestamp']
        missing = [c for c in feature_columns if c not in features_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:20]}")

        row = features_df.iloc[0]
        features_dict = {}
        for col in feature_columns:
            val = row[col]
            try:
                features_dict[str(col)] = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Feature '{col}' cannot be converted to float: {val}") from None
        return cls(feature_key=feature_key, ts=ts, features=features_dict)

    @classmethod
    def from_series(
        cls,
        feature_key: str,
        ts: pd.Timestamp,
        features_series: pd.Series,
    ) -> "FeatureRow":
        ts_clean = ts.tz_convert(None) if ts.tzinfo is not None else ts.tz_localize(None)
        features_dict = {}
        for col, val in features_series.items():
            try:
                features_dict[str(col)] = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Feature '{col}' cannot be converted to float: {val}") from None
        return cls(feature_key=feature_key, ts=ts_clean, features=features_dict)


def upsert_feature_rows(
    con: duckdb.DuckDBPyConnection,
    rows: Iterable[FeatureRow],
) -> int:
    ensure_table(con)
    data = [
        (row.feature_key, row.ts.to_pydatetime(), json.dumps(row.features, sort_keys=True))
        for row in rows
    ]
    if not data:
        return 0
    con.executemany(
        f"INSERT OR REPLACE INTO {TABLE_NAME} (feature_key, ts, features) VALUES (?, ?, ?)",
        data,
    )
    return len(data)


def fetch_recent(con: duckdb.DuckDBPyConnection, feature_key: str, limit: int) -> pd.DataFrame:
    ensure_table(con)
    query = f"""
        SELECT feature_key, ts, features, created_at
        FROM {TABLE_NAME}
        WHERE feature_key = ?
        ORDER BY ts DESC
        LIMIT ?
    """
    return con.execute(query, [feature_key, limit]).fetch_df()


def describe_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    ensure_table(con)
    return con.execute(f"DESCRIBE {TABLE_NAME}").fetch_df()
