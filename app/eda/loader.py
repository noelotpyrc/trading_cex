from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


@dataclass(frozen=True)
class EDAConfig:
    dataset_id: str
    ohlcv_duckdb_path: Path
    ohlcv_table: str
    features_duckdb_path: Path
    features_table: str
    default_feature_key: Optional[str] = None
    date_range_days: int = 90
    profile_minimal: bool = False
    report_dir: Path = Path("reports")
    # Feature store (CSV) support
    feature_store_folder: Optional[Path] = None
    fs_features_csv: str = "features.csv"
    fs_targets_csv: str = "targets.csv"
    fs_ohlcv_csv: str = "ohlcv.csv"


def _read_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyyaml is required to load EDA config: pip install pyyaml") from e

    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path) -> EDAConfig:
    data = _read_yaml(config_path) if config_path.exists() else {}

    dataset_id = data.get("dataset_id", "binance_btcusdt_perp_1h")

    ohlcv = data.get("ohlcv", {})
    features = data.get("features", {})
    defaults = data.get("defaults", {})

    fs = data.get("feature_store", {})

    return EDAConfig(
        dataset_id=dataset_id,
        ohlcv_duckdb_path=Path(ohlcv.get("duckdb_path", "/path/to/ohlcv.duckdb")),
        ohlcv_table=str(ohlcv.get("table", "ohlcv_btcusdt_1h")),
        features_duckdb_path=Path(features.get("duckdb_path", "/path/to/features.duckdb")),
        features_table=str(features.get("table", "features")),
        default_feature_key=features.get("default_feature_key"),
        date_range_days=int(defaults.get("date_range_days", 90)),
        profile_minimal=bool(defaults.get("profile_minimal", False)),
        report_dir=Path(defaults.get("report_dir", "reports")),
        feature_store_folder=Path(fs.get("folder")) if fs.get("folder") else None,
        fs_features_csv=str(fs.get("features_csv", "features.csv")),
        fs_targets_csv=str(fs.get("targets_csv", "targets.csv")),
        fs_ohlcv_csv=str(fs.get("ohlcv_csv", "ohlcv.csv")),
    )


def _normalize_timestamp_col(df: pd.DataFrame, *, ts_col: str) -> pd.DataFrame:
    data = df.copy()
    ts = pd.to_datetime(data[ts_col], errors="coerce", utc=True)
    data["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    if "timestamp" != ts_col and "timestamp" in data.columns:
        data = data.drop(columns=[ts_col])
    data = data.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return data


def load_ohlcv_duckdb(
    db_path: os.PathLike | str,
    table: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    # Reuse the repo utility if available
    try:
        from run.data_loader import load_ohlcv_duckdb as _load
        return _load(db_path, table=table, start=start, end=end, limit=limit)
    except Exception:
        # Fallback minimal loader with duckdb if import fails
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path))
        try:
            con.execute("SET TimeZone='UTC';")
            clauses = []
            params: list[object] = []
            if start is not None:
                s = pd.to_datetime(start, utc=True).tz_convert('UTC').tz_localize(None)
                clauses.append("timestamp >= ?")
                params.append(pd.Timestamp(s).to_pydatetime())
            if end is not None:
                e = pd.to_datetime(end, utc=True).tz_convert('UTC').tz_localize(None)
                clauses.append("timestamp <= ?")
                params.append(pd.Timestamp(e).to_pydatetime())
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            q = f"SELECT timestamp, open, high, low, close, volume FROM {table}{where} ORDER BY timestamp ASC"
            if limit is not None and int(limit) > 0:
                q = f"{q} LIMIT {int(limit)}"
            df = con.execute(q, params).fetch_df()
        finally:
            con.close()

        # Normalize
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = _normalize_timestamp_col(df, ts_col="timestamp")
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[[c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]]


def load_features_json_wide(
    db_path: os.PathLike | str,
    table: str,
    feature_key: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load features from DuckDB and flatten the JSON map to wide columns.

    Returns DataFrame with columns: timestamp, <feature columns...>
    """
    import duckdb  # type: ignore

    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET TimeZone='UTC';")
        clauses = ["feature_key = ?"]
        params: list[object] = [feature_key]
        if start is not None:
            s = pd.to_datetime(start, utc=True).tz_convert('UTC').tz_localize(None)
            clauses.append("ts >= ?")
            params.append(pd.Timestamp(s).to_pydatetime())
        if end is not None:
            e = pd.to_datetime(end, utc=True).tz_convert('UTC').tz_localize(None)
            clauses.append("ts <= ?")
            params.append(pd.Timestamp(e).to_pydatetime())
        where = " WHERE " + " AND ".join(clauses)
        q = f"SELECT ts, features FROM {table}{where} ORDER BY ts ASC"
        df = con.execute(q, params).fetch_df()
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame(columns=["timestamp"])  # empty sentinel

    df = _normalize_timestamp_col(df.rename(columns={"ts": "timestamp"}), ts_col="timestamp")

    # Flatten JSON
    feat_series = df["features"].apply(lambda x: json.loads(x) if isinstance(x, str) else (x or {}))
    feat_wide = pd.json_normalize(feat_series)
    feat_wide.columns = [str(c) for c in feat_wide.columns]
    out = pd.concat([df[["timestamp"]].reset_index(drop=True), feat_wide.reset_index(drop=True)], axis=1)
    # Deduplicate columns if any
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def align_join_ohlcv_features(
    ohlcv: pd.DataFrame,
    features_wide: pd.DataFrame,
) -> pd.DataFrame:
    if "timestamp" not in ohlcv.columns or "timestamp" not in features_wide.columns:
        raise ValueError("Both inputs must include 'timestamp' column")
    # Normalize types
    o = ohlcv.copy()
    f = features_wide.copy()
    o["timestamp"] = pd.to_datetime(o["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    joined = pd.merge(o, f, on="timestamp", how="inner")
    joined = joined.sort_values("timestamp").reset_index(drop=True)
    return joined


def available_time_range(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df.empty or "timestamp" not in df.columns:
        return None, None
    return pd.Timestamp(df["timestamp"].min()), pd.Timestamp(df["timestamp"].max())


def ohlcv_min_max(
    db_path: os.PathLike | str,
    table: str,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Fetch min/max timestamp directly from DuckDB without loading all rows."""
    try:
        import duckdb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("duckdb is required to query OHLCV range: pip install duckdb") from e

    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET TimeZone='UTC';")
        q = f"SELECT MIN(timestamp) AS t_min, MAX(timestamp) AS t_max FROM {table}"
        df = con.execute(q).fetch_df()
    finally:
        con.close()

    if df.empty or pd.isna(df.loc[0, "t_min"]) or pd.isna(df.loc[0, "t_max"]):
        return None, None
    t_min = pd.to_datetime(df.loc[0, "t_min"], utc=True).tz_convert("UTC").tz_localize(None)
    t_max = pd.to_datetime(df.loc[0, "t_max"], utc=True).tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(t_min), pd.Timestamp(t_max)


# ===== CSV Feature Store Loaders (pure pandas) =====

def _detect_ts_col(cols: List[str]) -> str:
    norm = [c.strip().lower() for c in cols]
    if "timestamp" in norm:
        return cols[norm.index("timestamp")]
    if "time" in norm:
        return cols[norm.index("time")]
    return cols[0]


def _csv_time_range(csv_path: os.PathLike | str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    path = Path(csv_path)
    if not path.exists():
        return None, None
    # Read header to detect timestamp column
    header = pd.read_csv(path, nrows=0)
    if header.shape[1] == 0:
        return None, None
    ts_col = _detect_ts_col(list(header.columns))
    ts_only = pd.read_csv(path, usecols=[ts_col], parse_dates=[ts_col])
    ts_only = ts_only.dropna()
    if ts_only.empty:
        return None, None
    t_min = pd.to_datetime(ts_only[ts_col], utc=True).dt.tz_convert("UTC").dt.tz_localize(None).min()
    t_max = pd.to_datetime(ts_only[ts_col], utc=True).dt.tz_convert("UTC").dt.tz_localize(None).max()
    return pd.Timestamp(t_min), pd.Timestamp(t_max)


def _csv_load_range(csv_path: os.PathLike | str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    path = Path(csv_path)
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    ts_col = _detect_ts_col(cols)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=200_000, parse_dates=[ts_col]):
        # Normalize ts and filter window
        ts = pd.to_datetime(chunk[ts_col], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        chunk = chunk.assign(timestamp=ts).drop(columns=[c for c in [ts_col] if c != "timestamp"])  # rename to timestamp
        if start is not None:
            chunk = chunk[chunk["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            chunk = chunk[chunk["timestamp"] <= pd.Timestamp(end)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=["timestamp"])  # empty
    df = pd.concat(chunks, axis=0, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def fs_min_max(folder: os.PathLike | str, ohlcv_csv: str = "ohlcv.csv") -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    path = Path(folder) / ohlcv_csv
    if not path.exists():
        return None, None
    return _csv_time_range(path)


def fs_load_ohlcv(folder: os.PathLike | str, ohlcv_csv: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    path = Path(folder) / ohlcv_csv
    if not path.exists():
        return pd.DataFrame(columns=["timestamp"])
    df = _csv_load_range(path, start, end)
    # Standardize OHLCV column naming
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for src, dst in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume")]:
        if src in cols_lower:
            rename_map[cols_lower[src]] = dst
        elif src.capitalize() in df.columns:
            rename_map[src.capitalize()] = dst
    df = df.rename(columns=rename_map)
    # Keep standard columns order if present
    ordered = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[ordered + [c for c in df.columns if c not in ordered]]


def fs_load_features(folder: os.PathLike | str, features_csv: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    path = Path(folder) / features_csv
    if not path.exists():
        return pd.DataFrame(columns=["timestamp"])
    df = _csv_load_range(path, start, end)
    return df


def fs_load_targets(folder: os.PathLike | str, targets_csv: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    path = Path(folder) / targets_csv
    if not path.exists():
        return pd.DataFrame(columns=["timestamp"])
    df = _csv_load_range(path, start, end)
    return df


def join_on_timestamp(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Inner-join multiple DataFrames on 'timestamp' sorted ascending."""
    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        return pd.DataFrame(columns=["timestamp"])  # empty
    out = non_empty[0].copy()
    for d in non_empty[1:]:
        out = pd.merge(out, d, on="timestamp", how="inner")
    return out.sort_values("timestamp").reset_index(drop=True)


def fs_detect_binary_targets(
    folder: os.PathLike | str,
    targets_csv: str,
) -> List[str]:
    """Detect binary target columns by reading the full CSV once (simple and robust).

    A column is considered binary if, across the entire file (non-null values):
      - It is boolean dtype, OR
      - It can be parsed as numeric and values are a subset of {0,1}, OR
      - When treated as strings (lower/stripped), values are a subset of
        {'0','1','true','false','yes','no','t','f','y','n'} (null-like removed).
    """
    path = Path(folder) / targets_csv
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if df.shape[1] == 0:
        return []
    ts_col = _detect_ts_col(list(df.columns))
    candidate_cols = [c for c in df.columns if c != ts_col]
    if not candidate_cols:
        return []

    out: List[str] = []
    null_like = {"", "nan", "none", "null"}
    allowed_tokens = {"0", "1", "true", "false", "yes", "no", "t", "f", "y", "n"}

    for c in candidate_cols:
        col = df[c]
        # Skip columns with no non-null anywhere
        if col.dropna().shape[0] == 0:
            continue
        # Boolean dtype
        if col.dtype == bool:
            out.append(c)
            continue
        # Numeric parse
        s_num = pd.to_numeric(col, errors="coerce").dropna()
        if s_num.shape[0] > 0 and set(pd.unique(s_num)).issubset({0, 1, 0.0, 1.0}):
            out.append(c)
            continue
        # String tokens
        s_str = col.astype(str).str.strip().str.lower()
        s_str = s_str[~s_str.isin(null_like)]
        if s_str.shape[0] > 0 and set(pd.unique(s_str)).issubset(allowed_tokens):
            out.append(c)

    return out
