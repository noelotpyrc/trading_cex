#!/usr/bin/env python3
"""
Backfill LightGBM inference for missing hourly bars using OHLCV from DuckDB.

Supports multiple selection modes for target timestamps:
  - window: explicit --start/--end time range of closed bars
  - last_from_predictions: start from last predicted timestamp + 1h to latest closed
  - ts_list: explicit timestamps via --ts or --ts-file

Ensures sufficient history (30d base + buffer) is loaded from DuckDB to build
lookbacks and features per timestamp, then predicts and persists results.

Example:
  python run/backfill_inference_missing.py \
    --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
    --table ohlcv_btcusdt_1h \
    --dataset "BINANCE_BTCUSDT.P, 60" \
    --model-root "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60" \
    --mode window --start "2025-08-01 00:00:00" --end "2025-08-14 23:00:00" \
    --buffer-hours 6 --at-most 200
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb  # type: ignore
import numpy as np
import pandas as pd

from run.data_loader import load_ohlcv_duckdb, validate_hourly_continuity
from run.lookbacks_builder import build_latest_lookbacks, trim_lookbacks_to_base_window, validate_lookbacks_exact
from run.features_builder import compute_latest_features_from_lookbacks, validate_features_for_model
from run.model_io_lgbm import load_lgbm_model, predict_latest_row
from run.predictions_table import PredictionRow, ensure_table as ensure_predictions_table, insert_predictions
from run.features_table import FeatureRow, ensure_table as ensure_features_table, upsert_feature_rows


DEFAULT_TABLE = "ohlcv_btcusdt_1h"


def _now_floor_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("h").tz_convert(None)


def _is_predictions_table_present(con) -> bool:
    try:
        res = con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'predictions' LIMIT 1").fetchone()
        return bool(res)
    except Exception:
        return False


def _prediction_exists(con, ts: pd.Timestamp) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM predictions WHERE ts = ? LIMIT 1",
            [pd.Timestamp(ts).to_pydatetime()],
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _list_closed_bars_in_window(con, table: str, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    now_floor = _now_floor_utc()
    cutoff = now_floor - pd.Timedelta(hours=1)
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)
    q = f"""
        SELECT timestamp FROM {table}
        WHERE timestamp BETWEEN ? AND ? AND timestamp <= ?
        ORDER BY timestamp
    """
    df = con.execute(q, [start.to_pydatetime(), end.to_pydatetime(), cutoff.to_pydatetime()]).fetch_df()
    return [pd.Timestamp(t) for t in df["timestamp"]] if not df.empty else []


def _last_prediction_ts(con) -> Optional[pd.Timestamp]:
    try:
        row = con.execute("SELECT MAX(ts) FROM predictions").fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


def _parse_ts_list(ts_list: Iterable[str]) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    for s in ts_list:
        t = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(t):
            continue
        out.append(t.tz_convert("UTC").tz_localize(None))
    return sorted(set(out))


def _load_ts_file(path: Path) -> List[pd.Timestamp]:
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return _parse_ts_list(lines)


@dataclass
class BackfillConfig:
    duckdb_path: Path  # OHLCV source DB
    pred_duckdb_path: Optional[Path]  # predictions target DB (defaults to duckdb_path)
    feat_duckdb_path: Optional[Path]  # features target DB (defaults to pred_duckdb_path or duckdb_path)
    table: str
    dataset: str
    model_root: Optional[str]
    model_path: Optional[str]
    feature_key: str
    mode: str
    start: Optional[str]
    end: Optional[str]
    ts: List[str]
    ts_file: Optional[Path]
    buffer_hours: int
    base_hours: int
    at_most: Optional[int]
    write_features: bool
    dry_run: bool
    timeframes: List[str]


def _select_target_timestamps(cfg: BackfillConfig) -> List[pd.Timestamp]:
    """Select closed-hour target timestamps using OHLCV DB and (optionally) predictions DB.

    - OHLCV reads come from cfg.duckdb_path.
    - If cfg.pred_duckdb_path is provided, last_from_predictions start is based on that DB; otherwise falls back to OHLCV DB.
    """
    con_ohlcv = duckdb.connect(str(cfg.duckdb_path))
    try:
        con_ohlcv.execute("SET TimeZone='UTC';")
        now_floor = _now_floor_utc()
        cutoff = now_floor - pd.Timedelta(hours=1)

        if cfg.mode == "ts_list":
            ts_list = _parse_ts_list(cfg.ts)
            if cfg.ts_file:
                ts_list = sorted(set(ts_list) | set(_load_ts_file(cfg.ts_file)))
            if not ts_list:
                return []
            # Keep only timestamps present in OHLCV and <= cutoff
            lo, hi = min(ts_list), max(ts_list)
            in_db = _list_closed_bars_in_window(con_ohlcv, cfg.table, lo, hi)
            ts = sorted(set(ts_list).intersection(set(in_db)))
            return [t for t in ts if t <= cutoff]

        if cfg.mode == "last_from_predictions":
            # Determine last predicted timestamp from predictions DB (or fallback to OHLCV DB)
            pred_db_path = cfg.pred_duckdb_path or cfg.duckdb_path
            try:
                con_pred = duckdb.connect(str(pred_db_path))
                con_pred.execute("SET TimeZone='UTC';")
                last_pred = _last_prediction_ts(con_pred)
            finally:
                try:
                    con_pred.close()
                except Exception:
                    pass

            start_ts = (last_pred + pd.Timedelta(hours=1)) if last_pred is not None else None
            end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
            if start_ts is None:
                # Fallback: backfill a small recent window if no predictions yet
                start_ts = end_ts - pd.Timedelta(hours=48)
            return _list_closed_bars_in_window(con_ohlcv, cfg.table, start_ts, end_ts)

        # Default: window mode
        if not cfg.start and not cfg.end:
            # Default to recent 48h ending at cutoff
            end_ts = cutoff
            start_ts = end_ts - pd.Timedelta(hours=48)
        else:
            start_ts = pd.to_datetime(cfg.start).tz_localize(None) if cfg.start else None
            end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
            if start_ts is None:
                start_ts = end_ts - pd.Timedelta(hours=48)
        return _list_closed_bars_in_window(con_ohlcv, cfg.table, start_ts, end_ts)
    finally:
        con_ohlcv.close()


def backfill(cfg: BackfillConfig) -> int:
    # Determine target timestamps
    targets = _select_target_timestamps(cfg)
    if cfg.at_most is not None and cfg.at_most > 0:
        targets = targets[: cfg.at_most]

    if not targets:
        print("No target timestamps found to backfill (after filtering/limits)")
        return 0

    print(
        f"Backfill plan: dataset={cfg.dataset} bars={len(targets)} "
        f"range=[{targets[0]} .. {targets[-1]}] base={cfg.base_hours}h buffer={cfg.buffer_hours}h"
    )
    if cfg.dry_run:
        for t in targets:
            print("  -", t)
        return 0

    # Load model once
    booster, run_dir = load_lgbm_model(model_root=cfg.model_root, model_path=cfg.model_path)
    model_feature_cols = list(booster.feature_name())
    resolved_model_path: Path
    if cfg.model_path:
        mp = Path(cfg.model_path)
        if mp.exists():
            resolved_model_path = mp if mp.is_file() else mp / "model.txt"
        else:
            resolved_model_path = Path(run_dir) / "model.txt"
    else:
        resolved_model_path = Path(run_dir) / "model.txt"

    # Compute earliest-needed OHLCV and load in one shot
    required_hours = int(cfg.base_hours + max(0, cfg.buffer_hours))
    earliest_needed = targets[0] - pd.Timedelta(hours=required_hours - 1)
    df_all = load_ohlcv_duckdb(
        str(cfg.duckdb_path), table=cfg.table, start=earliest_needed, end=targets[-1]
    )
    if df_all.empty:
        print("[ERROR] No OHLCV rows loaded from DuckDB for required range")
        return 2

    # Prepare target DB connections
    pred_db_path = cfg.pred_duckdb_path or cfg.duckdb_path
    feat_db_path = cfg.feat_duckdb_path or pred_db_path

    con_pred = duckdb.connect(str(pred_db_path))
    con_pred.execute("SET TimeZone='UTC';")
    ensure_predictions_table(con_pred)
    con_feat = None
    if cfg.write_features:
        con_feat = duckdb.connect(str(feat_db_path))
        con_feat.execute("SET TimeZone='UTC';")
        ensure_features_table(con_feat)

    appended = 0
    try:
        for ts in targets:
            # Skip if already predicted (when table present)
            if _is_predictions_table_present(con_pred) and _prediction_exists(con_pred, ts):
                print(f"SKIP {ts}: prediction already exists for {cfg.dataset}")
                continue

            # Ensure we have sufficient history window and continuity
            try:
                validate_hourly_continuity(df_all, end_ts=ts, required_hours=required_hours)
            except Exception as e:
                print(f"SKIP {ts}: insufficient/irregular history window ({e})")
                continue

            # Slice up to ts and build lookbacks
            df_slice = df_all[df_all["timestamp"] <= ts].copy()
            lookbacks = build_latest_lookbacks(df_slice, window_hours=required_hours, timeframes=cfg.timeframes)
            lookbacks = trim_lookbacks_to_base_window(lookbacks, base_hours=cfg.base_hours)
            try:
                validate_lookbacks_exact(lookbacks, base_hours=cfg.base_hours, end_ts=ts)
            except Exception as e:
                print(f"SKIP {ts}: lookbacks invalid ({e})")
                continue

            # Features and predict
            features_row = compute_latest_features_from_lookbacks(lookbacks)
            try:
                validate_features_for_model(booster, features_row)
            except Exception as e:
                print(f"SKIP {ts}: features not aligned for model ({e})")
                continue
            y_pred = predict_latest_row(booster, features_row)

            # Persist prediction and optional features
            row = PredictionRow.from_payload({
                'timestamp': ts,
                'model_path': str(resolved_model_path),
                'y_pred': y_pred,
                'feature_key': cfg.feature_key,
            })
            insert_predictions(con_pred, [row])
            if cfg.write_features and con_feat is not None:
                feature_series = features_row.iloc[0][model_feature_cols]
                feature_row = FeatureRow.from_series(cfg.feature_key, ts, feature_series)
                upsert_feature_rows(con_feat, [feature_row])

            appended += 1
            print(f"OK {ts}: y_pred={y_pred:.6f}")
    finally:
        try:
            con_pred.close()
        finally:
            if con_feat is not None:
                try:
                    con_feat.close()
                except Exception:
                    pass

    print(f"Backfill complete: wrote={appended} bars out of {len(targets)} planned")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> BackfillConfig:
    p = argparse.ArgumentParser(description="Backfill LightGBM predictions for missing bars from DuckDB OHLCV")
    p.add_argument("--duckdb", type=Path, required=True, help="DuckDB database path for OHLCV source")
    p.add_argument("--pred-duckdb", type=Path, default=None, help="DuckDB path for predictions table (default: --duckdb)")
    p.add_argument("--feat-duckdb", type=Path, default=None, help="DuckDB path for features table (default: --pred-duckdb or --duckdb)")
    p.add_argument("--table", type=str, default=DEFAULT_TABLE, help="OHLCV table name in DuckDB")
    p.add_argument("--dataset", type=str, required=True, help="Dataset label stored in predictions")
    p.add_argument("--model-root", default=None, help="Model root containing run_* dirs")
    p.add_argument("--model-path", default=None, help="Explicit run dir or model.txt path")
    p.add_argument("--feature-key", required=True, help="Feature snapshot key associated with these predictions")
    p.add_argument("--mode", choices=["window", "last_from_predictions", "ts_list"], default="window")
    p.add_argument("--start", default=None, help="Start timestamp (inclusive) for window mode")
    p.add_argument("--end", default=None, help="End timestamp (inclusive) for window/last_from_predictions")
    p.add_argument("--ts", nargs="*", default=[], help="Explicit timestamps for ts_list mode")
    p.add_argument("--ts-file", type=Path, default=None, help="File with one timestamp per line for ts_list mode")
    p.add_argument("--buffer-hours", type=int, default=6, help="Extra hours on top of base window")
    p.add_argument("--base-hours", type=int, default=30 * 24, help="Base training window in hours (default 720)")
    p.add_argument("--at-most", type=int, default=None, help="Cap the number of bars to process")
    p.add_argument("--write-features", action="store_true", help="Also store features_latest rows for audit")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not run predictions")
    p.add_argument("--timeframes", nargs="+", default=["1H", "4H", "12H", "1D"], help="Timeframes to use for lookbacks")
    args = p.parse_args(argv)

    feature_key = args.feature_key
    if args.write_features and not feature_key:
        raise SystemExit("--write-features requires --feature-key")

    return BackfillConfig(
        duckdb_path=args.duckdb,
        pred_duckdb_path=args.pred_duckdb,
        feat_duckdb_path=args.feat_duckdb,
        table=str(args.table),
        dataset=str(args.dataset),
        model_root=args.model_root,
        model_path=args.model_path,
        feature_key=feature_key,
        mode=str(args.mode),
        start=args.start,
        end=args.end,
        ts=list(args.ts or []),
        ts_file=args.ts_file,
        buffer_hours=int(args.buffer_hours),
        base_hours=int(args.base_hours),
        at_most=(int(args.at_most) if args.at_most is not None else None),
        write_features=bool(args.write_features),
        dry_run=bool(args.dry_run),
        timeframes=list(args.timeframes or ["1H", "4H", "12H", "1D"]),
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        return backfill(cfg)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
