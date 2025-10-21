#!/usr/bin/env python3
"""
Backfill LightGBM predictions by scoring precomputed features from a DuckDB
feature store (no on-the-fly feature generation).

Stops early if the feature source does not contain all features required by the
model. Persists predictions (with dataset/feature_key/model_path) and can skip
or overwrite existing rows.

Example:
  python run/backfill_inference_from_feature_store.py \
    --feat-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb" \
    --pred-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb" \
    --feature-key "prod_default" \
    --dataset "BINANCE_BTCUSDT.P, 60" \
    --model-path "/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h" \
    --mode window --start "2025-09-14 23:00:00"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import duckdb  # type: ignore
import pandas as pd

from run.model_io_lgbm import load_lgbm_model
from run.predictions_table import PredictionRow, ensure_table as ensure_predictions_table, insert_predictions


def _now_floor_utc() -> pd.Timestamp:
    return pd.Timestamp.utcnow().floor('h').tz_localize(None)


def _is_predictions_table_present(con) -> bool:
    try:
        res = con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'predictions' LIMIT 1").fetchone()
        return bool(res)
    except Exception:
        return False


def _prediction_exists(con, ts: pd.Timestamp, *, model_path: str, feature_key: str) -> bool:
    try:
        q = "SELECT 1 FROM predictions WHERE ts = ? AND model_path = ? AND feature_key = ? LIMIT 1"
        row = con.execute(q, [pd.Timestamp(ts).to_pydatetime(), model_path, feature_key]).fetchone()
        return row is not None
    except Exception:
        return False


def _list_feature_ts(con_feat, feature_key: str, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    q = """
        SELECT ts FROM features
        WHERE feature_key = ? AND ts BETWEEN ? AND ?
        ORDER BY ts
    """
    df = con_feat.execute(q, [feature_key, start.to_pydatetime(), end.to_pydatetime()]).fetch_df()
    return [pd.Timestamp(t) for t in df['ts']] if not df.empty else []


def _last_prediction_ts(con_pred, *, model_path: str, feature_key: str) -> Optional[pd.Timestamp]:
    try:
        row = con_pred.execute(
            "SELECT MAX(ts) FROM predictions WHERE model_path = ? AND feature_key = ?",
            [model_path, feature_key],
        ).fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


def _json_map_from_row(val) -> dict:
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except Exception:
        return {}


def _fetch_feature_map(con_feat, feature_key: str, ts: pd.Timestamp) -> dict | None:
    row = con_feat.execute(
        "SELECT features FROM features WHERE feature_key = ? AND ts = ? LIMIT 1",
        [feature_key, pd.Timestamp(ts).to_pydatetime()],
    ).fetchone()
    if not row:
        return None
    return _json_map_from_row(row[0])


@dataclass
class InferenceConfig:
    feat_db: Path
    pred_db: Path
    feature_key: str
    dataset: str
    model_root: Optional[str]
    model_path: Optional[str]
    mode: str
    start: Optional[str]
    end: Optional[str]
    ts: List[str]
    ts_file: Optional[Path]
    at_most: Optional[int]
    overwrite: bool
    dry_run: bool


def _select_ts(cfg: InferenceConfig, con_feat, con_pred, *, model_path: str) -> List[pd.Timestamp]:
    now_floor = _now_floor_utc()
    cutoff = now_floor - pd.Timedelta(hours=1)
    if cfg.mode == 'ts_list':
        raw = [pd.to_datetime(t, utc=True).tz_convert('UTC').tz_localize(None) for t in cfg.ts]
        ts = [t for t in raw if t <= cutoff]
        return sorted(ts)
    if cfg.mode == 'last_from_predictions':
        last = _last_prediction_ts(con_pred, model_path=model_path, feature_key=cfg.feature_key)
        start_ts = (last + pd.Timedelta(hours=1)) if last is not None else None
        end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
        if start_ts is None:
            start_ts = end_ts - pd.Timedelta(hours=48)
        return _list_feature_ts(con_feat, cfg.feature_key, start_ts, end_ts)
    # window mode
    start_ts = pd.to_datetime(cfg.start).tz_localize(None) if cfg.start else None
    end_ts = pd.to_datetime(cfg.end).tz_localize(None) if cfg.end else cutoff
    if start_ts is None:
        start_ts = end_ts - pd.Timedelta(hours=48)
    return _list_feature_ts(con_feat, cfg.feature_key, start_ts, end_ts)


def _preflight_or_die(con_feat, feature_key: str, model_feature_cols: List[str]) -> None:
    # Check one row's keys; if missing any model features, fail fast
    row = con_feat.execute(
        "SELECT features FROM features WHERE feature_key = ? ORDER BY ts DESC LIMIT 1",
        [feature_key],
    ).fetchone()
    if not row:
        raise SystemExit(f"No features found for feature_key={feature_key}")
    keys = set(_json_map_from_row(row[0]).keys())
    missing = [c for c in model_feature_cols if c not in keys]
    if missing:
        raise SystemExit(
            f"Feature source missing {len(missing)} required model features; sample: {missing[:20]}"
        )


def run_inference(cfg: InferenceConfig) -> int:
    # Connect DBs
    con_feat = duckdb.connect(str(cfg.feat_db))
    con_feat.execute("SET TimeZone='UTC';")
    con_pred = duckdb.connect(str(cfg.pred_db))
    con_pred.execute("SET TimeZone='UTC';")
    ensure_predictions_table(con_pred)

    # Load model
    booster, run_dir = load_lgbm_model(model_root=cfg.model_root, model_path=cfg.model_path)
    model_feature_cols = list(booster.feature_name())
    resolved_model_path: Path
    if cfg.model_path:
        mp = Path(cfg.model_path)
        resolved_model_path = mp if (mp.exists() and mp.is_file()) else mp / 'model.txt'
    else:
        resolved_model_path = Path(run_dir) / 'model.txt'

    # Preflight: ensure feature store covers all model features
    _preflight_or_die(con_feat, cfg.feature_key, model_feature_cols)

    # Select timestamps (use resolved model path for last_from_predictions)
    targets = _select_ts(cfg, con_feat, con_pred, model_path=str(resolved_model_path))
    if cfg.at_most and cfg.at_most > 0:
        targets = targets[: cfg.at_most]
    if not targets:
        print("No target timestamps found to backfill inference")
        return 0

    print(f"Inference plan: dataset={cfg.dataset} bars={len(targets)} range=[{targets[0]} .. {targets[-1]}]")
    if cfg.dry_run:
        for t in targets:
            print("  -", t)
        return 0

    appended = 0
    try:
        for ts in targets:
            # Skip or delete existing depending on overwrite flag
            exists = _is_predictions_table_present(con_pred) and _prediction_exists(
                con_pred, ts, model_path=str(resolved_model_path), feature_key=cfg.feature_key
            )
            if exists and not cfg.overwrite:
                print(f"SKIP {ts}: prediction already exists for model_path and feature_key")
                continue
            if exists and cfg.overwrite:
                con_pred.execute(
                    "DELETE FROM predictions WHERE ts = ? AND model_path = ? AND feature_key = ?",
                    [pd.Timestamp(ts).to_pydatetime(), str(resolved_model_path), cfg.feature_key],
                )

            fmap = _fetch_feature_map(con_feat, cfg.feature_key, ts)
            if fmap is None:
                print(f"SKIP {ts}: no features found for key={cfg.feature_key}")
                continue
            # Align order and check for missing
            missing = [c for c in model_feature_cols if c not in fmap]
            if missing:
                raise SystemExit(f"Missing model features at {ts}: {missing[:20]}")
            values = [float(fmap.get(c, float('nan'))) for c in model_feature_cols]
            import numpy as np
            pred = float(booster.predict(pd.DataFrame([values], columns=model_feature_cols))[0])

            row = PredictionRow.from_payload({
                'timestamp': ts,
                'model_path': str(resolved_model_path),
                'feature_key': cfg.feature_key,
                'y_pred': pred,
            })
            insert_predictions(con_pred, [row])
            appended += 1
            print(f"OK {ts}: y_pred={pred:.6f}")
    finally:
        try:
            con_feat.close()
        finally:
            con_pred.close()

    print(f"Inference complete: wrote={appended} bars out of {len(targets)} planned")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> InferenceConfig:
    p = argparse.ArgumentParser(description="Backfill LightGBM predictions from a DuckDB feature store")
    p.add_argument("--feat-duckdb", type=Path, required=True, help="DuckDB path containing the features table")
    p.add_argument("--pred-duckdb", type=Path, required=True, help="DuckDB path for predictions table")
    p.add_argument("--feature-key", required=True, help="Feature key to read from the features table")
    p.add_argument("--dataset", required=True, help="Dataset label stored in predictions")
    p.add_argument("--model-root", default=None, help="Model root containing run_* dirs")
    p.add_argument("--model-path", default=None, help="Explicit run dir or model.txt path")
    p.add_argument("--mode", choices=["window", "last_from_predictions", "ts_list"], default="window")
    p.add_argument("--start", default=None, help="Start timestamp (inclusive) for window mode")
    p.add_argument("--end", default=None, help="End timestamp (inclusive) for window/last_from_predictions")
    p.add_argument("--ts", nargs="*", default=[], help="Explicit timestamps for ts_list mode")
    p.add_argument("--ts-file", type=Path, default=None, help="File with one timestamp per line for ts_list mode")
    p.add_argument("--at-most", type=int, default=None, help="Cap number of bars to process")
    p.add_argument("--overwrite", action="store_true", help="Overwrite predictions if they already exist")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not run predictions")
    args = p.parse_args(argv)

    ts_list = [*(args.ts or [])]
    if args.ts_file and Path(args.ts_file).exists():
        ts_list += [ln.strip() for ln in Path(args.ts_file).read_text().splitlines() if ln.strip()]

    return InferenceConfig(
        feat_db=args.feat_duckdb,
        pred_db=args.pred_duckdb,
        feature_key=str(args.feature_key),
        dataset=str(args.dataset),
        model_root=args.model_root,
        model_path=args.model_path,
        mode=str(args.mode),
        start=args.start,
        end=args.end,
        ts=ts_list,
        ts_file=args.ts_file,
        at_most=(int(args.at_most) if args.at_most is not None else None),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        return run_inference(cfg)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
