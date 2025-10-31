#!/usr/bin/env python3
"""
Interactive correlation heatmaps for MLflow-first greedy feature selection.

This script:
- Loads a feature matrix from CSV or DuckDB (feature_key JSON expansion).
- Aggregates top-N features from MLflow runs (by experiment + metric filter).
- Runs the target-free greedy correlation selector (tau, per-family cap).
- Saves Plotly interactive heatmaps (HTML) for candidates and selected sets.

Usage example (RMSE lower-is-better):
  python analysis/heatmap_feature_correlations.py \
    --tracking-uri http://127.0.0.1:5000 \
    --experiment lgbm-btcusdt-1h-202301-202508-logret-7d \
    --metric rmse_test --threshold 0.06 --comparator '<=' \
    --top-n-per-run 100 --min-occurrence 2 --topK-overall 300 \
    --data-source csv \
    --features-csv "/Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_tradingview/feature_store/features.csv" \
    --start-ts 2023-01-01T00:00:00 \
    --end-ts 2025-08-01T00:00:00 \
    --tau 0.90 --cap-per-family 2 --min-overlap 200 \
    --out-dir results/heatmaps --method spearman --absolute \
    --cluster-order --heatmap-max-candidates 400

Notes
- For DuckDB input, this script uses app.dashboard_data.load_feature_frame.
- Correlation matrices may be large (O(p^2)); consider --heatmap-max-candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _add_repo_root_to_path() -> None:
    """Ensure repo root (containing 'utils') is importable."""
    root = Path.cwd()
    for _ in range(6):
        if (root / "utils").exists():
            break
        if root.parent == root:
            break
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_repo_root_to_path()

try:
    from utils.greedy_feature_selector import (
        select_mlflow_first_greedy,
        GreedyParams,
    )
    from utils.mlflow_feature_importance import get_feature_occurrence_series
except Exception as e:  # pragma: no cover
    raise SystemExit(
        f"Failed to import local utils.* modules. Ensure you run from repo or set PYTHONPATH. Error: {e}"
    )


def load_features_from_duckdb(db_path: Path, feature_key: str) -> pd.DataFrame:
    try:
        from app.dashboard_data import load_feature_frame
    except Exception as e:
        raise SystemExit(
            "DuckDB loader unavailable. Install dependencies and ensure app/ is importable."
        ) from e
    return load_feature_frame(db_path, feature_key=feature_key)


def load_feature_matrix(
    *,
    data_source: str,
    features_csv: Optional[Path],
    duckdb_path: Optional[Path],
    feature_key: Optional[str],
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    row_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data_source == "csv":
        if not features_csv:
            raise SystemExit("--features-csv is required for data-source=csv")
        df = pd.read_csv(features_csv)
    elif data_source == "duckdb":
        if not duckdb_path or not feature_key:
            raise SystemExit("--duckdb and --feature-key are required for data-source=duckdb")
        df = load_features_from_duckdb(duckdb_path, feature_key)
    else:
        raise SystemExit("--data-source must be one of: csv, duckdb")

    # Optional time-window filter by 'timestamp'
    if 'timestamp' in df.columns and (start_ts or end_ts):
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        # normalize to naive UTC to match other components
        ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
        df = df.copy()
        df['timestamp'] = ts
        if start_ts:
            try:
                st = pd.Timestamp(start_ts)
                df = df[df['timestamp'] >= st]
            except Exception:
                pass
        if end_ts:
            try:
                et = pd.Timestamp(end_ts)
                df = df[df['timestamp'] <= et]
            except Exception:
                pass

    if row_limit is not None and row_limit > 0:
        df = df.iloc[: int(row_limit)].copy()

    # Build X: drop timestamp and known target columns
    drop_cols = [c for c in df.columns if str(c) == "timestamp" or str(c).startswith("y_")]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    return df, X


def compute_corr(df: pd.DataFrame, method: str = "spearman", absolute: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if method == "spearman":
        r = df.rank(method="average", pct=True).corr(method="pearson")
    else:
        r = df.corr(method=method)
    if absolute:
        r = r.abs()
        # ensure diagonal is exactly 1.0
        for i in range(len(r)):
            r.iat[i, i] = 1.0
    return r


def reorder_corr_by_cluster(corr: pd.DataFrame) -> pd.DataFrame:
    if corr is None or corr.empty:
        return corr
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        D = 1.0 - corr.abs()
        Z = linkage(squareform(D.values, checks=False), method="average")
        order = leaves_list(Z)
        cols = corr.columns[order]
        return corr.loc[cols, cols]
    except Exception:
        return corr


def build_combined_heatmaps(
    corr_list: list[tuple[str, pd.DataFrame]],
    *,
    use_webgl: bool = True,
) -> "go.Figure":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Filter out empty matrices but keep placeholders for titles
    rows = len(corr_list)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=False, shared_yaxes=False,
                        subplot_titles=[name for name, _ in corr_list])

    # Shared coloraxis for consistent scale across subplots
    fig.update_layout(coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1, colorbar=dict(title='rho')))

    # Prefer WebGL if requested and available; otherwise fall back to regular Heatmap
    if use_webgl and hasattr(go, "Heatmapgl"):
        Heat = go.Heatmapgl
    else:
        if use_webgl and not hasattr(go, "Heatmapgl"):
            print("Warning: Plotly Heatmapgl not available in this environment; falling back to Heatmap.")
        Heat = go.Heatmap
    for i, (name, corr) in enumerate(corr_list, start=1):
        if corr is None or corr.empty or corr.shape[0] < 2:
            # Add a text annotation if not enough features
            fig.add_annotation(row=i, col=1, text=f"{name}: not enough features to plot", showarrow=False)
            continue
        x = corr.columns.tolist(); y = corr.index.tolist(); z = corr.values
        fig.add_trace(
            Heat(z=z, x=x, y=y, coloraxis='coloraxis', showscale=(i == rows)),
            row=i, col=1,
        )
        fig.update_xaxes(showgrid=False, tickfont=dict(size=8), row=i, col=1)
        fig.update_yaxes(showgrid=False, tickfont=dict(size=8), row=i, col=1)

    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive correlation heatmaps for MLflow-first greedy features")

    # MLflow filters
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--comparator", choices=[">=", ">", "<=", "<", "=", "=="], default=">=")
    ap.add_argument("--top-n-per-run", type=int, default=100)
    ap.add_argument("--min-occurrence", type=int, default=2)
    ap.add_argument("--topK-overall", type=int, default=300)

    # Feature input
    ap.add_argument("--data-source", choices=["csv", "duckdb"], required=True)
    ap.add_argument("--features-csv", type=Path, default=None)
    ap.add_argument("--duckdb", type=Path, default=None)
    ap.add_argument("--feature-key", type=str, default=None)
    ap.add_argument("--start-ts", type=str, default=None, help="ISO start timestamp to filter rows (by 'timestamp')")
    ap.add_argument("--end-ts", type=str, default=None, help="ISO end timestamp to filter rows (by 'timestamp')")
    ap.add_argument("--row-limit", type=int, default=None)

    # Greedy params
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--cap-per-family", type=int, default=2)
    ap.add_argument("--min-overlap", type=int, default=200)
    ap.add_argument("--known-tfs", nargs="*", default=["1H", "4H", "12H", "1D"])

    # Correlation/plot settings
    ap.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    ap.add_argument("--absolute", action="store_true")
    ap.add_argument("--cluster-order", action="store_true")
    ap.add_argument("--heatmap-max-candidates", type=int, default=None, help="cap candidates heatmap columns")
    ap.add_argument("--use-webgl", action="store_true", help="use Plotly Heatmapgl for better performance")
    ap.add_argument("--include-rest", action="store_true", help="After seeding with candidates, greedily add from all X")
    ap.add_argument("--target-total", type=int, default=None, help="Optional total feature target when including rest")
    ap.add_argument("--nan-penalty", type=float, default=1.0, help="Exponent for (1-NaN_rate) in quality score for rest ordering")
    ap.add_argument("--exclude-suffix", action="append", default=None, help="Suffix to exclude (repeatable). Default excludes _all_tf_normalized")
    ap.add_argument("--no-default-exclude", action="store_true", help="If set, do not exclude _all_tf_normalized by default")
    ap.add_argument("--restrict-features-json", type=Path, default=None, help="JSON list file; only consider features from this list")

    # Outputs
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (used if --out-html not set)")
    ap.add_argument("--out-html", type=Path, default=None, help="Single HTML file to write both heatmaps")
    ap.add_argument("--save-selected", type=Path, default=None, help="optional path to save selected features JSON")

    args = ap.parse_args()

    # Load features
    raw, X = load_feature_matrix(
        data_source=args.data_source,
        features_csv=args.features_csv,
        duckdb_path=args.duckdb,
        feature_key=args.feature_key,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        row_limit=args.row_limit,
    )
    print(f"Loaded features: raw={raw.shape} X={X.shape}")

    # MLflow counts and candidates
    counts = get_feature_occurrence_series(
        experiment_name=args.experiment,
        metric_name=args.metric,
        metric_threshold=args.threshold,
        top_n=args.top_n_per_run,
        comparator=args.comparator,
        tracking_uri=args.tracking_uri,
    )
    print(f"Unique features from MLflow runs: {0 if counts is None else len(counts)}")

    filtered = counts[counts >= int(args.min_occurrence)] if counts is not None else pd.Series(dtype=int)
    if args.topK_overall is not None and len(filtered) > int(args.topK_overall):
        filtered = filtered.head(int(args.topK_overall))
    candidates = [c for c in filtered.index if c in X.columns]
    print(f"Candidates after occurrence/topK & in X: {len(candidates)}")

    # Run greedy selection
    # Build exclusion suffix list
    exclude_suffixes = args.exclude_suffix or []
    if not args.no_default_exclude:
        if "_all_tf_normalized" not in exclude_suffixes:
            exclude_suffixes.append("_all_tf_normalized")

    # Restrict feature set if provided
    restrict_list = None
    if args.restrict_features_json:
        try:
            restrict_list = json.loads(Path(args.restrict_features_json).read_text())
        except Exception as e:
            print(f"Warning: failed to read restrict-features JSON: {e}")

    res = select_mlflow_first_greedy(
        X,
        experiment_name=args.experiment,
        metric_name=args.metric,
        metric_threshold=args.threshold,
        top_n_per_run=args.top_n_per_run,
        comparator=args.comparator,
        tracking_uri=args.tracking_uri,
        min_occurrence=args.min_occurrence,
        topK_overall=args.topK_overall,
        params=GreedyParams(
            tau=args.tau,
            cap_per_family=args.cap_per_family,
            min_overlap=args.min_overlap,
            known_tfs=args.known_tfs,
        ),
        order_known_tfs=args.known_tfs,
        include_rest=bool(args.include_rest),
        target_total=args.target_total,
        nan_penalty=args.nan_penalty,
        exclude_suffixes=exclude_suffixes,
        restrict_features=restrict_list,
    )
    ordered = [c for c in (res.get("ordered") or []) if c in X.columns]
    selected = [c for c in (res.get("selected") or []) if c in X.columns]
    print(f"Ordered candidates: {len(ordered)} | Selected: {len(selected)}")

    if args.save_selected:
        args.save_selected.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_selected, "w") as f:
            json.dump(selected, f, indent=2)
        print(f"Saved selected features to: {args.save_selected}")

    # Build subsets for heatmaps
    cand_cols = ordered
    if args.heatmap_max_candidates and len(cand_cols) > int(args.heatmap_max_candidates):
        cand_cols = cand_cols[: int(args.heatmap_max_candidates)]
    sel_cols = selected

    # Original panel: if a restrict list is provided, show its features; otherwise show candidates
    if restrict_list:
        orig_cols = [c for c in restrict_list if c in X.columns]
    else:
        orig_cols = cand_cols

    X_orig = X[orig_cols].copy() if orig_cols else pd.DataFrame()
    X_sel = X[sel_cols].copy() if sel_cols else pd.DataFrame()

    # Compute correlations
    C_orig = compute_corr(X_orig, method=args.method, absolute=args.absolute) if not X_orig.empty else pd.DataFrame()
    C_sel = compute_corr(X_sel, method=args.method, absolute=args.absolute) if not X_sel.empty else pd.DataFrame()

    if args.cluster_order:
        C_orig = reorder_corr_by_cluster(C_orig)
        C_sel = reorder_corr_by_cluster(C_sel)

    # Build combined figure (two subplots)
    title_suffix = f"{args.method}{' |abs|' if args.absolute else ''}"
    first_title = f"Original ({'restrict list' if restrict_list else 'candidates'}) ({title_suffix})"
    corr_list = [
        (first_title, C_orig),
        (f"Selected ({title_suffix})", C_sel),
    ]
    fig = build_combined_heatmaps(corr_list, use_webgl=args.use_webgl)

    # Determine single output path
    out_html: Optional[Path] = args.out_html
    if out_html is None:
        if args.out_dir is None:
            raise SystemExit("Provide --out-html or --out-dir for output.")
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_html = out_dir / "correlation_heatmaps.html"

    out_html.parent.mkdir(parents=True, exist_ok=True)
    # Use CDN to minimize file size
    import plotly.io as pio
    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote combined heatmaps: {out_html}")


if __name__ == "__main__":  # pragma: no cover
    main()
