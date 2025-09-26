#!/usr/bin/env python3
"""Compute SHAP explanations for trained LightGBM runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from model.lgbm_inference import align_features_for_booster, load_booster


logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _load_run_paths(run_dir: Path) -> dict:
    paths_file = run_dir / "paths.json"
    if not paths_file.exists():
        raise FileNotFoundError(f"paths.json not found under {run_dir}")
    return json.loads(paths_file.read_text())


def _load_split_frame(
    run_dir: Path,
    booster: lgb.Booster,
    split: str,
    prepared_dir: Optional[Path],
    max_rows: Optional[int],
    random_seed: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    paths_info = _load_run_paths(run_dir)
    source_dir = prepared_dir or Path(paths_info.get("prepared_data_dir", ""))
    if not source_dir or not source_dir.exists():
        raise FileNotFoundError(
            f"Prepared data directory not found for split '{split}'. Provide --prepared-dir explicitly."
        )

    x_path = source_dir / f"X_{split}.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"Feature split CSV missing: {x_path}")

    features = pd.read_csv(x_path)
    logger.info("Loaded %d rows from %s", len(features), x_path)

    if max_rows and len(features) > max_rows:
        features = features.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)
        logger.info("Sampled down to %d rows for SHAP computation", len(features))

    aligned = align_features_for_booster(features, booster, drop_extra=True)

    meta_df = None
    pred_path = run_dir / f"pred_{split}.csv"
    if pred_path.exists():
        meta = pd.read_csv(pred_path)
        if len(meta) == len(aligned):
            meta_df = meta[['timestamp', 'y_true', 'y_pred']] if {'timestamp', 'y_true', 'y_pred'}.issubset(meta.columns) else meta
        else:
            logger.warning(
                "Prediction file %s has %d rows, which does not match features (%d). Ignoring metadata.",
                pred_path,
                len(meta),
                len(aligned),
            )
    return aligned, meta_df


def _load_inference_frame(
    run_dir: Path,
    booster: lgb.Booster,
    duckdb_path: Path,
    feature_key: Optional[str],
    limit: Optional[int],
    start_ts: Optional[str],
    end_ts: Optional[str],
    max_rows: Optional[int],
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("duckdb is required for inference mode") from exc

    query = [
        "SELECT ts, feature_key, features FROM features WHERE 1=1",
    ]
    params: list = []
    if feature_key:
        query.append("AND feature_key = ?")
        params.append(feature_key)
    if start_ts:
        query.append("AND ts >= ?")
        params.append(start_ts)
    if end_ts:
        query.append("AND ts <= ?")
        params.append(end_ts)
    query.append("ORDER BY ts ASC")
    if limit:
        query.append("LIMIT ?")
        params.append(limit)

    sql = " ".join(query)
    logger.info("Fetching features from DuckDB: %s", sql)
    with duckdb.connect(str(duckdb_path)) as con:
        con.execute("SET TimeZone='UTC';")
        df = con.execute(sql, params).fetch_df()

    if df.empty:
        raise ValueError("No inference feature rows retrieved for the specified parameters")

    logger.info(
        "Retrieved %d rows%s",
        len(df),
        f" for feature_key={feature_key}" if feature_key else "",
    )

    feature_names = list(booster.feature_name())
    feature_matrix = []
    for idx, payload in enumerate(df["features"].tolist()):
        data = json.loads(payload) if isinstance(payload, str) else payload
        row = []
        for name in feature_names:
            if name not in data:
                raise KeyError(f"Feature '{name}' missing in inference row {idx}")
            row.append(float(data[name]))
        feature_matrix.append(row)

    features_raw = pd.DataFrame(feature_matrix, columns=feature_names)

    if max_rows and len(features_raw) > max_rows:
        features_raw, df = _sample_with_metadata(features_raw, df, max_rows, random_seed)

    aligned = align_features_for_booster(features_raw, booster, drop_extra=True)
    meta_dict = {"timestamp": pd.to_datetime(df["ts"], utc=True)}
    if "feature_key" in df.columns:
        meta_dict["feature_key"] = df["feature_key"].astype(str)
    meta = pd.DataFrame(meta_dict)
    feature_values = aligned.copy()
    feature_values.insert(0, "timestamp", meta_dict["timestamp"].to_numpy())
    if "feature_key" in meta_dict:
        feature_values.insert(1, "feature_key", meta_dict["feature_key"].to_numpy())
    return aligned, meta, feature_values


def _sample_with_metadata(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    max_rows: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    indices = features.sample(n=max_rows, random_state=random_seed).index
    return features.loc[indices].reset_index(drop=True), meta.loc[indices].reset_index(drop=True)


def _select_component(values, expected_value, class_index: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(values, list):
        if class_index is None:
            class_index = 1 if len(values) > 1 else 0
        if class_index >= len(values):
            raise IndexError(f"class_index {class_index} out of bounds for shap_values with {len(values)} classes")
        shap_array = np.asarray(values[class_index])
        base = np.asarray(expected_value[class_index]) if isinstance(expected_value, (list, tuple, np.ndarray)) else np.asarray(expected_value)
    else:
        shap_array = np.asarray(values)
        base = np.asarray(expected_value)
    return shap_array, base


def _expand_base_values(base: np.ndarray, n_rows: int) -> np.ndarray:
    if base.ndim == 0:
        return np.full(n_rows, base, dtype=float)
    if base.ndim == 1 and len(base) == n_rows:
        return base.astype(float)
    if base.size == 1:
        return np.full(n_rows, float(base.ravel()[0]))
    raise ValueError("Unexpected shape for base values returned by SHAP")


def _write_outputs(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    booster: lgb.Booster,
    features: pd.DataFrame,
    meta: Optional[pd.DataFrame],
    output_dir: Path,
    tag: str,
    write_plot: bool,
) -> None:
    feature_names = list(booster.feature_name())

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "base_value", _expand_base_values(base_values, len(shap_df)))

    preds = booster.predict(features, num_iteration=getattr(booster, "best_iteration", None))
    shap_df.insert(1, "prediction", preds)

    if meta is not None:
        for col in reversed(meta.columns):
            shap_df.insert(0, col, meta[col].to_numpy())

    out_values = output_dir / f"shap_values_{tag}.parquet"
    shap_df.to_parquet(out_values, index=False)
    logger.info("Wrote SHAP values to %s", out_values)

    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "std_shap": shap_values.std(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    summary["num_rows"] = len(features)

    out_summary = output_dir / f"shap_summary_{tag}.csv"
    summary.to_csv(out_summary, index=False)
    logger.info("Wrote SHAP summary to %s", out_summary)

    if write_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(
            shap_values,
            features,
            feature_names=feature_names,
            show=False,
            plot_type="bar" if len(features) <= 1 else None,
        )
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        fig.tight_layout()
        out_plot = output_dir / f"shap_summary_{tag}.png"
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)
        logger.info("Saved SHAP plot to %s", out_plot)


def _build_explainer(
    booster: lgb.Booster,
    background: Optional[pd.DataFrame],
) -> shap.TreeExplainer:
    if background is not None and len(background) > 0:
        return shap.TreeExplainer(booster, data=background, feature_perturbation="tree_path_dependent")
    return shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run SHAP analysis for a LightGBM model run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing model.txt and artifacts")
    parser.add_argument("--split", choices=["train", "val", "test", "inference"], default="test")
    parser.add_argument("--prepared-dir", type=Path, help="Override prepared data directory for evaluation splits")
    parser.add_argument("--duckdb", type=Path, help="DuckDB database file (required for inference split)")
    parser.add_argument("--feature-key", type=str, help="Optional feature key filter when reading from DuckDB")
    parser.add_argument("--limit", type=int, help="Limit number of inference rows fetched")
    parser.add_argument("--start", type=str, help="Start timestamp (inclusive, ISO, UTC) for inference rows")
    parser.add_argument("--end", type=str, help="End timestamp (inclusive, ISO, UTC) for inference rows")
    parser.add_argument("--max-rows", type=int, help="Sample at most this many rows before computing SHAP")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--background-sample", type=int, help="Sample size for background dataset")
    parser.add_argument("--class-index", type=int, help="Class index to explain for classification models")
    parser.add_argument("--output-dir", type=Path, help="Directory to write SHAP artifacts (defaults to run dir)")
    parser.add_argument("--plot", action="store_true", help="Generate SHAP summary plot (png)")
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Persist the feature values used for inference (CSV) alongside SHAP outputs",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    booster = load_booster(run_dir / "model.txt")

    if args.split == "inference":
        if not args.duckdb:
            raise ValueError("--duckdb is required for inference split")
        if args.start and args.end:
            start_ts = pd.to_datetime(args.start, utc=True)
            end_ts = pd.to_datetime(args.end, utc=True)
            if start_ts > end_ts:
                raise ValueError("--start must be earlier than or equal to --end")
            start_str, end_str = start_ts.isoformat(), end_ts.isoformat()
        else:
            start_str = pd.to_datetime(args.start, utc=True).isoformat() if args.start else None
            end_str = pd.to_datetime(args.end, utc=True).isoformat() if args.end else None

        features, meta, feature_values = _load_inference_frame(
            run_dir,
            booster,
            args.duckdb,
            args.feature_key,
            args.limit,
            start_str,
            end_str,
            args.max_rows,
            args.random_seed,
        )

        def _sanitize(value: str) -> str:
            return (
                value.replace(":", "")
                .replace("-", "")
                .replace("T", "")
                .replace("Z", "")
                .replace("+", "")
            )

        base_tag = "inference"
        if args.feature_key:
            base_tag += f"_{args.feature_key}"
        tag = base_tag
        if start_str or end_str:
            if start_str:
                tag += f"_from_{_sanitize(start_str)}"
            if end_str:
                tag += f"_to_{_sanitize(end_str)}"
    else:
        features, meta = _load_split_frame(
            run_dir,
            booster,
            args.split,
            args.prepared_dir,
            args.max_rows,
            args.random_seed,
        )
        tag = args.split
        feature_values = None

    background = None
    if args.background_sample:
        sample_size = min(args.background_sample, len(features))
        background = features.sample(n=sample_size, random_state=args.random_seed)

    explainer = _build_explainer(booster, background)
    shap_values_raw = explainer.shap_values(features)
    expected = explainer.expected_value
    shap_array, base_values = _select_component(shap_values_raw, expected, args.class_index)

    output_dir = args.output_dir.resolve() if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_outputs(
        shap_array,
        base_values,
        booster,
        features,
        meta,
        output_dir,
        tag,
        args.plot,
    )

    if feature_values is not None and args.save_features:
        features_out = output_dir / f"features_used_{tag}.csv"
        feature_values.to_csv(features_out, index=False)
        logger.info("Saved feature values to %s", features_out)


if __name__ == "__main__":
    main()
