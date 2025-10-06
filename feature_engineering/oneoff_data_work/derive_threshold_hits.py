"""Derive binary hit targets from MFE/MAE columns in the consolidated targets.csv.

This script reads the feature-store `targets.csv`, thresholds the existing
Max-Favorable-Excursion (MFE) and Max-Adverse-Excursion (MAE) columns, and
emits a new CSV containing only the timestamp and newly derived binary columns
that mark whether each forward window touched the desired upside or downside
levels.

By default it scans all available horizons (e.g. 24h, 48h) in the input file
and evaluates a configurable grid of upside/downside thresholds.  Use the CLI
flags to customise paths, horizons, and threshold percent moves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


# Project-specific defaults – adjust as needed for other symbols or intervals.
BASE_DIR = Path("/Volumes/Extreme SSD/trading_data/cex")
TRAINING_DIR = BASE_DIR / "training" / "BINANCE_BTCUSDT.P, 60"
DEFAULT_TARGETS_CSV = TRAINING_DIR / "feature_store" / "targets.csv"
DEFAULT_OUTPUT_CSV = TRAINING_DIR / "feature_store" / "targets_with_hit_flags.csv"

DEFAULT_UP_THRESHOLDS = (0.02, 0.03, 0.05)
DEFAULT_DOWN_THRESHOLDS = (0.01, 0.02, 0.03)


def parse_thresholds(arg: str | None, fallback: Sequence[float]) -> tuple[float, ...]:
    if not arg:
        return tuple(fallback)
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("Threshold list cannot be empty")
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:  # pragma: no cover - simple CLI validation
            raise ValueError(f"Invalid threshold '{part}'") from exc
        if value <= 0:
            raise ValueError(f"Thresholds must be positive, got {value}")
        values.append(value)
    return tuple(values)


def parse_horizons(arg: str | None, auto_horizons: Sequence[str]) -> tuple[str, ...]:
    if not arg:
        return tuple(auto_horizons)
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("Horizon list cannot be empty")
    invalid = sorted(set(parts) - set(auto_horizons))
    if invalid:
        raise ValueError(f"Unknown horizons requested: {invalid!r}. Available: {sorted(auto_horizons)!r}")
    return tuple(parts)


def format_threshold(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def infer_horizons(df: pd.DataFrame) -> list[str]:
    horizons: list[str] = []
    for column in df.columns:
        if column.startswith("y_mfe_") and "__" not in column:
            horizons.append(column[len("y_mfe_"):])
    return sorted(set(horizons))


def make_indicator(base: pd.Series, condition: pd.Series) -> pd.Series:
    indicator = pd.Series(pd.NA, index=base.index, dtype="Int8")
    valid_mask = base.notna()
    indicator.loc[valid_mask & condition] = 1
    indicator.loc[valid_mask & ~condition] = 0
    return indicator


def derive_hit_columns(
    df: pd.DataFrame,
    horizons: Iterable[str],
    up_thresholds: Sequence[float],
    down_thresholds: Sequence[float],
) -> dict[str, pd.Series]:
    new_columns: dict[str, pd.Series] = {}

    for horizon in horizons:
        mfe_col = f"y_mfe_{horizon}"
        mae_col = f"y_mae_{horizon}"

        if mfe_col not in df.columns:
            raise KeyError(f"Missing MFE column '{mfe_col}' in input data")
        if mae_col not in df.columns:
            raise KeyError(f"Missing MAE column '{mae_col}' in input data")

        mfe_values = df[mfe_col]
        mae_values = df[mae_col]

        for threshold in up_thresholds:
            column_name = f"y_hit_up_{horizon}_ge_{format_threshold(threshold)}"
            condition = mfe_values >= threshold
            new_columns[column_name] = make_indicator(mfe_values, condition)

        for threshold in down_thresholds:
            column_name = f"y_hit_down_{horizon}_ge_{format_threshold(threshold)}"
            condition = mae_values <= -threshold
            new_columns[column_name] = make_indicator(mae_values, condition)

    return new_columns


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--up-thresholds", help="Comma-separated list of upside thresholds (e.g. 0.02,0.03)")
    parser.add_argument("--down-thresholds", help="Comma-separated list of downside thresholds (e.g. 0.01,0.02)")
    parser.add_argument("--horizons", help="Comma-separated list of horizon labels to include (e.g. 24h,48h)")

    args = parser.parse_args()

    df = pd.read_csv(args.targets_csv)
    auto_horizons = infer_horizons(df)
    if not auto_horizons:
        raise RuntimeError("Could not infer any horizons from MFE columns in the input file")

    up_thresholds = parse_thresholds(args.up_thresholds, DEFAULT_UP_THRESHOLDS)
    down_thresholds = parse_thresholds(args.down_thresholds, DEFAULT_DOWN_THRESHOLDS)
    horizons = parse_horizons(args.horizons, auto_horizons)

    new_columns = derive_hit_columns(df, horizons, up_thresholds, down_thresholds)

    if "timestamp" not in df.columns:
        raise KeyError("Input targets CSV must include a 'timestamp' column")

    new_df = pd.DataFrame(new_columns, index=df.index)
    output_df = pd.concat([df[["timestamp"]], new_df], axis=1)
    output_df.to_csv(args.output_csv, index=False)

    print(
        f"Wrote {len(new_columns)} new columns for horizons {horizons} "
        f"to '{args.output_csv}'."
    )


if __name__ == "__main__":
    run()


