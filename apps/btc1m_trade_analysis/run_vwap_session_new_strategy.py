from __future__ import annotations

import argparse
import sys
from copy import deepcopy
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from apps.btc1m_trade_analysis import vwap_session_new_strategy_simple_backtest as strat


def _build_required_cols(entry_config: dict) -> list[str]:
    cols = {
        strat.DATETIME_COL,
        "open",
        "high",
        "low",
        "close",
    }
    for filt in entry_config["filters"]:
        cols.add(filt["col"])
    cols.add(entry_config["direction"]["long_rule"]["col"])
    cols.add(entry_config["direction"]["short_rule"]["col"])
    return sorted(cols)


def _load_data(path: Path, cols: Iterable[str], year: int | None, chunksize: int) -> pd.DataFrame:
    if year is None:
        return pd.read_csv(path, usecols=list(cols))

    chunks = []
    for chunk in pd.read_csv(path, usecols=list(cols), chunksize=chunksize):
        dt = pd.to_datetime(chunk[strat.DATETIME_COL], errors="coerce", utc=True)
        mask = dt.dt.year == year
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if chunks:
        return pd.concat(chunks, ignore_index=True)

    return pd.DataFrame(columns=list(cols))


def _load_json_configs(path: Path) -> tuple[dict, dict, dict | None, str | None]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entry_config = (
        data.get("entry_config")
        or data.get("entry")
        or data.get("ENTRY_CONFIG")
    )
    exit_config = (
        data.get("exit_config")
        or data.get("exit")
        or data.get("EXIT_CONFIG")
    )

    if entry_config is None or exit_config is None:
        raise ValueError("Config JSON must include entry_config and exit_config.")

    grid_config = data.get("grid_search") or data.get("grid") or None
    data_path = data.get("data_path")

    return entry_config, exit_config, grid_config, data_path


def _calc_overall_stats(trades_df: pd.DataFrame) -> dict:
    total_trades = len(trades_df)
    if total_trades == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_roi": 0.0,
            "final_capital": 100_000.0,
        }

    win_trades = len(trades_df[trades_df["pnl_pct"] > 0])
    win_rate = win_trades / total_trades * 100
    avg_pnl = trades_df["pnl_pct"].mean()

    capital = 100_000.0
    max_position = 100_000.0
    for pnl_pct in trades_df.sort_values("entry_time")["pnl_pct"]:
        position_size = min(capital, max_position)
        capital += position_size * (pnl_pct / 100)

    total_roi = (capital - 100_000.0) / 100_000.0 * 100
    return {
        "trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_roi": total_roi,
        "final_capital": capital,
    }


def _calc_direction_stats(trades_df: pd.DataFrame, direction: str) -> dict:
    subset = trades_df[trades_df["direction"] == direction]
    if len(subset) == 0:
        return {
            f"{direction}_trades": 0,
            f"{direction}_win_rate": 0.0,
            f"{direction}_avg_pnl": 0.0,
        }

    wins = len(subset[subset["pnl_pct"] > 0])
    win_rate = wins / len(subset) * 100
    avg_pnl = subset["pnl_pct"].mean()
    return {
        f"{direction}_trades": len(subset),
        f"{direction}_win_rate": win_rate,
        f"{direction}_avg_pnl": avg_pnl,
    }


def grid_search_exit_configs(
    entry_base: dict,
    exit_base: dict,
    stop_loss_pcts: Iterable[float],
    expiry_bars_list: Iterable[int],
    directions: Iterable[str],
    years: Iterable[int],
    data_path: Path,
    chunksize: int,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    entry_base = deepcopy(entry_base)
    exit_base = deepcopy(exit_base)

    include_long = "long" in directions or "both" in directions
    include_short = "short" in directions or "both" in directions

    cols = _build_required_cols(entry_base)
    data_by_year = {year: _load_data(data_path, cols, year, chunksize) for year in years}

    rows = []
    total_runs = len(list(stop_loss_pcts)) * len(list(expiry_bars_list)) * len(list(directions)) * len(list(years))
    completed = 0
    for stop_loss_pct in stop_loss_pcts:
        for expiry_bars in expiry_bars_list:
            for direction in directions:
                for year in years:
                    completed += 1
                    print(
                        f"[{completed}/{total_runs}] stop_loss_pct={stop_loss_pct} "
                        f"expiry_bars={expiry_bars} direction={direction} year={year}"
                    )
                    entry_config = deepcopy(entry_base)
                    entry_config["direction"]["filter"] = direction

                    exit_config = deepcopy(exit_base)
                    exit_config["stop_loss_pct"] = stop_loss_pct
                    exit_config["expiry_bars"] = expiry_bars

                    trades = strat.run_backtest(data_by_year[year], entry_config, exit_config)

                    overall = _calc_overall_stats(trades)
                    row = {
                        "stop_loss_pct": stop_loss_pct,
                        "expiry_bars": expiry_bars,
                        "direction": direction,
                        "year": year,
                        **overall,
                    }
                    if include_long:
                        row.update(_calc_direction_stats(trades, "long"))
                    if include_short:
                        row.update(_calc_direction_stats(trades, "short"))

                    rows.append(row)

    report = pd.DataFrame(rows)
    if not report.empty:
        print(report.to_string(index=False))
        if csv_path is not None:
            report.to_csv(csv_path, index=False)
            print(f"Saved grid search report to {csv_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VWAP session new strategy backtest.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to CSV data file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON file containing entry_config and exit_config.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search using grid_search config block from the JSON file.",
    )
    parser.add_argument("--year", type=int, default=None, help="Filter to a single year (e.g., 2023).")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="CSV chunk size for filtering.")

    args = parser.parse_args()

    entry_config, exit_config, grid_config, config_data_path = _load_json_configs(args.config)
    data_path = args.data_path
    if data_path is None:
        if grid_config and grid_config.get("data_path"):
            data_path = Path(grid_config["data_path"])
        elif config_data_path:
            data_path = Path(config_data_path)
        else:
            parser.error("Missing data path. Provide --data-path or add data_path to the config JSON.")

    if args.grid_search:
        if grid_config is None:
            raise ValueError("grid_search flag set but config JSON has no grid_search block.")
        stop_loss_pcts = grid_config.get("stop_loss_pcts")
        expiry_bars_list = grid_config.get("expiry_bars")
        directions = grid_config.get("directions")
        years = grid_config.get("years")
        csv_path_value = grid_config.get("csv_path")
        csv_path = Path(csv_path_value) if csv_path_value else None

        missing = [
            name
            for name, value in [
                ("stop_loss_pcts", stop_loss_pcts),
                ("expiry_bars", expiry_bars_list),
                ("directions", directions),
                ("years", years),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"grid_search config missing: {', '.join(missing)}")

        grid_search_exit_configs(
            entry_config,
            exit_config,
            stop_loss_pcts,
            expiry_bars_list,
            directions,
            years,
            data_path,
            args.chunksize,
            csv_path,
        )
        return

    cols = _build_required_cols(entry_config)
    df = _load_data(data_path, cols, args.year, args.chunksize)

    print(f"Loaded rows: {len(df):,}")
    trades = strat.run_backtest(df, entry_config, exit_config)
    print(f"Trades: {len(trades)}")
    if len(trades) > 0:
        win_rate = (trades["pnl_pct"] > 0).mean() * 100
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Average PnL: {trades['pnl_pct'].mean():.4f}%")
        print(trades.head())
        strat.analyze_results(trades)


if __name__ == "__main__":
    main()
