#!/usr/bin/env python3
"""
Build analysis-ready 15-minute market history CSV from raw JSON.

Differences vs the older cleaner:
1. De-duplicates multiple snapshots in the same minute bucket
   (default policy: keep latest snapshot by `t`).
2. Enforces full minute coverage 0..14 after de-duplication.
3. Emits optional diagnostics for dropped markets.

Usage:
    python apps/btc1m_trade_analysis/build_btc_15m_polymarket_price_history.py \
        apps/btc1m_trade_analysis/btc_15m_price_history.json \
        apps/btc1m_trade_analysis/btc_15m_polymarket_price_history.csv \
        --report apps/btc1m_trade_analysis/btc_15m_polymarket_price_history_report.json \
        --dropped-csv apps/btc1m_trade_analysis/btc_15m_polymarket_price_history_dropped.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WINDOW_SECONDS = 15 * 60
MINUTES_PER_MARKET = 15
OUTPUT_FIELDS = [
    "market_timestamp",
    "datetime",
    "minute",
    "t",
    "snapshot_datetime",
    "p",
]
DROPPED_FIELDS = [
    "slug",
    "market_timestamp",
    "reason",
    "in_window_points",
    "unique_minutes",
    "missing_minutes",
    "duplicate_minutes",
]


@dataclass
class Stats:
    markets_total: int = 0
    markets_kept: int = 0
    markets_dropped: int = 0
    rows_written: int = 0
    rows_in_window_total: int = 0
    duplicate_points_resolved: int = 0
    dropped_reasons: Counter | None = None

    def __post_init__(self) -> None:
        if self.dropped_reasons is None:
            self.dropped_reasons = Counter()


def to_dt_str(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean 15-minute Polymarket history CSV.")
    parser.add_argument(
        "input_json",
        nargs="?",
        default="apps/btc1m_trade_analysis/btc_15m_price_history.json",
        help="Path to raw JSON market history.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="apps/btc1m_trade_analysis/btc_15m_polymarket_price_history.csv",
        help="Path to clean output CSV.",
    )
    parser.add_argument(
        "--dedupe",
        choices=["latest", "earliest"],
        default="latest",
        help="Which snapshot to keep when multiple points map to the same minute.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--dropped-csv",
        default="",
        help="Optional CSV output path with dropped market diagnostics.",
    )
    return parser.parse_args()


def choose_point(existing: dict[str, Any], new: dict[str, Any], policy: str) -> dict[str, Any]:
    if policy == "latest":
        return new if int(new["t"]) >= int(existing["t"]) else existing
    return new if int(new["t"]) <= int(existing["t"]) else existing


def clean_market(
    slug: str,
    market_data: dict[str, Any],
    dedupe_policy: str,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, int, int]:
    """
    Returns:
        kept_rows_or_none, dropped_detail_or_none, in_window_points, duplicate_points_resolved
    """
    start = int(market_data["market_timestamp"])
    end = start + WINDOW_SECONDS
    history = market_data.get("price_history", [])

    in_window_points = []
    for p in history:
        if not isinstance(p, dict):
            continue
        if "t" not in p or "p" not in p:
            continue
        t = int(p["t"])
        if start <= t < end:
            in_window_points.append({"t": t, "p": float(p["p"])})

    if not in_window_points:
        dropped = {
            "slug": slug,
            "market_timestamp": start,
            "reason": "no_in_window_points",
            "in_window_points": 0,
            "unique_minutes": 0,
            "missing_minutes": ",".join(str(i) for i in range(MINUTES_PER_MARKET)),
            "duplicate_minutes": "",
        }
        return None, dropped, 0, 0

    by_minute: dict[int, dict[str, Any]] = {}
    minute_counts = Counter()
    for p in in_window_points:
        minute = (p["t"] - start) // 60
        if minute < 0 or minute >= MINUTES_PER_MARKET:
            continue
        minute_counts[minute] += 1
        if minute not in by_minute:
            by_minute[minute] = p
        else:
            by_minute[minute] = choose_point(by_minute[minute], p, dedupe_policy)

    duplicates_resolved = sum(max(0, c - 1) for c in minute_counts.values())
    missing_minutes = [m for m in range(MINUTES_PER_MARKET) if m not in by_minute]
    duplicate_minutes = [m for m, c in minute_counts.items() if c > 1]

    if missing_minutes:
        dropped = {
            "slug": slug,
            "market_timestamp": start,
            "reason": "missing_minutes_after_dedupe",
            "in_window_points": len(in_window_points),
            "unique_minutes": len(by_minute),
            "missing_minutes": ",".join(str(m) for m in missing_minutes),
            "duplicate_minutes": ",".join(str(m) for m in duplicate_minutes),
        }
        return None, dropped, len(in_window_points), duplicates_resolved

    kept_rows: list[dict[str, Any]] = []
    for minute in range(MINUTES_PER_MARKET):
        p = by_minute[minute]
        minute_ts = start + minute * 60
        kept_rows.append({
            "market_timestamp": start,
            "datetime": to_dt_str(minute_ts),
            "minute": minute,
            "t": int(p["t"]),
            "snapshot_datetime": to_dt_str(int(p["t"])),
            "p": float(p["p"]),
        })

    return kept_rows, None, len(in_window_points), duplicates_resolved


def clean_price_history(
    input_path: Path,
    output_path: Path,
    dedupe_policy: str,
    report_path: Path | None,
    dropped_csv_path: Path | None,
) -> Stats:
    with input_path.open() as f:
        data = json.load(f)

    stats = Stats()
    stats.markets_total = len(data)

    clean_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []

    for slug, market_data in data.items():
        kept, dropped, in_window_cnt, dup_resolved = clean_market(
            slug=slug,
            market_data=market_data,
            dedupe_policy=dedupe_policy,
        )
        stats.rows_in_window_total += in_window_cnt
        stats.duplicate_points_resolved += dup_resolved

        if kept is None:
            stats.markets_dropped += 1
            if dropped is not None:
                stats.dropped_reasons[dropped["reason"]] += 1
                dropped_rows.append(dropped)
            continue

        stats.markets_kept += 1
        clean_rows.extend(kept)

    stats.rows_written = len(clean_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(clean_rows)

    if dropped_csv_path is not None:
        dropped_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with dropped_csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DROPPED_FIELDS)
            w.writeheader()
            w.writerows(dropped_rows)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "input_json": str(input_path),
            "output_csv": str(output_path),
            "dedupe_policy": dedupe_policy,
            "markets_total": stats.markets_total,
            "markets_kept": stats.markets_kept,
            "markets_dropped": stats.markets_dropped,
            "rows_written": stats.rows_written,
            "rows_in_window_total": stats.rows_in_window_total,
            "duplicate_points_resolved": stats.duplicate_points_resolved,
            "dropped_reasons": dict(stats.dropped_reasons),
        }
        with report_path.open("w") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    return stats


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_csv)
    report_path = Path(args.report) if args.report else None
    dropped_csv_path = Path(args.dropped_csv) if args.dropped_csv else None

    stats = clean_price_history(
        input_path=input_path,
        output_path=output_path,
        dedupe_policy=args.dedupe,
        report_path=report_path,
        dropped_csv_path=dropped_csv_path,
    )

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Dedupe:  {args.dedupe}")
    print(f"Markets: kept {stats.markets_kept:,} / total {stats.markets_total:,}")
    print(f"Dropped: {stats.markets_dropped:,}")
    print(f"Rows:    {stats.rows_written:,}")
    print(f"In-window points seen: {stats.rows_in_window_total:,}")
    print(f"Duplicate in-window points resolved: {stats.duplicate_points_resolved:,}")
    if stats.dropped_reasons:
        print("Dropped reasons:")
        for reason, count in stats.dropped_reasons.most_common():
            print(f"  {reason}: {count:,}")

    if report_path is not None:
        print(f"Report JSON: {report_path}")
    if dropped_csv_path is not None:
        print(f"Dropped CSV: {dropped_csv_path}")


if __name__ == "__main__":
    main()
