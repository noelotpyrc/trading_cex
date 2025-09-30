#!/usr/bin/env python3
from __future__ import annotations

"""
Compare shapes and basic stats between two lookbacks_1H.pkl stores.

Defaults target files:
  A: /Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/lookbacks_1H.pkl
  B: /Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60/lookbacks_1H.pkl

Prints:
  - File sizes and key metadata (timeframe, base_index len, lookback_base_rows, rows count)
  - Counts of empty/missing lookback DataFrames
  - Sample row shapes (head/mid/tail)
  - Intersection/only-in-A/only-in-B counts for timestamp keys
  - Approximate memory usage per sampled lookback rows

This is a diagnostic script (no assertions) intended to explain differences
in “rows count vs pickle size on disk”.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


def _human(nbytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(nbytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _load_store(path: Path) -> Dict:
    store = pd.read_pickle(path)
    # Normalize base_index to pandas DatetimeIndex (naive)
    try:
        base_idx = pd.to_datetime(pd.Index(store.get('base_index')), errors='coerce', utc=True)
        store['base_index'] = base_idx.tz_convert('UTC').tz_localize(None)
    except Exception:
        pass
    return store


def _rows_stats(rows: Dict[str, pd.DataFrame]) -> Dict[str, int | float]:
    total = len(rows)
    empty = 0
    none = 0
    lengths: List[int] = []
    for lb in rows.values():
        if lb is None:
            none += 1
            continue
        if getattr(lb, 'empty', False):
            empty += 1
        else:
            try:
                lengths.append(int(len(lb)))
            except Exception:
                pass
    out = {
        'total_rows_entries': total,
        'none_entries': none,
        'empty_entries': empty,
        'min_len': min(lengths) if lengths else 0,
        'max_len': max(lengths) if lengths else 0,
        'median_len': float(np.median(lengths)) if lengths else 0.0,
    }
    return out


def _sample_keys(rows: Dict[str, pd.DataFrame], k: int = 3) -> List[str]:
    keys = sorted(rows.keys())
    if not keys:
        return []
    idxs = sorted(set([0, len(keys)//2, len(keys)-1]))
    if k > 3:
        # add a few randoms for more coverage
        rng = np.random.default_rng(0)
        extra = rng.choice(len(keys), size=min(k-3, len(keys)), replace=False).tolist()
        idxs = sorted(set(idxs + extra))
    return [keys[i] for i in idxs]


def _one_store_report(label: str, path: Path, store: Dict) -> None:
    print(f"[{label}] {path}")
    try:
        size = path.stat().st_size
        print("  file_size:", _human(size))
    except Exception:
        pass
    tf = store.get('timeframe')
    base_idx = store.get('base_index')
    lookback_rows = store.get('lookback_base_rows')
    rows = store.get('rows', {})
    print(f"  timeframe: {tf}")
    try:
        print(f"  base_index_len: {len(base_idx)}  range: {base_idx[0]} .. {base_idx[-1]}")
    except Exception:
        print(f"  base_index_len: {len(base_idx) if base_idx is not None else 0}")
    print(f"  lookback_base_rows: {lookback_rows}")
    print(f"  rows_entries: {len(rows)}")
    st = _rows_stats(rows)
    print(f"  rows_stats: total={st['total_rows_entries']} none={st['none_entries']} empty={st['empty_entries']} min_len={st['min_len']} median_len={st['median_len']} max_len={st['max_len']}")
    # Sample shapes
    sk = _sample_keys(rows, k=5)
    for key in sk:
        lb = rows.get(key)
        if lb is None:
            shape = (0, 0)
            cols = []
            idx_range = "<none>"
        else:
            shape = lb.shape
            cols = list(lb.columns)
            try:
                idx_range = f"{lb.index[0]} .. {lb.index[-1]}"
            except Exception:
                idx_range = "<index?>"
        print(f"    sample {key}: shape={shape} cols={cols[:6]} idx={idx_range}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare two lookbacks_1H.pkl stores')
    parser.add_argument('--a', default='/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/lookbacks_1H.pkl', help='Path to first lookbacks_1H.pkl')
    parser.add_argument('--b', default='/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60/lookbacks_1H.pkl', help='Path to second lookbacks_1H.pkl')
    args = parser.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)
    if not path_a.exists():
        print(f'SKIP: file not found: {path_a}')
        return
    if not path_b.exists():
        print(f'SKIP: file not found: {path_b}')
        return

    store_a = _load_store(path_a)
    store_b = _load_store(path_b)

    print('=== Store A ===')
    _one_store_report('A', path_a, store_a)
    print('=== Store B ===')
    _one_store_report('B', path_b, store_b)

    # Compare key sets
    keys_a = set(store_a.get('rows', {}).keys())
    keys_b = set(store_b.get('rows', {}).keys())
    inter = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    print('=== Key Set Comparison ===')
    print(f'  A∩B: {len(inter)}  A\B: {len(only_a)}  B\A: {len(only_b)}')
    # Show a few examples to orient
    for title, s in [('A_only_head', sorted(list(only_a))[:3]), ('B_only_head', sorted(list(only_b))[:3])]:
        print(f'  {title}: {s}')

    # Approximate memory usage for a small sample (helps explain pickle size)
    def _approx_df_mem(lb: pd.DataFrame | None) -> int:
        if lb is None or lb.empty:
            return 0
        try:
            return int(lb.memory_usage(deep=True).sum())
        except Exception:
            return 0

    rng = np.random.default_rng(0)
    sample_keys = list(inter) if inter else list(keys_a or keys_b)
    if sample_keys:
        idxs = rng.choice(len(sample_keys), size=min(20, len(sample_keys)), replace=False).tolist()
        samp = [sample_keys[i] for i in idxs]
        mem_a = sum(_approx_df_mem(store_a['rows'].get(k)) for k in samp)
        mem_b = sum(_approx_df_mem(store_b['rows'].get(k)) for k in samp)
        print('=== Approx Memory (sample of up to 20 lookbacks) ===')
        print(f'  A sample bytes: {mem_a} ({_human(mem_a)})')
        print(f'  B sample bytes: {mem_b} ({_human(mem_b)})')

    print('lookbacks_pkl comparison OK')


if __name__ == '__main__':
    main()

