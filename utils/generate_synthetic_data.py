#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small synthetic dataset for pipeline smoke tests")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--rows", type=int, default=240, help="Number of rows")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    n = args.rows

    ts = pd.date_range("2024-01-01", periods=n, freq="H")
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    noise = 0.05 * rng.normal(size=n)
    y = 0.3 * f1 - 0.2 * f2 + 0.1 * f3 + noise

    df = pd.DataFrame({
        "timestamp": ts,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "y_logret_24h": y,
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote synthetic dataset to: {args.output}")


if __name__ == "__main__":
    main()


