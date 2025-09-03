#!/usr/bin/env python3
"""
Lightweight smoke tests for strategies.quantile_signals using plain asserts.
Run:
  /Users/noel/projects/trading_cex/venv/bin/python strategies/tests/smoke_quantile_signals.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on sys.path for direct execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.quantile_signals import (
    DEFAULT_QCOLS,
    add_basic_signals,
    apply_monotonic_fix,
    compute_static_thresholds,
    detect_monotonic_violations,
    estimate_prob_up,
    expected_return_left_riemann,
    expected_return_trapezoid,
)


def _make_df_from_row(qvals):
    assert len(qvals) == len(DEFAULT_QCOLS)
    data = {col: [val] for col, val in zip(DEFAULT_QCOLS, qvals)}
    data["y_true"] = [0.0]
    return pd.DataFrame(data)


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def test_monotonic_fix_and_diagnostics():
    q = [0.00, -0.01, 0.0, -0.002, 0.0, 0.01, 0.02, 0.025, 0.03]
    df = _make_df_from_row(q)
    diags = detect_monotonic_violations(df)
    assert int(diags.cross_violations.iloc[0]) >= 2
    assert _approx(diags.max_cross_gap.iloc[0], 0.01)

    fixed, _ = apply_monotonic_fix(df)
    fixed_q = fixed[DEFAULT_QCOLS].iloc[0].to_numpy()
    assert np.all(np.diff(fixed_q) >= -1e-12)


def test_expected_returns_constant():
    q = [0.02] * len(DEFAULT_QCOLS)
    df = _make_df_from_row(q)
    row = df.iloc[0]
    e_avg = expected_return_trapezoid(row)
    e_left = expected_return_left_riemann(row)
    assert _approx(e_avg, 0.02, 1e-10)
    assert _approx(e_left, 0.02, 1e-10)


def test_expected_returns_symmetric_near_zero():
    q = [-0.10, -0.08, -0.05, -0.02, 0.0, 0.02, 0.05, 0.08, 0.10]
    df = _make_df_from_row(q)
    row = df.iloc[0]
    e_avg = expected_return_trapezoid(row)
    e_left = expected_return_left_riemann(row)
    # Trapezoid should be near zero for symmetric q
    assert abs(e_avg) < 1e-3
    # Left Riemann is conservative for increasing Q(p): e_left <= e_avg
    assert e_left <= e_avg + 1e-12
    assert min(q) <= e_avg <= max(q)
    assert min(q) <= e_left <= max(q)


def test_probability_extremes_and_interpolation():
    q_pos = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    q_neg = [-0.009, -0.008, -0.007, -0.006, -0.005, -0.004, -0.003, -0.002, -0.001]
    df_pos = _make_df_from_row(q_pos)
    df_neg = _make_df_from_row(q_neg)
    assert estimate_prob_up(df_pos.iloc[0]) == 1.0
    assert estimate_prob_up(df_neg.iloc[0]) == 0.0

    q_cross = [-0.05, -0.04, -0.03, -0.02, -0.02, 0.02, 0.03, 0.04, 0.05]
    df_cross = _make_df_from_row(q_cross)
    p_up = estimate_prob_up(df_cross.iloc[0])
    assert _approx(p_up, 0.375, 1e-6)


def test_static_thresholds_and_signals():
    f1 = pd.DataFrame({"pred_q90": [1.0, 2.0, 3.0], "pred_q10": [1.0, 2.0, 3.0]})
    f2 = pd.DataFrame({"pred_q90": [4.0, 5.0, 6.0], "pred_q10": [4.0, 5.0, 6.0]})
    th = compute_static_thresholds([f1, f2], ["pred_q90", "pred_q10"], [0.90, 0.10])
    arr = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    assert _approx(th["pred_q90@0.9000"], float(np.percentile(arr, 90)))
    assert _approx(th["pred_q10@0.1000"], float(np.percentile(arr, 10)))

    df = pd.DataFrame({
        "exp_ret_avg": [-1e-5, 0.0, 2e-5],
        "exp_ret_conservative": [-1e-5, 0.0, 2e-5],
        "prob_up": [0.40, 0.50, 0.60],
        "pred_q90": [0.01, 0.02, 0.03],
        "pred_q10": [-0.03, -0.02, -0.01],
    })
    out = add_basic_signals(
        df,
        selected_method="avg",
        exp_zero_band=1e-6,
        prob_thresholds=(0.55, 0.45),
        quantile_thresholds=(0.02, -0.02),
    )
    assert out["signal_exp"].tolist() == [-1, 0, 1]
    assert out["signal_prob"].tolist() == [-1, 0, 1]
    assert out["signal_quantile"].tolist() == [-1, 1, 1]


def main() -> None:
    test_monotonic_fix_and_diagnostics()
    test_expected_returns_constant()
    test_expected_returns_symmetric_near_zero()
    test_probability_extremes_and_interpolation()
    test_static_thresholds_and_signals()
    print("All quantile_signals smoke tests passed.")


if __name__ == "__main__":
    main()


