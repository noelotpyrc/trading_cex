#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    # Ensure project root on path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Lazy import after path fix
    from model.lgbm_inference import predict_from_csv

    model_root = Path("/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60")
    assert model_root.exists(), f"Model root does not exist: {model_root}"

    # Pick a run_* directory (random choice per user request)
    run_dirs = [p for p in model_root.glob("run_*") if p.is_dir()]
    assert run_dirs, f"No run_* directories found under {model_root}"
    run_dir = random.choice(sorted(run_dirs))
    print("Selected run dir:", run_dir)

    # Load paths.json to locate prepared splits
    paths_json = run_dir / "paths.json"
    assert paths_json.exists(), f"paths.json missing in {run_dir}"
    paths = json.loads(paths_json.read_text())
    prepared_dir = Path(paths.get("prepared_data_dir", ""))
    assert prepared_dir.exists(), f"prepared_data_dir missing or invalid: {prepared_dir}"

    # Use the test split features
    x_test_csv = prepared_dir / "X_test.csv"
    assert x_test_csv.exists(), f"X_test.csv not found: {x_test_csv}"

    # Run inference against the selected run's model
    preds_df = predict_from_csv(input_csv=x_test_csv, model_path=run_dir)
    assert "y_pred" in preds_df.columns, "predict_from_csv must return a y_pred column"

    # Compare with saved predictions from the run directory
    saved_pred_csv = run_dir / "pred_test.csv"
    assert saved_pred_csv.exists(), f"pred_test.csv not found in {run_dir}"
    saved_df = pd.read_csv(saved_pred_csv)
    assert "y_pred" in saved_df.columns, "pred_test.csv must contain a y_pred column"

    assert len(preds_df) == len(saved_df), f"Prediction length mismatch: {len(preds_df)} vs {len(saved_df)}"
    a = preds_df["y_pred"].to_numpy()
    b = saved_df["y_pred"].to_numpy()

    # Allow tiny numerical differences due to environment/library variations
    assert np.allclose(a, b, rtol=1e-8, atol=1e-10), "Predictions differ from saved pred_test.csv beyond tolerance"
    print("Inference predict_from_csv test passed for:", run_dir)


if __name__ == "__main__":
    main()



