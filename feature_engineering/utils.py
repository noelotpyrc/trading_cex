import numpy as np
import pandas as pd
from typing import Dict


def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """Validate that `data` has required OHLCV columns and is non-empty.

    Required columns: 'open', 'high', 'low', 'close', 'volume'.
    Returns True if valid; False otherwise.
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if data is None or not isinstance(data, pd.DataFrame):
        return False
    if len(data) == 0:
        return False
    if not all(col in data.columns for col in required_cols):
        return False
    return True


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Safely divide two numbers.

    - Returns `default` if denominator is zero or either input is NaN.
    - Otherwise returns numerator / denominator.
    """
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return default
    return numerator / denominator


