from __future__ import annotations

import pandas as pd

def trailing_mean_forecast(
        history: pd.Series,
        window: int = 28,
) -> float:
    """
    Forecast future daily demand using the mean demand from the most recent 
    observed days.
    """

    if history.empty:
        raise ValueError("History cannot be empty.")

    if window <= 0:
        raise ValueError("Window must be greater than zero.")

    recent_history = history.tail(window)

    return float(recent_history.mean())