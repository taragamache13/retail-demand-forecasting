from __future__ import annotations

import pandas as pd


def croston_sba_forecast(
    history: pd.Series,
    alpha: float = 0.1,
) -> float:
    """
    Forecast intermittent demand using Croston's method with the
    Syntetos-Boylan Approximation (SBA) bias correction.

    Parameters
    ----------
    history:
        Historical daily demand values.

    alpha:
        Smoothing parameter between 0 and 1.

    Returns
    -------
    float
        Expected daily demand.
    """

    if history.empty:
        raise ValueError("History cannot be empty.")

    if not 0 < alpha <= 1:
        raise ValueError(
            "alpha must be greater than 0 and less than or equal to 1."
        )

    demand = history.astype(float)

    positive_positions = demand[demand > 0].index

    if len(positive_positions) == 0:
        return 0.0

    first_positive_position = next(
        position
        for position, value in enumerate(demand)
        if value > 0
    )

    demand_size = demand.iloc[
        first_positive_position
    ]

    demand_interval = float(
        first_positive_position + 1
    )

    interval_counter = 1.0

    for value in demand.iloc[
        first_positive_position + 1:
    ]:
        if value > 0:
            demand_size = (
                demand_size
                + alpha * (value - demand_size)
            )

            demand_interval = (
                demand_interval
                + alpha
                * (
                    interval_counter
                    - demand_interval
                )
            )

            interval_counter = 1.0

        else:
            interval_counter += 1.0

    croston_forecast = (
        demand_size / demand_interval
    )

    sba_correction = (
        1 - alpha / 2
    )

    return float(
        sba_correction * croston_forecast
    )