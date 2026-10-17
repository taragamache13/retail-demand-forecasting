import pandas as pd
import pytest

from retail_demand.croston import (
    croston_sba_forecast,
)


def test_croston_sba_returns_zero_for_zero_history():
    history = pd.Series(
        [0, 0, 0, 0, 0]
    )

    forecast = croston_sba_forecast(
        history
    )

    assert forecast == 0.0


def test_croston_sba_forecasts_sparse_demand():
    history = pd.Series(
        [
            5,
            0,
            0,
            5,
            0,
            0,
            5,
        ]
    )

    forecast = croston_sba_forecast(
        history,
        alpha=0.2,
    )

    assert forecast == pytest.approx(
        2.6163,
        abs=0.0001,
    )