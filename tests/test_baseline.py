import pandas as pd

from retail_demand.baseline import trailing_mean_forecast

def test_trailing_mean_forecast_uses_recent_history():
    history = pd.Series(
        [1,2,3,4,5,6]
    )

    prediction = trailing_mean_forecast(
        history,
        window=3,
    )

    assert prediction == 5.0