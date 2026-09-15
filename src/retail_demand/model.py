from __future__ import annotations

import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from retail_demand.modeling import FEATURE_COLUMNS

def train_demand_model(
        train: pd.DataFrame,
) -> HistGradientBoostingRegressor:
    """
    Train a gradient-boosting model to predict daily product demand.
    """

    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        random_state=42,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["UnitsSold"],
    )

    return model

def evaluate_demand_model(
        model: HistGradientBoostingRegressor,
        test: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate an ML demand model and a matching 28-day-mean baseline.
    """

    predictions = model.predict(
        test[FEATURE_COLUMNS]
    )

    # Demand cannot be negative.
    predictions = predictions.clip(min=0)

    ml_mae = mean_absolute_error(
        test["UnitsSold"],
        predictions,
    )

    # rolling_mean_28 is already leakage-safe and represents the
    # trailing 28 day average available before each prediction date.
    baseline_predictions = test["rolling_mean_28"]

    baseline_mae = mean_absolute_error(
        test["UnitsSold"],
        baseline_predictions,
    )

    actual_total = test["UnitsSold"].sum()

    ml_absolute_error = (
        test["UnitsSold"] - predictions
    ).abs().sum()

    baseline_absolute_error = (
        test["UnitsSold"] - baseline_predictions
    ).abs().sum()

    ml_wape = (
        ml_absolute_error / actual_total
        if actual_total > 0
        else float("nan")
    )

    baseline_wape = (
        baseline_absolute_error / actual_total
        if actual_total > 0
        else float("nan")
    )

    return {
        "ml_mae": float(ml_mae),
        "baseline_mae": float(baseline_mae),
        "ml_wape": float(ml_wape),
        "baseline_wape": float(baseline_wape),
        "prediction_mean": float(predictions.mean()),
        "prediction_max": float(predictions.max()),
        "zero_prediction_rate": float(
            (predictions == 0).mean()
        ),
        "actual_mean": float(
            test["UnitsSold"].mean()
        ),
    }