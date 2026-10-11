from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error

from retail_demand.modeling import FEATURE_COLUMNS


def train_hurdle_model(
    train: pd.DataFrame,
) -> tuple[
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
]:
    """
    Train a two-stage hurdle model for intermittent demand.

    Stage 1 predicts whether demand will occur.
    Stage 2 predicts quantity when demand is positive.
    """

    features = train[FEATURE_COLUMNS]

    demand_occurs = (
        train["UnitsSold"] > 0
    ).astype(int)

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        random_state=42,
    )

    classifier.fit(
        features,
        demand_occurs,
    )

    positive_train = train.loc[
        train["UnitsSold"] > 0
    ].copy()

    if positive_train.empty:
        raise ValueError(
            "Training data must contain positive demand."
        )

    quantity_model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        random_state=42,
    )

    quantity_model.fit(
        positive_train[FEATURE_COLUMNS],
        positive_train["UnitsSold"],
    )

    return classifier, quantity_model


def predict_hurdle_demand(
    classifier: HistGradientBoostingClassifier,
    quantity_model: HistGradientBoostingRegressor,
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Generate expected daily demand from the two hurdle stages.
    """

    features = data[FEATURE_COLUMNS]

    demand_probability = classifier.predict_proba(
        features
    )[:, 1]

    positive_quantity = quantity_model.predict(
        features
    )

    positive_quantity = np.clip(
        positive_quantity,
        a_min=0,
        a_max=None,
    )

    predictions = (
        demand_probability
        * positive_quantity
    )

    return predictions


def evaluate_hurdle_model(
    classifier: HistGradientBoostingClassifier,
    quantity_model: HistGradientBoostingRegressor,
    test: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate hurdle forecasts against the same trailing-mean baseline.
    """

    predictions = predict_hurdle_demand(
        classifier,
        quantity_model,
        test,
    )

    baseline_predictions = test[
        "rolling_mean_28"
    ]

    hurdle_mae = mean_absolute_error(
        test["UnitsSold"],
        predictions,
    )

    baseline_mae = mean_absolute_error(
        test["UnitsSold"],
        baseline_predictions,
    )

    actual_total = test["UnitsSold"].sum()

    hurdle_absolute_error = (
        test["UnitsSold"] - predictions
    ).abs().sum()

    baseline_absolute_error = (
        test["UnitsSold"]
        - baseline_predictions
    ).abs().sum()

    hurdle_wape = (
        hurdle_absolute_error / actual_total
        if actual_total > 0
        else float("nan")
    )

    baseline_wape = (
        baseline_absolute_error / actual_total
        if actual_total > 0
        else float("nan")
    )

    return {
        "hurdle_mae": float(hurdle_mae),
        "baseline_mae": float(baseline_mae),
        "hurdle_wape": float(hurdle_wape),
        "baseline_wape": float(baseline_wape),
        "prediction_mean": float(
            predictions.mean()
        ),
        "prediction_max": float(
            predictions.max()
        ),
        "actual_mean": float(
            test["UnitsSold"].mean()
        ),
    }