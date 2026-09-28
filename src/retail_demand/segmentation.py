from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "StockCode",
    "UnitsSold",
}


def calculate_demand_segments(
    daily_demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate demand-pattern metrics and segment each product.

    Segments are based on:
    - ADI: average demand interval
    - CV²: squared coefficient of variation of nonzero demand
    """

    missing_columns = REQUIRED_COLUMNS.difference(
        daily_demand.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    records = []

    for stock_code, product_data in daily_demand.groupby("StockCode"):
        positive_demand = product_data.loc[
            product_data["UnitsSold"] > 0,
            "UnitsSold",
        ]

        history_days = len(product_data)
        active_days = len(positive_demand)

        if active_days == 0:
            adi = np.inf
            cv_squared = np.nan
            segment = "insufficient"

        else:
            adi = history_days / active_days

            if active_days < 2:
                cv_squared = np.nan
                segment = "insufficient"

            else:
                mean_positive = positive_demand.mean()
                std_positive = positive_demand.std()

                cv_squared = (
                    std_positive / mean_positive
                ) ** 2

                if adi < 1.32 and cv_squared < 0.49:
                    segment = "smooth"

                elif adi >= 1.32 and cv_squared < 0.49:
                    segment = "intermittent"

                elif adi < 1.32 and cv_squared >= 0.49:
                    segment = "erratic"

                else:
                    segment = "lumpy"

        records.append(
            {
                "StockCode": stock_code,
                "ADI": adi,
                "CV2": cv_squared,
                "DemandSegment": segment,
            }
        )

    return pd.DataFrame(records)