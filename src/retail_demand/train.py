from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from retail_demand.aggregation import extend_products_to_end_date
from retail_demand.artifacts import (
    save_metadata,
    save_model_artifact,
)
from retail_demand.features import create_demand_features
from retail_demand.model import train_demand_model
from retail_demand.modeling import FEATURE_COLUMNS
from retail_demand.segmentation import calculate_demand_segments
from retail_demand.selection import (
    calculate_product_metrics,
    select_active_products,
    select_eligible_products,
)


DAILY_DEMAND_PATH = Path(
    "data/processed/daily_product_demand.parquet"
)

ARTIFACT_DIR = Path("artifacts")

MODEL_PATH = ARTIFACT_DIR / "poisson_model.joblib"
SEGMENTS_PATH = ARTIFACT_DIR / "product_segments.parquet"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

ACTIVITY_DAYS = 28
CROSTON_ALPHA = 0.1


def train_production_system() -> None:
    """
    Train the production forecasting system using all available data.

    The production system uses:
    - Poisson gradient boosting for lumpy, erratic, and smooth demand.
    - Croston/SBA for intermittent demand.
    """

    demand = pd.read_parquet(
        DAILY_DEMAND_PATH
    )

    demand["Date"] = pd.to_datetime(
        demand["Date"],
        errors="raise",
    )

    final_date = demand["Date"].max()

    product_metrics = calculate_product_metrics(
        demand
    )

    eligible = select_eligible_products(
        product_metrics
    )

    active_stock_codes = select_active_products(
        daily_demand=demand,
        eligible_stock_codes=eligible[
            "StockCode"
        ].tolist(),
        cutoff_date=final_date,
        activity_days=ACTIVITY_DAYS,
    )

    production_demand = extend_products_to_end_date(
        daily_demand=demand,
        stock_codes=active_stock_codes,
        end_date=final_date,
    )

    segments = calculate_demand_segments(
        production_demand
    )

    features = create_demand_features(
        production_demand
    )

    training_data = features.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    model = train_demand_model(
        training_data
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model_artifact(
        model,
        MODEL_PATH,
    )

    segments.to_parquet(
        SEGMENTS_PATH,
        index=False,
    )

    metadata = {
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_through": final_date.isoformat(),
        "eligible_products": len(eligible),
        "active_products": len(active_stock_codes),
        "training_rows": len(training_data),
        "feature_columns": FEATURE_COLUMNS,
        "activity_days": ACTIVITY_DAYS,
        "croston_alpha": CROSTON_ALPHA,
        "routing": {
            "intermittent": "croston_sba",
            "lumpy": "poisson_gradient_boosting",
            "erratic": "poisson_gradient_boosting",
            "smooth": "poisson_gradient_boosting",
        },
    }

    save_metadata(
        metadata,
        METADATA_PATH,
    )

    print("PRODUCTION TRAINING COMPLETE")
    print(f"Data through: {final_date.date()}")
    print(f"Eligible products: {len(eligible):,}")
    print(
        f"Active products: "
        f"{len(active_stock_codes):,}"
    )
    print(
        f"Training rows: "
        f"{len(training_data):,}"
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(
        f"Segments saved to: "
        f"{SEGMENTS_PATH}"
    )
    print(
        f"Metadata saved to: "
        f"{METADATA_PATH}"
    )


if __name__ == "__main__":
    train_production_system()