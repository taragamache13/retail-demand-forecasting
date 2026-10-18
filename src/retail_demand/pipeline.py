from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_demand.artifacts import load_metadata
from retail_demand.forecast import generate_next_day_forecasts
from retail_demand.prepare_data import prepare_dataset
from retail_demand.train import train_production_system


MODEL_PATH = Path(
    "artifacts/poisson_model.joblib"
)

SEGMENTS_PATH = Path(
    "artifacts/product_segments.parquet"
)

METADATA_PATH = Path(
    "artifacts/metadata.json"
)


def validate_production_artifacts(
    model_path: Path = MODEL_PATH,
    segments_path: Path = SEGMENTS_PATH,
    metadata_path: Path = METADATA_PATH,
) -> None:
    """
    Validate that production artifacts exist and agree with metadata.
    """

    required_paths = [
        model_path,
        segments_path,
        metadata_path,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"Missing production artifacts: {missing_paths}"
        )

    metadata = load_metadata(
        metadata_path
    )

    segments = pd.read_parquet(
        segments_path
    )

    expected_products = metadata[
        "active_products"
    ]

    actual_products = len(segments)

    if actual_products != expected_products:
        raise ValueError(
            "Artifact validation failed: "
            f"metadata expects {expected_products:,} "
            f"active products but segment artifact "
            f"contains {actual_products:,}."
        )

    print("ARTIFACT VALIDATION PASSED")
    print(
        f"Validated products: "
        f"{actual_products:,}"
    )


def run_production_pipeline() -> pd.DataFrame:
    """
    Run the full retail forecasting production pipeline.

    Steps:
    1. Prepare raw data.
    2. Train production forecasting system.
    3. Validate saved artifacts.
    4. Generate next-day forecasts.
    """

    print("STEP 1: PREPARING DATA")
    prepare_dataset()

    print("\nSTEP 2: TRAINING PRODUCTION SYSTEM")
    train_production_system()

    print("\nSTEP 3: VALIDATING ARTIFACTS")
    validate_production_artifacts()

    print("\nSTEP 4: GENERATING FORECASTS")
    forecasts = generate_next_day_forecasts()

    print("\nPRODUCTION PIPELINE COMPLETE")

    return forecasts


if __name__ == "__main__":
    run_production_pipeline()