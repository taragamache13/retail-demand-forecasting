from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_model_artifact(
    model: Any,
    model_path: Path,
) -> None:
    """
    Save a trained model to disk.
    """

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )


def load_model_artifact(
    model_path: Path,
) -> Any:
    """
    Load a previously saved model from disk.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


def save_metadata(
    metadata: dict,
    metadata_path: Path,
) -> None:
    """
    Save model metadata as JSON.
    """

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def load_metadata(
    metadata_path: Path,
) -> dict:
    """
    Load model metadata from JSON.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)