from __future__ import annotations

from dataclasses import asdict
from typing import List

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.train import Artifacts


def load_artifacts() -> Artifacts:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m src.train"
        )
    artifacts: Artifacts = joblib.load(MODEL_PATH)
    return artifacts


def validate_input_df(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Ensures df contains exactly the expected feature columns (raw, pre-onehot).
    - Adds any missing columns as NA
    - Drops extra columns
    - Reorders to expected order
    """
    df = df.copy()

    for col in expected_features:
        if col not in df.columns:
            df[col] = pd.NA

    extra_cols = [c for c in df.columns if c not in expected_features]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    df = df[expected_features]
    return df


def predict_regions(df_features: pd.DataFrame) -> pd.Series:
    artifacts = load_artifacts()
    df_features = validate_input_df(df_features, artifacts.feature_columns)

    y_pred = artifacts.pipeline.predict(df_features)
    regions = artifacts.label_encoder.inverse_transform(y_pred)
    return pd.Series(regions, name="predicted_region")
