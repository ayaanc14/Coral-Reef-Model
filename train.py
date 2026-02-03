from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.config import DATA_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE
from src.data import load_dataset


TARGET_COL = "region"
DROP_COLS = ["id", TARGET_COL]


@dataclass
class Artifacts:
    pipeline: Pipeline
    label_encoder: LabelEncoder
    feature_columns: List[str]  # original raw feature columns expected (before one-hot)


def build_pipeline(categorical_cols: List[str], numeric_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]

    # Label encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train/Val/Test split: 80/10/10
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put coral_reef_sites.csv in /data."
        )

    df = load_dataset(DATA_PATH)

    # Define categorical columns based on your notebook
    categorical_cols = ["substrate_type", "light_availability", "marine_protection_status"]

    # Everything else (except target/id) treated as numeric
    X_all = df.drop(columns=DROP_COLS)
    missing_cats = [c for c in categorical_cols if c not in X_all.columns]
    if missing_cats:
        raise ValueError(f"Expected categorical columns missing from CSV: {missing_cats}")

    numeric_cols = [c for c in X_all.columns if c not in categorical_cols]

    X_train, X_val, X_test, y_train, y_val, y_test, le = split_data(df)

    pipeline = build_pipeline(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    pipeline.fit(X_train, y_train)

    # Evaluate on val/test
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    # Optional detailed report (decoded)
    print("\nClassification report (test):")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=le.classes_
    ))

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = Artifacts(
        pipeline=pipeline,
        label_encoder=le,
        feature_columns=list(X_all.columns)  # raw input features the app expects
    )
    joblib.dump(artifacts, MODEL_PATH)
    print(f"\nSaved model artifacts to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
