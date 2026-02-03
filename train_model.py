# train_model.py
# Location: repo root (same level as app.py, requirements.txt)

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

# Paths (relative to repo root)
DATA_PATH = Path("data/coral_reef_sites.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "region_model.joblib"

TARGET_COL = "region"
DROP_COLS = ["id", TARGET_COL]

CATEGORICAL_COLS = ["substrate_type", "light_availability", "marine_protection_status"]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    return df


def build_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
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
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATA_PATH}. Put the CSV at data/coral_reef_sites.csv"
        )

    df = load_dataset(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV must contain target column '{TARGET_COL}' for training.")

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]

    # Validate categorical columns exist
    missing = [c for c in CATEGORICAL_COLS if c not in X.columns]
    if missing:
        raise ValueError(f"CSV missing required categorical columns: {missing}")

    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # 80/10/10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    pipeline = build_pipeline(CATEGORICAL_COLS, numeric_cols)
    pipeline.fit(X_train, y_train)

    # Quick evaluation
    val_acc = accuracy_score(y_val, pipeline.predict(X_val))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")

    # Save everything needed for inference
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "expected_feature_columns": list(X.columns),  # raw columns expected at inference
    }
    joblib.dump(artifacts, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
