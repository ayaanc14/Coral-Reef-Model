# app.py
# Location: repo root

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/region_model.joblib")

st.set_page_config(page_title="Coral Reef Region Predictor", layout="wide")
st.title("Coral Reef Region Predictor")

# Must match training
CATEGORICAL_COLS = ["substrate_type", "light_availability", "marine_protection_status"]

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. Train it first with: python3 train_model.py"
        )
    return joblib.load(MODEL_PATH)

def align_features(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Make uploaded data match training-time raw feature columns:
    - add missing cols as np.nan (NOT pd.NA)
    - drop extra cols
    - reorder columns
    - coerce numeric columns to numeric (bad values -> np.nan)
    """
    df = df.copy()

    # Add missing columns using np.nan so sklearn can handle it
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Drop extra columns
    df = df.drop(columns=[c for c in df.columns if c not in expected_cols])

    # Reorder
    df = df[expected_cols]

    # Replace any pandas NA types with np.nan
    df = df.replace({pd.NA: np.nan})

    # Coerce numeric columns to numeric (everything except known categoricals)
    numeric_cols = [c for c in expected_cols if c not in CATEGORICAL_COLS]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure categoricals are strings or nan
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
            df[c] = df[c].replace({"<NA>": np.nan})

    return df

artifacts = load_artifacts()
pipeline = artifacts["pipeline"]
label_encoder = artifacts["label_encoder"]
expected_cols = artifacts["expected_feature_columns"]

st.write("Upload a CSV of features (**no `region` column**).")

with st.expander("CSV columns"):
    st.write("Your CSV should contain these feature columns (order doesn’t matter):")
    st.code(", ".join(expected_cols))
    template_df = pd.DataFrame(columns=expected_cols)
    st.download_button(
        "Download CSV template (headers only)",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="coral_reef_input_template.csv",
        mime="text/csv",
    )

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the CSV. Error: {e}")
        st.stop()

    if df.shape[0] == 0:
        st.error("Your CSV has **0 rows**. Add at least one data row and re-upload.")
        st.stop()

    st.write("Preview:")
    st.dataframe(df.head(20), use_container_width=True)

    X = align_features(df, expected_cols)

    # Safety checks
    if X.shape[0] == 0:
        st.error("After aligning columns, there are 0 rows to predict on.")
        st.stop()

    try:
        y_pred = pipeline.predict(X)
    except Exception as e:
        st.error(f"Prediction failed.\n\nFull error:\n{e}")
        st.stop()

    regions = label_encoder.inverse_transform(y_pred)

    out = df.copy()
    out["predicted_region"] = regions

    st.success("Done!")
    st.dataframe(out.head(50), use_container_width=True)

    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="coral_reef_predictions.csv",
        mime="text/csv",
    )
