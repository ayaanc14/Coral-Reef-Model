# app.py
# Location: repo root (same level as train_model.py, requirements.txt)

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/region_model.joblib")

st.set_page_config(page_title="Coral Reef Region Predictor", layout="wide")
st.title("Coral Reef Region Predictor")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. Run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)

def align_features(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Make uploaded data match training-time raw feature columns:
    - add missing cols as NA
    - drop extra cols
    - reorder columns
    """
    df = df.copy()
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df.drop(columns=[c for c in df.columns if c not in expected_cols])
    return df[expected_cols]

artifacts = load_artifacts()
pipeline = artifacts["pipeline"]
label_encoder = artifacts["label_encoder"]
expected_cols = artifacts["expected_feature_columns"]

st.write("Upload a CSV **without** the `region` column (features only).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(df.head(20), use_container_width=True)

    X = align_features(df, expected_cols)
    y_pred = pipeline.predict(X)
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
