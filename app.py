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
            f"Missing model at {MODEL_PATH}. Train it first with: python3 train_model.py"
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

st.write("Upload a CSV of features (**no `region` column**).")

with st.expander("What columns should my CSV have?"):
    st.write("Your CSV should contain these feature columns (order doesn’t matter):")
    st.code(", ".join(expected_cols))

    # Provide a downloadable empty template with correct headers
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

    # ✅ Critical guard: stop if CSV has headers but no rows
    if df.shape[0] == 0:
        st.error(
            "Your CSV was read successfully, but it contains **0 rows**.\n\n"
            "This usually happens if you uploaded a template/header-only file, or the file has no data.\n\n"
            "Add at least **one observation row** and re-upload."
        )
        st.stop()

    st.write("Preview:")
    st.dataframe(df.head(20), use_container_width=True)

    X = align_features(df, expected_cols)

    # Another guard: if somehow rows got lost (shouldn't happen), stop safely
    if X.shape[0] == 0:
        st.error("After aligning columns, there are 0 rows to predict on. Please upload a CSV with data rows.")
        st.stop()

    try:
        y_pred = pipeline.predict(X)
    except Exception as e:
        st.error(
            "Prediction failed. Common causes:\n"
            "- Wrong delimiter/format (e.g. semicolon-separated file)\n"
            "- Columns not matching expected names\n"
            "- Non-CSV file uploaded\n\n"
            f"Full error:\n{e}"
        )
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
