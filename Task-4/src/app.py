from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from config import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, BASE_FEATURES
from data_loading import find_dataset_file

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("AI Fraud Detection Dashboard")

st.write("Upload a CSV file with the same columns as the Kaggle dataset, or use the demo sample.")

@st.cache_resource
def load_artifacts():
    model = joblib.load(DEFAULT_MODEL_PATH)
    meta = json.loads(DEFAULT_META_PATH.read_text(encoding="utf-8"))
    return model, meta

model, meta = load_artifacts()
threshold = float(meta["threshold"])

uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    demo = Path("data/demo/sample_transactions.csv")
    if demo.exists():
        df = pd.read_csv(demo)
    else:
        st.info("No file uploaded. Place the Kaggle dataset in data/raw/creditcard.csv to use the pipeline.")
        st.stop()

missing = set(BASE_FEATURES) - set(df.columns)
if missing:
    st.error(f"Missing columns: {sorted(missing)}")
    st.stop()

probs = model.predict_proba(df[BASE_FEATURES])[:, 1]
preds = (probs >= threshold).astype(int)

result = df.copy()
result["fraud_probability"] = probs
result["fraud_prediction"] = preds

st.subheader("Predictions")
st.dataframe(result.head(50), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(result))
col2.metric("Fraud flagged", int(result["fraud_prediction"].sum()))
col3.metric("Threshold", f"{threshold:.4f}")

st.download_button(
    "Download predictions as CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="fraud_predictions.csv",
    mime="text/csv",
)
