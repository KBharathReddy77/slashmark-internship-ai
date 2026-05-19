
from pathlib import Path
import streamlit as st
import joblib

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "sentiment_model.pkl"

st.title("Sentiment Analysis AI Tool")

model = joblib.load(model_path)

user_input = st.text_area("Enter your text")

if st.button("Analyze Sentiment"):
    clean_text = preprocess_text(user_input)
    prediction = model.predict([clean_text])

    st.success(f"Predicted Sentiment: {prediction[0]}")
