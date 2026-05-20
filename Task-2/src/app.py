from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

model = tf.keras.models.load_model(BASE_DIR / "models" / "cnn_model.h5")

st.title("Dogs vs Cats CNN Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    img = img.resize((128,128))

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Prediction: Dog")
    else:
        st.success("Prediction: Cat")