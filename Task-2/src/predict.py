from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

model = tf.keras.models.load_model(BASE_DIR / "models" / "cnn_model.h5")

img_path = input("Enter image path: ")

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Dog")
else:
    print("Cat")