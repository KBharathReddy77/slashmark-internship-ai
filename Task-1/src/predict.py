
from pathlib import Path
import joblib

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "sentiment_model.pkl"

model = joblib.load(model_path)

text = input("Enter text: ")

clean_text = preprocess_text(text)

prediction = model.predict([clean_text])

print("Predicted Sentiment:", prediction[0])
