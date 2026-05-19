
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent

dataset_path = BASE_DIR / "data" / "twitter_sentiment_dataset.csv"
model_path = BASE_DIR / "models" / "sentiment_model.pkl"

df = pd.read_csv(dataset_path)

df["clean_text"] = df["text"].apply(preprocess_text)

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\\nModel Evaluation")
print("-" * 40)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

print("\\nClassification Report:\\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, model_path)

print("\\nModel saved successfully!")
