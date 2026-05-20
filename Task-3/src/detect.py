from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent

dataset_path = BASE_DIR / "data" / "plagiarism_dataset.csv"
report_path = BASE_DIR / "reports" / "plagiarism_report.csv"

df = pd.read_csv(dataset_path)

results = []

for _, row in df.iterrows():

    text1 = clean_text(row['text1'])
    text2 = clean_text(row['text2'])

    vectorizer = TfidfVectorizer(ngram_range=(1,2))

    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    cosine_score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    fuzzy_score = fuzz.ratio(text1, text2)

    prediction = "Plagiarized" if cosine_score > 0.5 or fuzzy_score > 70 else "Not Plagiarized"

    results.append({
        "Text1": row['text1'],
        "Text2": row['text2'],
        "Cosine Similarity": round(float(cosine_score), 2),
        "Fuzzy Score": fuzzy_score,
        "Prediction": prediction
    })

report_df = pd.DataFrame(results)

report_df.to_csv(report_path, index=False)

print(report_df)

print("\nReport generated successfully!")