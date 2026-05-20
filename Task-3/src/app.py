import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

st.title("AI Plagiarism Detection Tool")

text1 = st.text_area("Enter First Text")
text2 = st.text_area("Enter Second Text")

if st.button("Check"):

    vectorizer = TfidfVectorizer(ngram_range=(1,2))

    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    cosine_score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    fuzzy_score = fuzz.ratio(text1, text2)

    st.write(f"Cosine Similarity: {round(float(cosine_score), 2)}")
    st.write(f"Fuzzy Score: {fuzzy_score}")

    if cosine_score > 0.5 or fuzzy_score > 70:
        st.error("Potential Plagiarism Detected")
    else:
        st.success("No Significant Plagiarism Found")