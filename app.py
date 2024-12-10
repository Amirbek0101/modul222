import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

st.header("Sentiment")
page_description = """Bu model so'zlarni sentimentga ajratadi"""
st.markdown(page_description)

matn = st.text_input("Ingliz tilida so'z kiriting:")

# Modelni yuklash
with open("/Users/amir/Desktop/Suniy2/tweets_sentiment (1).pkl", "rb") as fl:
    pr = pickle.load(fl)

# Matnli ma'lumotlar
data = {"text": [matn]}


# Pandas DataFrame
df = pd.DataFrame(data)

# TF-IDF Vectorizer obyekti
tfidf_vectorizer = TfidfVectorizer()

# Matnni vektorizatsiya qilish
tfidf_vectors = tfidf_vectorizer.fit_transform(df['text'])


if st.button("Tekshirish"):
    if matn:  # Matn kiritilganligini tekshirish
        sentiment_natija = pr.predict(tfidf_vectors)
        if sentiment_natija == 'negative':
            st.write("Negativ!")
        elif sentiment_natija == 'positive':
            st.write("Positive!")
        else:
            st.write('Neutral!')
    else:
        st.warning("Iltimos, matn kiriting.")




