import streamlit as st
import pickle

st.header("Sentiment")
page_description = """Bu model so'zlarni sentimentga ajratadi"""
st.markdown(page_description)

matn = st.text_input("Ingliz tilida so'z kiriting:")

# `pr` o'zgaruvchisini dastlabki qiymat bilan belgilash
pr = None

# Modelni yuklashga harakat qilish
try:
    with open("C:\\Modul 2 suniy\\tweets_sentiment.pkl", "rb") as fl:
        pr = pickle.load(fl)
except Exception as e:
    st.error(f"Modelni yuklashda xato: {e}")

if st.button("Tekshirish"):
    if pr is not None:  # Model yuklanganligini tekshirish
        if matn:
            sentiment_natija = pr.predict([[matn]])[0]  # Natijani olish
            if sentiment_natija == 'negative':
                st.write("**Natija:** Negativ!")
            elif sentiment_natija == 'positive':
                st.write("**Natija:** Positive!")
            else:
                st.write("**Natija:** Neutral!")
        else:
            st.warning("Iltimos, matn kiriting.")
    else:
        st.warning("Model yuklanmagan, iltimos, qayta tekshiring.")