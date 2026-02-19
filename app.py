import streamlit as st
from predict import predict_news

st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check if it is Real or Fake.")

user_input = st.text_area("Paste News Article Here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_news(user_input)
        if result == "Fake News":
            st.error("🚨 This is Fake News")
        else:
            st.success("✅ This is Real News")
