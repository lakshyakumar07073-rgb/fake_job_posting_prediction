import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake Job Detector 🔍")

text = st.text_area("Enter Job Description")

if st.button("Predict"):
    vec = vectorizer.transform([text])
    result = model.predict(vec)

    if result[0] == 1:
        st.error("Fake Job 🚨")
    else:
        st.success("Real Job ✅")