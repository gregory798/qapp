import streamlit as st
import requests

st.title("Question Answering App")

model_name = st.selectbox("Choisir un modèle :", ["bert", "albert", "distilbert"])
context = st.text_area("Contexte (ou charger un fichier ci-dessous)")
question = st.text_input("Question")
file = st.file_uploader("Charger un fichier texte", type=["txt"])

if file:
    context = file.read().decode("utf-8")

if st.button("Obtenir la réponse") and context and question:
    res = requests.post("http://localhost:8000/predict", json={
        "question": question,
        "context": context,
        "model_name": model_name
    })
    st.subheader("Réponse :")
    st.write(res.json()["answer"])