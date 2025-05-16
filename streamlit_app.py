import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os

st.title("🧠 Question Answering App")

# Choix du modèle
model_name = st.selectbox("Choisir un modèle :", [
    "bert-base-uncased-finetuned",
    "albert-base-v2-finetuned",
    "distilbert-base-uncased-finetuned"
])

# Charger le modèle et le tokenizer
@st.cache_resource
def load_model(model_name):
    model_path = os.path.join("models", model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model(model_name)

# Contexte
uploaded_file = st.file_uploader("📄 Charger un fichier .txt", type=["txt"])
if uploaded_file:
    context = uploaded_file.read().decode("utf-8")
else:
    context = st.text_area("✍️ Écris ou colle un contexte ici")

# Question
question = st.text_input("❓ Ta question")

# Prédiction
if st.button("🎯 Obtenir la réponse") and context and question:
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)

    if answer.strip().lower() in ["", "[cls]"]:
        st.warning("😕 Je n’ai pas trouvé de réponse.")
    else:
        st.success("✅ Réponse :")
        st.markdown(f"**{answer}**")