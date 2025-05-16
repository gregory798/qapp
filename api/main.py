from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = FastAPI()

# Charger tous les modèles
models = {
    "bert": {
        "model": AutoModelForQuestionAnswering.from_pretrained("api/models/bert-base-uncased-finetuned"),
        "tokenizer": AutoTokenizer.from_pretrained("api/models/bert-base-uncased-finetuned")
    },
    "albert": {
        "model": AutoModelForQuestionAnswering.from_pretrained("api/models/albert-base-v2-finetuned"),
        "tokenizer": AutoTokenizer.from_pretrained("api/models/albert-base-v2-finetuned")
    },
    "distilbert": {
        "model": AutoModelForQuestionAnswering.from_pretrained("api/models/distilbert-base-uncased-finetuned"),
        "tokenizer": AutoTokenizer.from_pretrained("api/models/distilbert-base-uncased-finetuned")
    }
}

class QARequest(BaseModel):
    question: str
    context: str
    model_name: str

@app.post("/predict")
def predict(request: QARequest):
    m = models[request.model_name]
    inputs = m["tokenizer"](request.question, request.context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = m["model"](**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = m["tokenizer"].decode(inputs["input_ids"][0][start:end])
    return {"answer": answer}