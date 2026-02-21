import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def init():
    global tokenizer, model
    model_path = "./models/spam_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    text = data["text"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits).item()

    return {"prediction": int(prediction)}
