import torch
import json
import random
import joblib
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load model and encoder
le = joblib.load("label_encoder.pkl")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)
model.load_state_dict(torch.load("distilbert_pandora.pt", map_location=device))
model.to(device)
model.eval()

# RAG Library
EMOTION_MAP = {
    "fear": "negative", "anger": "negative", "sad": "sad",
    "joy": "positive", "love": "positive", "suprise": "suprise"
}


def get_final_context(user_text):
    # NLP Prediction
    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    pred_idx = torch.argmax(outputs.logits, dim=1).item()
    detected_emotion = le.inverse_transform([pred_idx])[0]

    # Get RAG Data
    rag_tag = EMOTION_MAP.get(detected_emotion, "positive")
    with open('rag_emotion_dataset.json', 'r', encoding='utf-8') as f:
        rag_data = json.load(f)

    for item in rag_data:
        if item['emotion'] == rag_tag:
            doc = random.choice(item['documents'])
            return detected_emotion, doc['response_hint']

    return detected_emotion, "Provide support emphatically."

# Test
#emotion, hint = get_final_context("I caught my friend is lying to me.")
#print(f"NLP: {emotion}, RAG: {hint}")