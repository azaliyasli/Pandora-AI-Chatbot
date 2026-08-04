import torch
import json
import joblib
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

le = joblib.load("label_encoder.pkl")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)
model.load_state_dict(torch.load("distilbert_pandora.pt", map_location=device))
model.to(device)
model.eval()

embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

corpus_queries = []
corpus_hints = []

with open('rag_emotion_dataset.json', 'r', encoding='utf-8') as f:
    rag_data = json.load(f)
    for category in rag_data:
        for doc in category['documents']:
            corpus_queries.append(doc['query'])
            corpus_hints.append(doc['response_hint'])

corpus_embeddings = embedder.encode(corpus_queries, convert_to_tensor=True)


def get_final_context(user_text):

    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    pred_idx = torch.argmax(outputs.logits, dim=1).item()
    detected_emotion = le.inverse_transform([pred_idx])[0]

    query_embedding = embedder.encode(user_text, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    top_results = torch.topk(cos_scores, k=3)

    retrieved_context = "Similar past user expressions from database:\n"
    for idx in top_results[1]:
        retrieved_context += f"- {corpus_queries[idx]}\n"

    best_hint = corpus_hints[top_results[1][0]]

    return detected_emotion, best_hint, retrieved_context