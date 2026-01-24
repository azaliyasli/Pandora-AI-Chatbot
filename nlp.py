import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

# Data Preparation
df = pd.read_csv("combined_emotion.csv")
df = df.groupby('emotion').apply(lambda x: x.sample(5000)).reset_index(drop=True)

X = df["sentence"].values
y = df["emotion"].values

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_data(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

train_enc = tokenize_data(X_train)
test_enc = tokenize_data(X_test)

# Dataset Preparation
train_dataset = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(y_train))
test_dataset = TensorDataset(test_enc["input_ids"], test_enc["attention_mask"], torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in loop:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "distilbert_pandora.pt") # save model for RAG and FastAPI

# Prediction
model.eval()
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        y_pred.extend(preds.numpy())

# Evaluation
print(classification_report(y_test, y_pred, target_names=le.classes_))