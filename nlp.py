import pandas as pd
import json

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW

# Data Preparation
with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

all_patterns = []
all_tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
        all_tags.append(intent['tag'])

df = pd.DataFrame({
    'patterns': all_patterns,
    'tag': all_tags
})

X = df["patterns"]
y = df["tag"]

# Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

X_train_tokens = tokenizer(
    list(X_train),
    padding=True,
    truncation=True,
    max_length=100,
    return_tensors="pt"
)
X_test_tokens = tokenizer(
    list(X_test),
    padding=True,
    truncation=True,
    max_length=100,
    return_tensors="pt"
)

y_train_tensor = torch.tensor(y_train.values)
y_test_tensor = torch.tensor(y_test.values)

# Prepare Data Sets
train_dataset = TensorDataset(
    X_train_tokens["input_ids"],
    X_train_tokens["attention_mask"],
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tokens["input_ids"],
    X_test_tokens["attention_mask"],
    y_test_tensor
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training
device = torch.device("cpu")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 10
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
) # Slow the learning process down for higher accuracy

for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(
            b_input_ids,
            attention_mask=b_input_mask,
            labels=b_labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_train_loss:.4f}")

torch.save(model.state_dict(), "distilbert_pandora.pt") # For RAG and FastAPI

# Prediction
y_true = []
y_pred = []

model.eval()

with torch.no_grad():
    for batch in test_loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()

        y_true.extend(b_labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
print(classification_report(y_test, y_pred))