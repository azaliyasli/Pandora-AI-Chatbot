import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertTokenizer
import torch

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