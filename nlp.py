import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

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

# Preprocessing
le = LabelEncoder()
y = le.fit_transform(y)