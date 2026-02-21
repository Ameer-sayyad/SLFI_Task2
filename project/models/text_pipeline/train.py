# ==============================
# TEXT PIPELINE TRAIN
# BERT + Linear Classifier
# ==============================

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# ------------------------------
# Paths
# ------------------------------
train_base = "/content/TESS_SPLIT/train"

# ------------------------------
# Emotion mapping
# ------------------------------
emotion_map = {
    "angry":0,
    "disgust":1,
    "fear":2,
    "happy":3,
    "ps":4,
    "sad":5,
    "neutral":6
}

# ------------------------------
# Load train paths
# ------------------------------
train_paths, train_labels = [], []

for emo in os.listdir(train_base):
    emo_path = os.path.join(train_base, emo)
    for file in os.listdir(emo_path):
        train_paths.append(os.path.join(emo_path, file))
        train_labels.append(emotion_map[emo])

print("Train samples:", len(train_paths))

# ------------------------------
# Build text from filename
# ------------------------------
def build_text(path):
    file = path.split("/")[-1]
    word = file.split("_")[1]
    sentence = f"say the word {word}"
    return sentence

train_texts = [build_text(p) for p in train_paths]

# ------------------------------
# Device
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load BERT
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

# Freeze BERT (feature extractor only)
for param in bert_model.parameters():
    param.requires_grad = False

# ------------------------------
# Text Classifier
# ------------------------------
class TextEmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 7)

    def forward(self, x):
        return self.fc(x)

text_model = TextEmotionClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(text_model.parameters(), lr=2e-5)

epochs = 30

# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(epochs):

    text_model.train()
    total_loss = 0

    for text, label in zip(train_texts, train_labels):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=16
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        cls_embedding = outputs.last_hidden_state[:,0,:]  # [CLS]

        label_tensor = torch.tensor([label]).to(device)

        preds = text_model(cls_embedding)
        loss = criterion(preds, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss)

# ------------------------------
# Save model
# ------------------------------
torch.save(text_model.state_dict(),
           "/content/New_Project_pipeline/text_model.pt")

print("Text model training complete.")
