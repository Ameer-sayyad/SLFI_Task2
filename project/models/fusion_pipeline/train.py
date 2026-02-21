# ==========================================
# FUSION PIPELINE TRAIN
# Speech (HuBERT + BiLSTM) + Text (BERT)
# ==========================================

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# ------------------------------
# PATHS
# ------------------------------
train_base = "/content/TESS_SPLIT/train"
hubert_train_dir = "/content/New_Project_pipeline/hubert_train_embeddings"

# ------------------------------
# Emotion Mapping
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
# Load Train Data
# ------------------------------
train_paths, train_labels = [], []

for emo in os.listdir(train_base):
    emo_path = os.path.join(train_base, emo)
    for file in os.listdir(emo_path):
        train_paths.append(os.path.join(emo_path, file))
        train_labels.append(emotion_map[emo])

# Build Text
def build_text(path):
    file = path.split("/")[-1]
    word = file.split("_")[1]
    return f"say the word {word}"

train_texts = [build_text(p) for p in train_paths]

# ------------------------------
# Device
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load Speech Model (BiLSTM feature extractor)
# ------------------------------
class EmotionBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, 7)

    def forward(self, x, return_features=False):
        x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        if return_features:
            return pooled
        return self.fc(pooled)

speech_model = EmotionBiLSTM().to(device)
speech_model.load_state_dict(
    torch.load("/content/New_Project_pipeline/speech_model.pt",
               map_location=device)
)
speech_model.eval()

# ------------------------------
# Load BERT
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

for param in bert_model.parameters():
    param.requires_grad = False

# ------------------------------
# Load HuBERT embeddings
# ------------------------------
def load_train_embedding(path):
    file_name = path.split("/")[-1].replace(".wav", ".pt")
    full_path = os.path.join(hubert_train_dir, file_name)
    if os.path.exists(full_path):
        return torch.load(full_path)
    else:
        return None

# ------------------------------
# Fusion Model (NO ReLU)
# 1024 → 7
# ------------------------------
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 7)

    def forward(self, speech_emb, text_emb):
        fused = torch.cat((speech_emb, text_emb), dim=1)
        return self.fc(fused)

fusion_model = FusionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-4)

epochs = 20

# ------------------------------
# TRAINING LOOP
# ------------------------------
for epoch in range(epochs):

    fusion_model.train()
    total_loss = 0

    for path, text, label in zip(train_paths, train_texts, train_labels):

        speech_frames = load_train_embedding(path)
        if speech_frames is None:
            continue

        speech_frames = speech_frames.to(device)

        with torch.no_grad():
            speech_emb = speech_model(speech_frames, return_features=True)

        inputs = tokenizer(text,
                           return_tensors="pt",
                           truncation=True,
                           padding=True,
                           max_length=16).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        text_emb = outputs.last_hidden_state[:,0,:]

        label_tensor = torch.tensor([label]).to(device)

        preds = fusion_model(speech_emb, text_emb)
        loss = criterion(preds, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss)

# Save model
torch.save(fusion_model.state_dict(),
           "/content/New_Project_pipeline/fusion_model.pt")

print("Fusion training complete.")
