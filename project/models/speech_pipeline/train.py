# ==============================
# SPEECH PIPELINE TRAIN
# HuBERT + BiLSTM
# ==============================

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import librosa

# ------------------------------
# Paths
# ------------------------------
train_base = "/content/TESS_SPLIT/train"
test_base  = "/content/TESS_SPLIT/test"

hubert_train_dir = "/content/New_Project_pipeline/hubert_train_embeddings"
hubert_test_dir  = "/content/New_Project_pipeline/hubert_test_embeddings"

os.makedirs(hubert_train_dir, exist_ok=True)
os.makedirs(hubert_test_dir, exist_ok=True)

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
# Device
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load HuBERT
# ------------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)

# ------------------------------
# HuBERT extraction
# ------------------------------
def extract_hubert(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)

    inputs = feature_extractor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = hubert_model(input_values)

    return outputs.last_hidden_state.squeeze(0).cpu()

# ------------------------------
# Save embeddings
# ------------------------------
for path in tqdm(train_paths):
    emb = extract_hubert(path)
    file_name = path.split("/")[-1].replace(".wav", ".pt")
    torch.save(emb, os.path.join(hubert_train_dir, file_name))

# ------------------------------
# Load embeddings
# ------------------------------
def load_train_embedding(path):
    file_name = path.split("/")[-1].replace(".wav", ".pt")
    return torch.load(os.path.join(hubert_train_dir, file_name))

# ------------------------------
# BiLSTM Model
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

    def forward(self, x):
        x = x.unsqueeze(0)         # [1,T,768]
        out, _ = self.lstm(x)      # [1,T,256]
        pooled = out.mean(dim=1)   # [1,256]
        output = self.fc(pooled)   # [1,7]
        return output

# ------------------------------
# Training setup
# ------------------------------
model = EmotionBiLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 23

# ------------------------------
# Training loop
# ------------------------------
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for path, label in zip(train_paths, train_labels):

        features = load_train_embedding(path).to(device)
        label_tensor = torch.tensor([label]).to(device)

        outputs = model(features)
        loss = criterion(outputs, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss)

# ------------------------------
# Save trained model
# ------------------------------
torch.save(model.state_dict(),
           "/content/New_Project_pipeline/speech_model.pt")

print("Speech model training complete.")
