# ============================================================
# SPEECH PIPELINE TRAIN SCRIPT
# HuBERT + BiLSTM
# Train + Validation + Pooled feature saving
# ============================================================

import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torch.nn as nn

# ============================================================
# DEVICE
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================
# DATASET PATH
# ============================================================
dataset_path = "/content/drive/MyDrive/TESS Toronto emotional speech set data"

emotion_files = {}

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):

            emotion = file.split("_")[-1].replace(".wav","")

            if emotion not in emotion_files:
                emotion_files[emotion] = []

            emotion_files[emotion].append(os.path.join(folder_path, file))

# ============================================================
# SPLIT 80 / 10 / 10
# ============================================================

train_paths, val_paths, test_paths = [], [], []
train_labels, val_labels, test_labels = [], [], []

emotion_map = {
    "angry":0,
    "disgust":1,
    "fear":2,
    "happy":3,
    "pleasant_surprise":4,
    "ps":4,
    "sad":5,
    "neutral":6
}

for emotion, files in emotion_files.items():

    train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
    val_files, test_files   = train_test_split(temp_files, test_size=0.5, random_state=42)

    train_paths.extend(train_files)
    val_paths.extend(val_files)
    test_paths.extend(test_files)

    train_labels.extend([emotion_map[emotion]] * len(train_files))
    val_labels.extend([emotion_map[emotion]] * len(val_files))
    test_labels.extend([emotion_map[emotion]] * len(test_files))

print(len(train_paths), len(val_paths), len(test_paths))

# ============================================================
# PROJECT DIR
# ============================================================

project_root = "/content/drive/MyDrive/New_Project_pipeline2"

hubert_train_dir = os.path.join(project_root, "hubert_embeddings_train")
hubert_val_dir   = os.path.join(project_root, "hubert_embeddings_val")

speech_train_pooled_dir = os.path.join(project_root, "speech_train_pooled")
speech_val_pooled_dir   = os.path.join(project_root, "speech_val_pooled")

os.makedirs(hubert_train_dir, exist_ok=True)
os.makedirs(hubert_val_dir, exist_ok=True)
os.makedirs(speech_train_pooled_dir, exist_ok=True)
os.makedirs(speech_val_pooled_dir, exist_ok=True)

# ============================================================
# HUBERT MODEL
# ============================================================

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
hubert_model.eval()

def extract_hubert(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = hubert_model(input_values)

    return outputs.last_hidden_state.squeeze(0).cpu()

# ============================================================
# SAVE EMBEDDINGS
# ============================================================

for path in tqdm(train_paths):
    emb = extract_hubert(path)
    torch.save(emb, os.path.join(hubert_train_dir, path.split("/")[-1].replace(".wav",".pt")))

for path in tqdm(val_paths):
    emb = extract_hubert(path)
    torch.save(emb, os.path.join(hubert_val_dir, path.split("/")[-1].replace(".wav",".pt")))

# ============================================================
# LOADERS
# ============================================================

def load_train_embedding(path):
    return torch.load(os.path.join(hubert_train_dir, path.split("/")[-1].replace(".wav",".pt")))

def load_val_embedding(path):
    return torch.load(os.path.join(hubert_val_dir, path.split("/")[-1].replace(".wav",".pt")))

# ============================================================
# MODEL
# ============================================================

class EmotionBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(768, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 7)

    def forward(self, x, return_features=False):
        x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)

        if return_features:
            return pooled

        return self.fc(pooled)

model = EmotionBiLSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ============================================================
# TRAINING
# ============================================================

epochs = 20
train_losses, val_losses = [], []

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

    train_losses.append(total_loss)

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for path, label in zip(val_paths, val_labels):

            features = load_val_embedding(path).to(device)
            label_tensor = torch.tensor([label]).to(device)

            outputs = model(features)
            loss = criterion(outputs, label_tensor)

            val_loss += loss.item()

    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(model.state_dict(), os.path.join(project_root,"final_hubert_bilstm.pt"))

# ============================================================
# SAVE POOLED FEATURES
# ============================================================

model.eval()

for path in tqdm(train_paths):
    pooled = model(load_train_embedding(path).to(device), return_features=True)
    torch.save(pooled.cpu(), os.path.join(speech_train_pooled_dir, path.split("/")[-1].replace(".wav",".pt")))

for path in tqdm(val_paths):
    pooled = model(load_val_embedding(path).to(device), return_features=True)
    torch.save(pooled.cpu(), os.path.join(speech_val_pooled_dir, path.split("/")[-1].replace(".wav",".pt")))

print("TRAINING COMPLETE")
