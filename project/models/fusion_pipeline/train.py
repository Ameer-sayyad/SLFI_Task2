# ============================================================
# FUSION PIPELINE TRAIN SCRIPT
# Speech pooled + Text CLS
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn as nn

# ============================================================
# DEVICE
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================
# DATASET SPLIT (80 / 10 / 10)
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

train_paths, val_paths, test_paths = [], [], []
train_labels, val_labels, test_labels = [], [], []

emotion_map = {
    "angry":0,"disgust":1,"fear":2,"happy":3,
    "pleasant_surprise":4,"ps":4,"sad":5,"neutral":6
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

# ============================================================
# PATHS
# ============================================================

project_root = "/content/drive/MyDrive/New_Project_pipeline2"

speech_train_pooled_dir = os.path.join(project_root,"speech_train_pooled")
speech_val_pooled_dir   = os.path.join(project_root,"speech_val_pooled")

text_train_dir = os.path.join(project_root,"bert_embeddings_train")
text_val_dir   = os.path.join(project_root,"bert_embeddings_val")

# ============================================================
# LOADERS
# ============================================================

def load_speech_train(path):
    return torch.load(os.path.join(speech_train_pooled_dir, path.split("/")[-1].replace(".wav",".pt")))

def load_speech_val(path):
    return torch.load(os.path.join(speech_val_pooled_dir, path.split("/")[-1].replace(".wav",".pt")))

def load_text_train(path):
    return torch.load(os.path.join(text_train_dir, path.split("/")[-1].replace(".wav",".pt")))

def load_text_val(path):
    return torch.load(os.path.join(text_val_dir, path.split("/")[-1].replace(".wav",".pt")))

# ============================================================
# MODEL
# ============================================================

class FusionEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 7)

    def forward(self, speech_emb, text_emb):
        fused = torch.cat((speech_emb, text_emb), dim=1)
        return self.fc(fused)

fusion_model = FusionEmotionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=3e-4)

# ============================================================
# TRAINING
# ============================================================

epochs = 20
train_losses, val_losses = [], []

best_val_loss = float("inf")

for epoch in range(epochs):

    fusion_model.train()
    total_loss = 0

    for path, label in zip(train_paths, train_labels):

        speech_emb = load_speech_train(path).to(device)
        text_emb   = load_text_train(path).to(device)

        label_tensor = torch.tensor([label]).to(device)

        preds = fusion_model(speech_emb, text_emb)
        loss = criterion(preds, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_losses.append(total_loss)

    # VALIDATION
    fusion_model.eval()
    val_loss = 0

    with torch.no_grad():
        for path, label in zip(val_paths, val_labels):

            speech_emb = load_speech_val(path).to(device)
            text_emb   = load_text_val(path).to(device)

            label_tensor = torch.tensor([label]).to(device)

            preds = fusion_model(speech_emb, text_emb)
            loss = criterion(preds, label_tensor)

            val_loss += loss.item()

    val_losses.append(val_loss)

    # SAVE BEST MODEL
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(fusion_model.state_dict(),
                   os.path.join(project_root,"best_fusion_model.pt"))

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

print("FUSION TRAINING COMPLETE")
