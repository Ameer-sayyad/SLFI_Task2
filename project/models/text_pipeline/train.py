# ============================================================
# TEXT PIPELINE TRAIN SCRIPT
# BERT CLS + Linear Classifier
# Train + Validation
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import matplotlib.pyplot as plt

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

print("Train:", len(train_paths))
print("Val:", len(val_paths))
print("Test:", len(test_paths))

# ============================================================
# BUILD TEXT DATASET
# ============================================================

def build_text_dataset(paths):
    texts = []
    for path in paths:
        file = path.split("/")[-1]
        word = file.split("_")[1]
        sentence = f"say the word {word}"
        texts.append(sentence)
    return texts

train_texts = build_text_dataset(train_paths)
val_texts   = build_text_dataset(val_paths)

# ============================================================
# PROJECT DIR
# ============================================================

project_root = "/content/drive/MyDrive/New_Project_pipeline2"

text_train_dir = os.path.join(project_root, "bert_embeddings_train")
text_val_dir   = os.path.join(project_root, "bert_embeddings_val")

os.makedirs(text_train_dir, exist_ok=True)
os.makedirs(text_val_dir, exist_ok=True)

# ============================================================
# LOAD BERT
# ============================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

def extract_cls(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=16
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    cls_embedding = outputs.last_hidden_state[:,0,:]
    return cls_embedding.cpu()

# ============================================================
# SAVE EMBEDDINGS
# ============================================================

for text, path in tqdm(zip(train_texts, train_paths), total=len(train_texts)):
    emb = extract_cls(text)
    torch.save(emb, os.path.join(text_train_dir, path.split("/")[-1].replace(".wav",".pt")))

for text, path in tqdm(zip(val_texts, val_paths), total=len(val_texts)):
    emb = extract_cls(text)
    torch.save(emb, os.path.join(text_val_dir, path.split("/")[-1].replace(".wav",".pt")))

# ============================================================
# LOADERS
# ============================================================

def load_text_train(path):
    return torch.load(os.path.join(text_train_dir, path.split("/")[-1].replace(".wav",".pt")))

def load_text_val(path):
    return torch.load(os.path.join(text_val_dir, path.split("/")[-1].replace(".wav",".pt")))

# ============================================================
# MODEL
# ============================================================

class TextEmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 7)

    def forward(self, x):
        return self.fc(x)

text_model = TextEmotionClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(text_model.parameters(), lr=2e-5)

# ============================================================
# TRAINING
# ============================================================

epochs = 30
train_losses = []
val_losses = []

for epoch in range(epochs):

    text_model.train()
    total_loss = 0

    for path, label in zip(train_paths, train_labels):

        emb = load_text_train(path).to(device)
        label_tensor = torch.tensor([label]).to(device)

        preds = text_model(emb)
        loss = criterion(preds, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_losses.append(total_loss)

    # VALIDATION
    text_model.eval()
    val_loss = 0

    with torch.no_grad():
        for path, label in zip(val_paths, val_labels):

            emb = load_text_val(path).to(device)
            label_tensor = torch.tensor([label]).to(device)

            preds = text_model(emb)
            loss = criterion(preds, label_tensor)

            val_loss += loss.item()

    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(text_model.state_dict(),
           os.path.join(project_root,"final_bert_text_model.pt"))

print("TEXT TRAINING COMPLETE")
