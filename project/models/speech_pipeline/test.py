# ============================================================
# SPEECH PIPELINE TEST SCRIPT
# ============================================================

import os
import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PATHS
# ============================================================

project_root = "/content/drive/MyDrive/New_Project_pipeline2"

hubert_test_dir  = os.path.join(project_root, "hubert_embeddings_test")
results_dir = os.path.join(project_root, "Results")

os.makedirs(results_dir, exist_ok=True)

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
model.load_state_dict(torch.load(os.path.join(project_root,"final_hubert_bilstm.pt")))
model.eval()

# ============================================================
# LOAD TEST EMBEDDINGS
# ============================================================

def load_test_embedding(path):
    return torch.load(os.path.join(hubert_test_dir, path.split("/")[-1].replace(".wav",".pt")))

# (Use same test_paths and test_labels from dataset split script)

# ============================================================
# EVALUATION
# ============================================================

preds, labels, features = [], [], []

for path, label in zip(test_paths, test_labels):

    emb = load_test_embedding(path).to(device)

    with torch.no_grad():
        outputs = model(emb)

    pred = torch.argmax(outputs, dim=1).item()

    preds.append(pred)
    labels.append(label)

    pooled = model(emb, return_features=True)
    features.append(pooled.cpu().numpy())

features = np.vstack(features)

# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(labels, preds)
print("Speech Accuracy:", acc)

pd.DataFrame({"Model":["Speech"],"Accuracy":[acc]}).to_csv(
    os.path.join(results_dir,"speech_accuracy.csv"),index=False)

report = classification_report(labels, preds)
print(report)

with open(os.path.join(results_dir,"speech_classification_report.txt"),"w") as f:
    f.write(report)

# CONFUSION MATRIX
cm = confusion_matrix(labels, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig(os.path.join(results_dir,"speech_confusion_matrix.png"))

# TSNE
scaler = StandardScaler()
features_norm = scaler.fit_transform(features)

tsne = TSNE(n_components=2, perplexity=30, n_iter=2000)
tsne_2d = tsne.fit_transform(features_norm)

plt.figure(figsize=(10,8))
plt.scatter(tsne_2d[:,0], tsne_2d[:,1])
plt.savefig(os.path.join(results_dir,"speech_tsne.png"))

print("TEST COMPLETE")
