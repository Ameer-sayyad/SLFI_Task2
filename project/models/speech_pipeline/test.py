# ==============================
# SPEECH PIPELINE TEST
# ==============================

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

# ------------------------------
# Paths
# ------------------------------
test_base = "/content/TESS_SPLIT/test"
hubert_test_dir = "/content/New_Project_pipeline/hubert_test_embeddings"

results_dir = "/content/New_Project_pipeline/results_speech"
os.makedirs(results_dir, exist_ok=True)

emotion_map = {
    "angry":0,
    "disgust":1,
    "fear":2,
    "happy":3,
    "ps":4,
    "sad":5,
    "neutral":6
}

emotion_names = list(emotion_map.keys())

# ------------------------------
# Load test paths
# ------------------------------
test_paths, test_labels = [], []

for emo in os.listdir(test_base):
    emo_path = os.path.join(test_base, emo)
    for file in os.listdir(emo_path):
        test_paths.append(os.path.join(emo_path, file))
        test_labels.append(emotion_map[emo])

print("Test samples:", len(test_paths))

# ------------------------------
# Load embeddings
# ------------------------------
def load_test_embedding(path):
    file_name = path.split("/")[-1].replace(".wav", ".pt")
    return torch.load(os.path.join(hubert_test_dir, file_name))

# ------------------------------
# Model definition
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
        x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        output = self.fc(pooled)
        return output

# ------------------------------
# Load model
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = EmotionBiLSTM().to(device)
model.load_state_dict(torch.load(
    "/content/New_Project_pipeline/speech_model.pt",
    map_location=device
))
model.eval()

# ------------------------------
# Testing
# ------------------------------
preds = []
features_tsne = []

for path in test_paths:

    features = load_test_embedding(path).to(device)

    with torch.no_grad():
        outputs = model(features)

    pred = torch.argmax(outputs, dim=1).item()
    preds.append(pred)

    features_tsne.append(features.mean(dim=0).numpy())

# ------------------------------
# Metrics
# ------------------------------
acc = accuracy_score(test_labels, preds)
print("Speech Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(test_labels, preds))

# Save report
with open(os.path.join(results_dir, "speech_report.txt"), "w") as f:
    f.write(classification_report(test_labels, preds))

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(test_labels, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=emotion_names,
            yticklabels=emotion_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Speech Confusion Matrix")
plt.savefig(os.path.join(results_dir, "speech_confusion.png"))
plt.close()

# ------------------------------
# t-SNE Visualization
# ------------------------------
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(np.array(features_tsne))

plt.figure(figsize=(10,8))
for i, emo in enumerate(emotion_names):
    idx = np.where(np.array(test_labels) == i)
    plt.scatter(features_2d[idx,0], features_2d[idx,1], label=emo)

plt.legend()
plt.title("Speech t-SNE Representation")
plt.savefig(os.path.join(results_dir, "speech_tsne.png"))
plt.close()

print("Speech testing complete.")
