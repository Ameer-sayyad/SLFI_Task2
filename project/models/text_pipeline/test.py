# ==============================
# TEXT PIPELINE TEST
# ==============================

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel

# ------------------------------
# Paths
# ------------------------------
test_base = "/content/TESS_SPLIT/test"
results_dir = "/content/New_Project_pipeline/results_text"
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
# Build text
# ------------------------------
def build_text(path):
    file = path.split("/")[-1]
    word = file.split("_")[1]
    sentence = f"say the word {word}"
    return sentence

test_texts = [build_text(p) for p in test_paths]

# ------------------------------
# Device
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load BERT
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

for param in bert_model.parameters():
    param.requires_grad = False

# ------------------------------
# Load classifier
# ------------------------------
class TextEmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 7)

    def forward(self, x):
        return self.fc(x)

text_model = TextEmotionClassifier().to(device)
text_model.load_state_dict(
    torch.load("/content/New_Project_pipeline/text_model.pt",
               map_location=device)
)
text_model.eval()

# ------------------------------
# Testing
# ------------------------------
preds = []
features_tsne = []

for text in test_texts:

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

    pred = torch.argmax(text_model(cls_embedding), dim=1).item()
    preds.append(pred)

    features_tsne.append(cls_embedding.cpu().numpy().flatten())

# ------------------------------
# Metrics
# ------------------------------
acc = accuracy_score(test_labels, preds)
print("Text Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(test_labels, preds))

with open(os.path.join(results_dir, "text_report.txt"), "w") as f:
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
plt.title("Text Confusion Matrix")
plt.savefig(os.path.join(results_dir, "text_confusion.png"))
plt.close()

# ------------------------------
# t-SNE
# ------------------------------
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(np.array(features_tsne))

plt.figure(figsize=(10,8))
for i, emo in enumerate(emotion_names):
    idx = np.where(np.array(test_labels) == i)
    plt.scatter(features_2d[idx,0], features_2d[idx,1], label=emo)

plt.legend()
plt.title("Text t-SNE Representation")
plt.savefig(os.path.join(results_dir, "text_tsne.png"))
plt.close()

print("Text testing complete.")
