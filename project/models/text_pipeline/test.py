# ============================================================
# TEXT PIPELINE TEST SCRIPT
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

project_root = "/content/drive/MyDrive/New_Project_pipeline2"
text_test_dir = os.path.join(project_root, "bert_embeddings_test")
results_dir = os.path.join(project_root, "Results")
os.makedirs(results_dir, exist_ok=True)

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
text_model.load_state_dict(torch.load(os.path.join(project_root,"final_bert_text_model.pt")))
text_model.eval()

# ============================================================
# LOAD TEST EMBEDDINGS
# ============================================================

def load_text_test(path):
    return torch.load(os.path.join(text_test_dir, path.split("/")[-1].replace(".wav",".pt")))

# (Use same test_paths and test_labels as split step)

preds = []
labels = []
features = []

for path, label in zip(test_paths, test_labels):

    emb = load_text_test(path).to(device)

    with torch.no_grad():
        outputs = text_model(emb)

    pred = torch.argmax(outputs, dim=1).item()

    preds.append(pred)
    labels.append(label)
    features.append(emb.cpu().numpy())

features = np.vstack(features)

# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(labels, preds)
print("Text Accuracy:", acc)

pd.DataFrame({"Model":["Text"],"Accuracy":[acc]}).to_csv(
    os.path.join(results_dir,"text_accuracy.csv"),index=False)

report = classification_report(labels, preds)
print(report)

with open(os.path.join(results_dir,"text_classification_report.txt"),"w") as f:
    f.write(report)

# CONFUSION MATRIX
cm = confusion_matrix(labels, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig(os.path.join(results_dir,"text_confusion_matrix.png"))

# TSNE
scaler = StandardScaler()
features_norm = scaler.fit_transform(features)

tsne = TSNE(n_components=2, perplexity=30, n_iter=2000)
tsne_2d = tsne.fit_transform(features_norm)

plt.figure(figsize=(10,8))
plt.scatter(tsne_2d[:,0], tsne_2d[:,1])
plt.savefig(os.path.join(results_dir,"text_tsne.png"))

print("TEXT TEST COMPLETE")
