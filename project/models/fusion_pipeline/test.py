# ============================================================
# FUSION PIPELINE TEST SCRIPT
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

speech_test_pooled_dir = os.path.join(project_root,"speech_test_pooled")
text_test_dir          = os.path.join(project_root,"bert_embeddings_test")
results_dir            = os.path.join(project_root,"Results")

os.makedirs(results_dir, exist_ok=True)

# ============================================================
# LOADERS
# ============================================================

def load_speech_test(path):
    return torch.load(os.path.join(speech_test_pooled_dir, path.split("/")[-1].replace(".wav",".pt")))

def load_text_test(path):
    return torch.load(os.path.join(text_test_dir, path.split("/")[-1].replace(".wav",".pt")))

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
fusion_model.load_state_dict(torch.load(os.path.join(project_root,"best_fusion_model.pt")))
fusion_model.eval()

# (Use same test_paths and test_labels from split step)

fusion_preds = []
fusion_labels = []
fusion_features = []

for path, label in zip(test_paths, test_labels):

    speech_emb = load_speech_test(path).to(device)
    text_emb   = load_text_test(path).to(device)

    with torch.no_grad():
        outputs = fusion_model(speech_emb, text_emb)

    pred = torch.argmax(outputs, dim=1).item()

    fusion_preds.append(pred)
    fusion_labels.append(label)

    fused_vec = torch.cat((speech_emb, text_emb), dim=1)
    fusion_features.append(fused_vec.cpu().numpy())

fusion_features = np.vstack(fusion_features)

# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(fusion_labels, fusion_preds)
print("Fusion Accuracy:", acc)

pd.DataFrame({"Model":["Fusion"],"Accuracy":[acc]}).to_csv(
    os.path.join(results_dir,"fusion_accuracy.csv"),index=False)

report = classification_report(fusion_labels, fusion_preds)
print(report)

with open(os.path.join(results_dir,"fusion_classification_report.txt"),"w") as f:
    f.write(report)

# CONFUSION MATRIX
cm = confusion_matrix(fusion_labels, fusion_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig(os.path.join(results_dir,"fusion_confusion_matrix.png"))

# TSNE
scaler = StandardScaler()
fusion_norm = scaler.fit_transform(fusion_features)

tsne = TSNE(n_components=2, perplexity=30, n_iter=2000)
fusion_2d = tsne.fit_transform(fusion_norm)

plt.figure(figsize=(10,8))
plt.scatter(fusion_2d[:,0], fusion_2d[:,1])
plt.savefig(os.path.join(results_dir,"fusion_tsne.png"))

print("FUSION TEST COMPLETE")
