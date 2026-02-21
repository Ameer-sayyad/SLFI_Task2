# ==========================================
# FUSION PIPELINE TEST
# ==========================================

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel

# ------------------------------
# PATHS
# ------------------------------
test_base = "/content/TESS_SPLIT/test"
hubert_test_dir = "/content/New_Project_pipeline/hubert_test_embeddings"
results_dir = "/content/New_Project_pipeline/results_fusion"
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

def build_text(path):
    file = path.split("/")[-1]
    word = file.split("_")[1]
    return f"say the word {word}"

test_texts = [build_text(p) for p in test_paths]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load Speech Model
# ------------------------------
class EmotionBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(768,128,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(256,7)

    def forward(self,x,return_features=False):
        x = x.unsqueeze(0)
        out,_ = self.lstm(x)
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

# ------------------------------
# Fusion Model
# ------------------------------
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024,7)

    def forward(self,speech_emb,text_emb):
        fused = torch.cat((speech_emb,text_emb),dim=1)
        return self.fc(fused)

fusion_model = FusionModel().to(device)
fusion_model.load_state_dict(
    torch.load("/content/New_Project_pipeline/fusion_model.pt",
               map_location=device)
)
fusion_model.eval()

# ------------------------------
# Load embeddings
# ------------------------------
def load_test_embedding(path):
    file_name = path.split("/")[-1].replace(".wav",".pt")
    full_path = os.path.join(hubert_test_dir,file_name)
    if os.path.exists(full_path):
        return torch.load(full_path)
    return None

fusion_preds = []
fusion_labels = []
fusion_features = []

for path,label,text in zip(test_paths,test_labels,test_texts):

    speech_frames = load_test_embedding(path)
    if speech_frames is None:
        continue

    speech_frames = speech_frames.to(device)

    with torch.no_grad():
        speech_emb = speech_model(speech_frames,return_features=True)

    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=16).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    text_emb = outputs.last_hidden_state[:,0,:]

    with torch.no_grad():
        outputs = fusion_model(speech_emb,text_emb)

    pred = torch.argmax(outputs,dim=1).item()

    fusion_preds.append(pred)
    fusion_labels.append(label)

    fused_vec = torch.cat((speech_emb,text_emb),dim=1)
    fusion_features.append(fused_vec.cpu().numpy())

fusion_features = np.vstack(fusion_features)

# ------------------------------
# Metrics
# ------------------------------
acc = accuracy_score(fusion_labels,fusion_preds)
print("Fusion Accuracy:",acc)
print(classification_report(fusion_labels,fusion_preds))

with open(os.path.join(results_dir,"fusion_report.txt"),"w") as f:
    f.write(classification_report(fusion_labels,fusion_preds))

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(fusion_labels,fusion_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt="d",
            xticklabels=emotion_names,
            yticklabels=emotion_names)
plt.title("Fusion Confusion Matrix")
plt.savefig(os.path.join(results_dir,"fusion_confusion.png"))
plt.close()

# ------------------------------
# t-SNE (Clean clustering)
# ------------------------------
scaler = StandardScaler()
fusion_features_norm = scaler.fit_transform(fusion_features)

tsne = TSNE(n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=2000,
            random_state=42,
            init="pca")

fusion_2d = tsne.fit_transform(fusion_features_norm)

plt.figure(figsize=(10,8))
for i, emo in enumerate(emotion_names):
    idx = np.where(np.array(fusion_labels)==i)
    plt.scatter(fusion_2d[idx,0],
                fusion_2d[idx,1],
                label=emo)

plt.legend()
plt.title("Fusion t-SNE Representation")
plt.savefig(os.path.join(results_dir,"fusion_tsne.png"),dpi=300)
plt.close()

print("Fusion testing complete.")
