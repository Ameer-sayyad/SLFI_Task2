
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# PATHS
# ---------------------------
model_path = "/content/drive/MyDrive/project/models/speech_pipeline/speech_model.pt"
test_embed_dir = "/content/drive/MyDrive/Speech_Emotion_Project/hubert_test_embeddings"
test_split_path = "/content/drive/MyDrive/Speech_Emotion_Project/TESS_SPLIT/test"

# ---------------------------
# DEVICE
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# EMOTION MAP
# ---------------------------
emotion_map = {
    "angry":0,
    "disgust":1,
    "fear":2,
    "happy":3,
    "ps":4,
    "sad":5,
    "neutral":6
}

# ---------------------------
# LOAD TEST DATA
# ---------------------------
test_paths = []
test_labels = []

for emo in os.listdir(test_split_path):
    emo_path = os.path.join(test_split_path, emo)

    for file in os.listdir(emo_path):
        test_paths.append(os.path.join(emo_path, file))
        test_labels.append(emotion_map[emo])

# ---------------------------
# LOAD EMBEDDINGS
# ---------------------------
def load_test_embedding(path):
    file_name = path.split("/")[-1].replace(".wav",".pt")
    return torch.load(os.path.join(test_embed_dir,file_name))

# ---------------------------
# MODEL
# ---------------------------
class EmotionBiLSTM(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=128,num_layers=1,num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim*2,num_classes)

    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)

model = EmotionBiLSTM().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ---------------------------
# TEST LOOP
# ---------------------------
preds = []

for path in test_paths:
    features = load_test_embedding(path).unsqueeze(0).to(device)
    outputs = model(features)
    pred = torch.argmax(outputs,dim=1).item()
    preds.append(pred)

# ---------------------------
# RESULTS
# ---------------------------
acc = accuracy_score(test_labels,preds)

print("Speech Test Accuracy:",acc)
print("\nClassification Report:\n")
print(classification_report(test_labels,preds))
