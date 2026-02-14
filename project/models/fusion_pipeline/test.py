
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# PATHS
# ---------------------------
model_path = "/content/drive/MyDrive/project/models/fusion_pipeline/fusion_model.pt"
test_split_path = "/content/drive/MyDrive/Speech_Emotion_Project/TESS_SPLIT/test"
test_embed_dir = "/content/drive/MyDrive/Speech_Emotion_Project/hubert_test_embeddings"

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
test_texts = []
test_labels = []

for emo in os.listdir(test_split_path):
    emo_path = os.path.join(test_split_path, emo)

    for file in os.listdir(emo_path):

        path = os.path.join(emo_path,file)
        word = file.split("_")[1]
        sentence = f"say the word {word}"

        test_paths.append(path)
        test_texts.append(sentence)
        test_labels.append(emotion_map[emo])

# ---------------------------
# LOAD EMBEDDINGS
# ---------------------------
def load_test_embedding(path):
    file_name = path.split("/")[-1].replace(".wav",".pt")
    return torch.load(os.path.join(test_embed_dir,file_name))

# ---------------------------
# LOAD BERT
# ---------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# ---------------------------
# MODEL
# ---------------------------
class FusionEmotionModel(nn.Module):
    def __init__(self,speech_dim=768,text_dim=768,num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(speech_dim+text_dim,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,num_classes)

    def forward(self,speech_emb,text_emb):
        fused = torch.cat((speech_emb,text_emb),dim=1)
        x = self.fc1(fused)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = FusionEmotionModel().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ---------------------------
# TEST LOOP
# ---------------------------
preds = []

for path,text in zip(test_paths,test_texts):

    speech_emb = load_test_embedding(path).unsqueeze(0).to(device)
    speech_emb = speech_emb.mean(dim=1)

    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=16).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    text_emb = outputs.last_hidden_state[:,0,:]

    pred = torch.argmax(model(speech_emb,text_emb),dim=1).item()
    preds.append(pred)

# ---------------------------
# RESULTS
# ---------------------------
acc = accuracy_score(test_labels,preds)

print("Fusion Test Accuracy:",acc)
print("\nClassification Report:\n")
print(classification_report(test_labels,preds))
