
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# PATHS
# ---------------------------
model_path = "/content/drive/MyDrive/project/models/text_pipeline/text_model.pt"
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
test_texts = []
test_labels = []

for emo in os.listdir(test_split_path):
    emo_path = os.path.join(test_split_path, emo)

    for file in os.listdir(emo_path):
        word = file.split("_")[1]
        sentence = f"say the word {word}"

        test_texts.append(sentence)
        test_labels.append(emotion_map[emo])

# ---------------------------
# LOAD BERT
# ---------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# ---------------------------
# MODEL
# ---------------------------
class TextEmotionClassifier(nn.Module):
    def __init__(self,input_dim=768,num_classes=7):
        super().__init__()
        self.fc = nn.Linear(input_dim,num_classes)

    def forward(self,x):
        return self.fc(x)

model = TextEmotionClassifier().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ---------------------------
# TEST LOOP
# ---------------------------
preds = []

for text in test_texts:

    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=16).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    text_emb = outputs.last_hidden_state[:,0,:]

    pred = torch.argmax(model(text_emb),dim=1).item()
    preds.append(pred)

# ---------------------------
# RESULTS
# ---------------------------
acc = accuracy_score(test_labels,preds)

print("Text Test Accuracy:",acc)
print("\nClassification Report:\n")
print(classification_report(test_labels,preds))
