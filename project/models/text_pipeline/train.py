
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ---------------------------
# PATHS
# ---------------------------
train_split_path = "/content/drive/MyDrive/Speech_Emotion_Project/TESS_SPLIT/train"
model_save_path = "/content/drive/MyDrive/project/models/text_pipeline/text_model.pt"

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
# LOAD TRAIN DATA
# ---------------------------
train_texts = []
train_labels = []

for emo in os.listdir(train_split_path):
    emo_path = os.path.join(train_split_path, emo)

    for file in os.listdir(emo_path):
        word = file.split("_")[1]
        sentence = f"say the word {word}"

        train_texts.append(sentence)
        train_labels.append(emotion_map[emo])

# ---------------------------
# LOAD BERT
# ---------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# ---------------------------
# TEXT CLASSIFIER
# ---------------------------
class TextEmotionClassifier(nn.Module):
    def __init__(self,input_dim=768,num_classes=7):
        super().__init__()
        self.fc = nn.Linear(input_dim,num_classes)

    def forward(self,x):
        return self.fc(x)

model = TextEmotionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)

epochs = 10

# ---------------------------
# TRAIN LOOP
# ---------------------------
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for text,label in tqdm(zip(train_texts,train_labels), total=len(train_texts)):

        inputs = tokenizer(text,
                           return_tensors="pt",
                           truncation=True,
                           padding=True,
                           max_length=16).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        text_emb = outputs.last_hidden_state[:,0,:]

        label_tensor = torch.tensor([label]).to(device)

        preds = model(text_emb)
        loss = criterion(preds,label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss)

# ---------------------------
# SAVE MODEL
# ---------------------------
torch.save(model.state_dict(), model_save_path)
print("Text model saved.")
