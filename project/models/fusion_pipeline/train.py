
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ---------------------------
# PATHS
# ---------------------------
train_split_path = "/content/drive/MyDrive/Speech_Emotion_Project/TESS_SPLIT/train"
train_embed_dir = "/content/drive/MyDrive/Speech_Emotion_Project/hubert_train_embeddings"
model_save_path = "/content/drive/MyDrive/project/models/fusion_pipeline/fusion_model.pt"

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
train_paths = []
train_texts = []
train_labels = []

for emo in os.listdir(train_split_path):
    emo_path = os.path.join(train_split_path, emo)

    for file in os.listdir(emo_path):
        path = os.path.join(emo_path, file)

        word = file.split("_")[1]
        sentence = f"say the word {word}"

        train_paths.append(path)
        train_texts.append(sentence)
        train_labels.append(emotion_map[emo])

# ---------------------------
# LOAD EMBEDDINGS
# ---------------------------
def load_train_embedding(path):
    file_name = path.split("/")[-1].replace(".wav",".pt")
    return torch.load(os.path.join(train_embed_dir,file_name))

# ---------------------------
# LOAD BERT
# ---------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# ---------------------------
# FUSION MODEL
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs = 12

# ---------------------------
# TRAIN LOOP
# ---------------------------
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for path,text,label in tqdm(zip(train_paths,train_texts,train_labels),
                                total=len(train_paths)):

        # speech embedding
        speech_emb = load_train_embedding(path).unsqueeze(0).to(device)
        speech_emb = speech_emb.mean(dim=1)

        # text embedding
        inputs = tokenizer(text,
                           return_tensors="pt",
                           truncation=True,
                           padding=True,
                           max_length=16).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        text_emb = outputs.last_hidden_state[:,0,:]

        label_tensor = torch.tensor([label]).to(device)

        preds = model(speech_emb,text_emb)
        loss = criterion(preds,label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:",total_loss)

# ---------------------------
# SAVE MODEL
# ---------------------------
torch.save(model.state_dict(),model_save_path)
print("Fusion model saved.")
