
import os
import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------
# PATHS
# ---------------------------
train_embed_dir = "/content/drive/MyDrive/Speech_Emotion_Project/hubert_train_embeddings"
train_split_path = "/content/drive/MyDrive/Speech_Emotion_Project/TESS_SPLIT/train"
model_save_path = "/content/drive/MyDrive/project/models/speech_pipeline/speech_model.pt"

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
train_labels = []

for emo in os.listdir(train_split_path):
    emo_path = os.path.join(train_split_path, emo)

    for file in os.listdir(emo_path):
        train_paths.append(os.path.join(emo_path, file))
        train_labels.append(emotion_map[emo])

# ---------------------------
# LOAD EMBEDDINGS
# ---------------------------
def load_train_embedding(path):
    file_name = path.split("/")[-1].replace(".wav",".pt")
    return torch.load(os.path.join(train_embed_dir,file_name))

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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs = 20

# ---------------------------
# TRAIN LOOP
# ---------------------------
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for path,label in tqdm(zip(train_paths,train_labels), total=len(train_paths)):

        features = load_train_embedding(path).unsqueeze(0).to(device)
        label_tensor = torch.tensor([label]).to(device)

        outputs = model(features)
        loss = criterion(outputs,label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss)

# ---------------------------
# SAVE MODEL
# ---------------------------
torch.save(model.state_dict(), model_save_path)
print("Speech model saved.")
