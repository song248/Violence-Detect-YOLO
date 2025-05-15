import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os


# ============================
# 1. Dataset 정의
# ============================

class PoseSequenceDataset(Dataset):
    def __init__(self, features_path, labels_path, seq_len=30):
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
        self.seq_len = seq_len
        self.X_seq, self.y_seq = self._build_sequences()

    def _build_sequences(self):
        X_seq, y_seq = [], []
        for i in range(len(self.features) - self.seq_len + 1):
            X_seq.append(self.features[i:i + self.seq_len])
            y_seq.append(self.labels[i + self.seq_len - 1])
        return np.array(X_seq), np.array(y_seq)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return torch.tensor(self.X_seq[idx], dtype=torch.float32), \
               torch.tensor(self.y_seq[idx], dtype=torch.float32).squeeze()


# ============================
# 2. LSTM 모델 정의
# ============================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 시점의 출력
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)


# ============================
# 3. 학습 루프
# ============================

def train(model, dataloader, optimizer, criterion, device, grad_clip=None):
    model.train()
    losses, preds, targets = [], [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        preds.extend((output > 0.5).cpu().numpy())
        targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return np.mean(losses), acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            losses.append(loss.item())
            preds.extend((output > 0.5).cpu().numpy())
            targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return np.mean(losses), acc


# ============================
# 4. 실행
# ============================

def main():
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4
    SEQ_LEN = 30
    GRAD_CLIP = 1.0
    PATIENCE = 10
    BEST_MODEL_PATH = "model/fight/lstm_fight_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoseSequenceDataset("features.npy", "labels.npy", seq_len=SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, grad_clip=GRAD_CLIP)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping: {PATIENCE}")
                break

if __name__ == "__main__":
    os.makedirs("model/fight", exist_ok=True)
    main()
