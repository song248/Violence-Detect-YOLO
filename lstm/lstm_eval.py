import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from lstm_train import LSTMClassifier, PoseSequenceDataset
from torch.utils.data import DataLoader

SEQ_LEN = 30
BATCH_SIZE = 64
MODEL_PATH = "model/fight/lstm_fight_model.pth"

dataset = PoseSequenceDataset("features.npy", "labels.npy", seq_len=SEQ_LEN)

test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = (outputs > 0.5).float()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=["normal", "violence"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
