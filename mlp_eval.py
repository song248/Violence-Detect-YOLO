import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from mlp_train import ThreeLayerClassifier

# 1. 데이터 로딩
X_test = torch.tensor(np.load("test_features.npy"), dtype=torch.float32)
y_test = torch.tensor(np.load("test_labels.npy"), dtype=torch.float32)

# 2. 모델 로딩
model = ThreeLayerClassifier(input_size=16)
model.load_state_dict(torch.load("model/fight/new-fight-model.pth"))
model.eval()

# 3. 예측
with torch.no_grad():
    preds = model(X_test).squeeze()
    preds_label = (preds > 0.6).float()

# 4. 평가
print(classification_report(y_test.numpy(), preds_label.numpy(), target_names=["normal", "violence"]))
print(confusion_matrix(y_test.numpy(), preds_label.numpy()))
