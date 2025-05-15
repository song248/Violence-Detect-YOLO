import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO

# ────────────────────────────────────────────────
# 1. MLP 모델 정의
# ────────────────────────────────────────────────
class ThreeLayerClassifier(nn.Module):
    def __init__(self, input_size=16, hidden_size=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc3(x))
        return x

# ────────────────────────────────────────────────
# 2. 각도 추출 함수
# ────────────────────────────────────────────────
def angle_between(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_pose_features(keypoints, confidences):
    coordinate_for_angle = [
        [8, 6, 2],
        [11, 5, 7],
        [6, 8, 10],
        [5, 7, 9],
        [6, 12, 14],
        [5, 11, 13],
        [12, 14, 16],
        [11, 13, 15]
    ]
    
    features = []
    try:
        for a, b, c in coordinate_for_angle:
            angle = angle_between(keypoints[a], keypoints[b], keypoints[c])
            mean_conf = np.mean([confidences[a], confidences[b], confidences[c]])
            features.append(angle)
            features.append(mean_conf)
    except:
        return None
    return features

# ────────────────────────────────────────────────
# 3. Dataset 처리 함수
# ────────────────────────────────────────────────
def process_dataset(dataset_path, label, model):
    features, labels = [], []
    for video_file in tqdm(os.listdir(dataset_path)):
        if not video_file.endswith(".mp4"):
            continue
        video_path = os.path.join(dataset_path, video_file)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            keypoints_obj = results[0].keypoints

            if keypoints_obj is None or keypoints_obj.xy is None or keypoints_obj.conf is None:
                continue

            keypoints_list = keypoints_obj.xy
            conf_list = keypoints_obj.conf

            for kp, conf in zip(keypoints_list, conf_list):
                kp = kp.cpu().numpy()
                conf = conf.cpu().numpy()
                feature_vector = extract_pose_features(kp, conf)
                if feature_vector and len(feature_vector) == 16:
                    features.append(feature_vector)
                    labels.append([label])
        cap.release()
    return features, labels

# ────────────────────────────────────────────────
# 4. 학습 함수
# ────────────────────────────────────────────────
def train(X, y):
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    model = ThreeLayerClassifier(input_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("[INFO] Training model...")
    for epoch in range(100):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    os.makedirs("model/fight", exist_ok=True)
    torch.save(model.state_dict(), "model/fight/new-fight-model.pth")
    print("[INFO] Model saved to model/fight/new-fight-model.pth")

# ────────────────────────────────────────────────
# 5. 메인 실행부
# ────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists("features.npy") and os.path.exists("labels.npy"):
        print("[INFO] Loading cached features...")
        X = torch.tensor(np.load("features.npy"), dtype=torch.float32)
        y = torch.tensor(np.load("labels.npy"), dtype=torch.float32)
    else:
        print("[INFO] Extracting features from video...")
        yolo_model = YOLO("model/yolo/yolov8x-pose.pt")
        fight_features, fight_labels = process_dataset("dataset/fight", 1, yolo_model)
        normal_features, normal_labels = process_dataset("dataset/normal", 0, yolo_model)

        all_features = fight_features + normal_features
        all_labels = fight_labels + normal_labels

        np.save("features.npy", np.array(all_features))
        np.save("labels.npy", np.array(all_labels))

        X = torch.tensor(all_features, dtype=torch.float32)
        y = torch.tensor(all_labels, dtype=torch.float32)

    train(X, y)
