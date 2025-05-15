import cv2
import torch
import os
import csv
from collections import deque
from tqdm import tqdm
from fight_module.yolo_pose_estimation import YoloPoseEstimation
from fight_module.util import calculate_angle, is_coordinate_zero
from lstm_train import LSTMClassifier

# ========== 설정 ==========
yolo_model_path = "model/yolo/yolov8x-pose.pt"
lstm_model_path = "model/fight/lstm_fight_model.pth"
input_dir = "violence"
output_dir = "output"
seq_len = 30
threshold = 0.5
os.makedirs(output_dir, exist_ok=True)
# ==========================

# 관절 조합 (MLP에서 쓰던 것과 동일)
KEYPOINT_PAIRS = [
    [8, 6, 2], [11, 5, 7], [6, 8, 10], [5, 7, 9],
    [6, 12, 14], [5, 11, 13], [12, 14, 16], [11, 13, 15]
]

def extract_features(conf, xyn):
    features = []
    for a, b, c in KEYPOINT_PAIRS:
        if is_coordinate_zero(xyn[a], xyn[b], xyn[c]):
            return None
        angle = calculate_angle(xyn[a], xyn[b], xyn[c])
        avg_conf = sum([conf[a], conf[b], conf[c]]) / 3
        features.extend([angle, avg_conf])
    return features

# 모델 로드
pose_estimator = YoloPoseEstimation(yolo_model_path)
model = LSTMClassifier()
model.load_state_dict(torch.load(lstm_model_path, map_location=torch.device("cpu")))
model.eval()

# 영상 목록
video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(output_dir, video_file)
    csv_path = os.path.join(output_dir, os.path.splitext(video_file)[0] + "_lstm.csv")

    out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    csv_data = [["frame", "violence"]]

    feature_buffer = deque(maxlen=seq_len)
    frame_index_buffer = deque(maxlen=seq_len)
    frame_count = 0
    latest_prediction = 0

    with tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = frame.copy()
            results = pose_estimator.estimate(frame)

            for r in results:
                if r.keypoints and r.keypoints.xy is not None and r.keypoints.conf is not None:
                    keypoints = r.keypoints.xy[0].cpu().numpy().tolist()
                    confs = r.keypoints.conf[0].cpu().numpy().tolist()
                    features = extract_features(confs, keypoints)
                    if features:
                        feature_buffer.append(features)
                        frame_index_buffer.append(frame_count)

                    # 시퀀스가 준비되면 LSTM 예측
                    if len(feature_buffer) == seq_len:
                        X = torch.tensor([list(feature_buffer)], dtype=torch.float32)
                        with torch.no_grad():
                            pred = model(X).item()
                        latest_prediction = int(pred > threshold)

                        # 해당 시퀀스 프레임들에 결과 저장
                        for idx in list(frame_index_buffer):
                            # 아직 기록되지 않은 프레임만 append
                            while len(csv_data) <= idx + 1:
                                csv_data.append([idx, latest_prediction])
                    break

            # 시퀀스 부족한 프레임 → 0으로 간주
            if len(feature_buffer) < seq_len:
                csv_data.append([frame_count, 0])

            # 영상 상단에 텍스트 표시
            label = "FIGHT" if latest_prediction else "NORMAL"
            color = (0, 0, 255) if latest_prediction else (0, 255, 0)
            cv2.putText(annotated_frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            out_video.write(annotated_frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out_video.release()

    # CSV 저장
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"[INFO] 저장 완료: {out_path}")
    print(f"[INFO] CSV 저장 완료: {csv_path}")
