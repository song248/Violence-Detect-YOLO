import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
from collections import deque
from tqdm import tqdm
from fight_module.yolo_pose_estimation import YoloPoseEstimation
from lstm_train import LSTMClassifier
from fight_module.util import calculate_angle, is_coordinate_zero

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

def run_lstm_inference(video_path):
    yolo_model_path = "model/yolo/yolov8x-pose.pt"
    lstm_model_path = "model/fight/lstm_fight_model.pth"
    seq_len = 30
    threshold = 0.5
    output_path = f"lstm_result_{os.path.basename(video_path)}"

    pose_estimator = YoloPoseEstimation(yolo_model_path)
    model = LSTMClassifier()
    model.load_state_dict(torch.load(lstm_model_path, map_location=torch.device("cpu")))
    model.eval()

    feature_buffer = deque(maxlen=seq_len)
    latest_prediction = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] fail to read video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
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

                    if len(feature_buffer) == seq_len:
                        X = torch.tensor([list(feature_buffer)], dtype=torch.float32)
                        with torch.no_grad():
                            pred = model(X).item()
                        latest_prediction = pred

                    annotated_frame = r.plot()
                    break

            if len(feature_buffer) < seq_len or latest_prediction is None:
                label = "WARMING UP"
                color = (200, 200, 0)
            else:
                label = "FIGHT" if latest_prediction > threshold else "NORMAL"
                color = (0, 0, 255) if label == "FIGHT" else (0, 255, 0)

            cv2.putText(annotated_frame, f"{label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            out.write(annotated_frame)
            cv2.imshow("LSTM Fight Detection", annotated_frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Complete to save result: {output_path}")
    print(f"[INFO] Total frames: Input {total_frames}, Save {frame_count}")

if __name__ == "__main__":
    video_path = "test_video.mp4"
    run_lstm_inference(video_path)
