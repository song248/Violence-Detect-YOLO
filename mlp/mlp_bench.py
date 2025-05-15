import cv2
import torch
import os
import csv
from tqdm import tqdm
from ultralytics import YOLO
from fight_module.fight_detector import FightDetector
from fight_module.yolo_pose_estimation import YoloPoseEstimation

# 설정
yolo_model_path = "model/yolo/yolov8x-pose.pt"
fight_model_path = "model/fight/new-fight-model.pth"
input_dir = "violence"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 모델 로드
pose_estimator = YoloPoseEstimation(yolo_model_path)
fight_detector = FightDetector(fight_model_path)
fight_detector.threshold = 0.6  # 실험에서 좋은 성능을 보인 값
fight_detector.conclusion_threshold = 2
fight_detector.final_threshold = 15

# 영상 처리 루프
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
    csv_path = os.path.join(output_dir, os.path.splitext(video_file)[0] + ".csv")
    out_video = cv2.VideoWriter(out_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps,
                                 (width, height))

    # CSV 저장을 위한 리스트
    csv_data = [["frame", "violence"]]
    frame_count = 0

    with tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = frame.copy()
            is_fight = 0

            results = pose_estimator.estimate(frame)

            for r in results:
                if (
                    r.keypoints is not None and 
                    r.keypoints.xy is not None and 
                    r.keypoints.conf is not None
                ):
                    keypoints = r.keypoints.xy[0].cpu().numpy().tolist()
                    confs = r.keypoints.conf[0].cpu().numpy().tolist()
                    is_fight = int(fight_detector.detect(confs, keypoints))
                    annotated_frame = r.plot()
                    label = "FIGHT" if is_fight else "NORMAL"
                    color = (0, 0, 255) if is_fight else (0, 255, 0)
                    cv2.putText(annotated_frame, label, (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    break

            # 결과 저장
            out_video.write(annotated_frame)
            csv_data.append([frame_count, is_fight])

            frame_count += 1
            pbar.update(1)

    cap.release()
    out_video.release()

    # CSV 파일 저장
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"[INFO] 저장 완료: {out_path}, {csv_path}")
