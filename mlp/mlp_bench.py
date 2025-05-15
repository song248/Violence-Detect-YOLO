import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import csv
from tqdm import tqdm
from ultralytics import YOLO
from fight_module.fight_detector import FightDetector
from fight_module.yolo_pose_estimation import YoloPoseEstimation

yolo_model_path = "model/yolo/yolov8x-pose.pt"
fight_model_path = "model/fight/new-fight-model.pth"
input_dir = "violence"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

pose_estimator = YoloPoseEstimation(yolo_model_path)
fight_detector = FightDetector(fight_model_path)
fight_detector.threshold = 0.6  # default = 0.5
fight_detector.conclusion_threshold = 2
fight_detector.final_threshold = 15

video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] fail to read video: {video_path}")
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

            out_video.write(annotated_frame)
            csv_data.append([frame_count, is_fight])

            frame_count += 1
            pbar.update(1)

    cap.release()
    out_video.release()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"[INFO] Complete to save: {out_path}, {csv_path}")
