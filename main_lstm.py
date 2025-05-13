import cv2
import torch
import os
import csv
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

from fight_module.fight_lstm import FightDetector
from fight_module.util import calculate_angle, is_coordinate_zero

def run_fight_detection(video_path):
    # 설정
    yolo_model_path = "model/yolo/yolov8n.pt"
    fight_model_path = "model/fight/lstm_model.pth"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"result_{os.path.basename(video_path)}")
    csv_log_path = os.path.join(output_dir, f"log_{os.path.basename(video_path)}.csv")
    prob_plot_path = os.path.join(output_dir, f"probability_plot_{os.path.basename(video_path)}.png")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    yolo_detector = YOLO(yolo_model_path)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True)
    fight_detector = FightDetector(fight_model=fight_model_path, fps=int(fps), sequence_length=15)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    csv_file = open(csv_log_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'violence'])

    frame_count = 0
    person_buffers = {}
    frame_list = []
    prob_list = []
    last_avg_prob = 0.0
    last_frame_violence = 0

    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = frame.copy()
            results = yolo_detector(frame, verbose=False)[0]

            frame_violence = 0
            frame_probs = []

            for det_index, det in enumerate(results.boxes.data):
                cls_id = int(det[-1].item())
                if cls_id != 0:
                    continue  # 사람 class만 처리

                x1, y1, x2, y2 = map(int, det[:4])
                crop = frame[max(y1-10,0):min(y2+10,frame.shape[0]), max(x1-10,0):min(x2+10,frame.shape[1])]
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pose_result = mp_pose.process(crop_rgb)

                if not pose_result.pose_landmarks:
                    continue

                keypoints = []
                for lm in pose_result.pose_landmarks.landmark:
                    keypoints.append([
                        lm.x * crop.shape[1] + x1,
                        lm.y * crop.shape[0] + y1,
                        lm.visibility
                    ])

                input_list = []
                keypoint_unseen = False
                for group in fight_detector.coordinate_for_angel:
                    try:
                        c1, c2, c3 = keypoints[group[0]], keypoints[group[1]], keypoints[group[2]]
                        if is_coordinate_zero(c1, c2, c3) or c1[2] < 0.2 or c2[2] < 0.2 or c3[2] < 0.2:
                            keypoint_unseen = True
                            break
                        angle = calculate_angle(c1, c2, c3)
                        conf = (c1[2] + c2[2] + c3[2]) / 3
                        input_list.append(angle)
                        input_list.append(conf)
                    except IndexError:
                        keypoint_unseen = True
                        break

                if keypoint_unseen:
                    continue

                person_id = f"person_{det_index}"
                if person_id not in person_buffers:
                    person_buffers[person_id] = []
                person_buffers[person_id].append(input_list)
                if len(person_buffers[person_id]) > fight_detector.sequence_length:
                    person_buffers[person_id].pop(0)

                if len(person_buffers[person_id]) == fight_detector.sequence_length:
                    input_tensor = torch.Tensor([person_buffers[person_id]])
                    with torch.no_grad():
                        prob = fight_detector.model(input_tensor).item()
                    frame_probs.append(prob)
                    if prob > 0.5:
                        frame_violence = 1

                    label = "FIGHT" if prob > 0.5 else "NORMAL"
                    color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 마지막 상태 유지
            if not frame_probs:
                avg_prob = last_avg_prob
                frame_violence = last_frame_violence
            else:
                avg_prob = np.mean(frame_probs)
                last_avg_prob = avg_prob
                last_frame_violence = frame_violence

            frame_list.append(frame_count)
            prob_list.append(avg_prob)
            csv_writer.writerow([frame_count, frame_violence])
            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    csv_file.close()
    mp_pose.close()

    # 확률 그래프 저장
    plt.figure(figsize=(12, 6))
    plt.plot(frame_list, prob_list, marker='o')
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.title('Frame-wise Fight Probability')
    plt.xlabel('Frame')
    plt.ylabel('Average Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig(prob_plot_path)
    plt.close()
    print(f"[INFO] 확률 그래프 저장 완료: {prob_plot_path}")

if __name__ == "__main__":
    video_path = "fight_0051.mp4"
    run_fight_detection(video_path)
