import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
from tqdm import tqdm

from fight_module.yolo_pose_estimation import YoloPoseEstimation
from fight_module.fight_detector import FightDetector

def run_fight_detection(video_path):
    yolo_model_path = "model/yolo/yolov8x-pose.pt"
    fight_model_path = "model/fight/new-fight-model.pth"
    threshold = 0.6
    conclusion_threshold = 2
    final_threshold = 15
    output_path = f"mlp_result_{os.path.basename(video_path)}"

    pose_estimator = YoloPoseEstimation(yolo_model_path)
    fight_detector = FightDetector(
        fight_model=fight_model_path,
        threshold=threshold,
        conclusion_threshold=conclusion_threshold,
        final_threshold=final_threshold
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] fail to read video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = frame.copy()
            results = pose_estimator.estimate(frame)

            for r in results:
                if (
                    r.keypoints is not None and 
                    r.keypoints.xy is not None and 
                    r.keypoints.conf is not None
                ):
                    keypoints = r.keypoints.xy[0].cpu().numpy().tolist()
                    confs = r.keypoints.conf[0].cpu().numpy().tolist()

                    is_fight = fight_detector.detect(confs, keypoints)
                    annotated_frame = r.plot()
                    label = "FIGHT" if is_fight else "NORMAL"
                    color = (0, 0, 255) if is_fight else (0, 255, 0)
                    cv2.putText(annotated_frame, label, (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    break

            out.write(annotated_frame)
            cv2.imshow("Fight Detection", annotated_frame)
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
    run_fight_detection(video_path)
