import cv2
import torch
import os
from tqdm import tqdm
from fight_module.yolo_pose_estimation import YoloPoseEstimation
from fight_module.fight_detector import FightDetector

def run_fight_detection(video_path):
    # ======= 설정 =======
    yolo_model_path = "model/yolo/yolov8x-pose.pt"
    # fight_model_path = "model/fight/fight-model.pth"
    fight_model_path = "model/fight/new-fight-model.pth"
    threshold = 0.6
    conclusion_threshold = 2
    final_threshold = 15
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"result_{os.path.basename(video_path)}")
    # ====================

    pose_estimator = YoloPoseEstimation(yolo_model_path)
    fight_detector = FightDetector(fight_model=fight_model_path)
    fight_detector.threshold = threshold
    fight_detector.conclusion_threshold = conclusion_threshold
    fight_detector.final_threshold = final_threshold

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 결과 영상 저장용 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    # tqdm 프로그래스바 시작
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 기본적으로 원본 프레임을 저장할 준비
            annotated_frame = frame.copy()

            # 포즈 추정 실행
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
                    break  # 첫 유효 결과만 처리

            # 결과 프레임을 반드시 저장
            out.write(annotated_frame)

            # 화면에 표시
            cv2.imshow("Fight Detection", annotated_frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

            frame_count += 1
            pbar.update(1)  # 프로그래스바 한 칸 이동

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 결과 영상 저장 완료: {output_path}")
    print(f"[INFO] 총 프레임 수: 입력 {total_frames}, 저장 {frame_count}")

if __name__ == "__main__":
    video_path = "ttt.mp4"  # 영상 경로 지정
    run_fight_detection(video_path)
