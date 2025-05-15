import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 경로 설정
gt_dir = "violence"
pred_dir = "output"

# 결과 누적용
all_y_true = []
all_y_pred = []

# violence 폴더 내 모든 정답 CSV 순회
for file in os.listdir(gt_dir):
    if not file.endswith(".csv"):
        continue

    gt_path = os.path.join(gt_dir, file)
    pred_path = os.path.join(pred_dir, file)

    if not os.path.exists(pred_path):
        print(f"[WARNING] 예측 결과 누락: {file}")
        continue

    # CSV 불러오기
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    # frame 수가 다를 경우 자르기
    min_len = min(len(gt_df), len(pred_df))
    gt_df = gt_df[:min_len]
    pred_df = pred_df[:min_len]

    y_true = gt_df["violence"].tolist()
    y_pred = pred_df["violence"].tolist()

    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

# 전체 평가 결과 출력
print("===== 전체 평가 결과 (모든 영상 기준) =====")
print(f"Accuracy:  {accuracy_score(all_y_true, all_y_pred):.4f}")
print(f"Precision: {precision_score(all_y_true, all_y_pred):.4f}")
print(f"Recall:    {recall_score(all_y_true, all_y_pred):.4f}")
print(f"F1 Score:  {f1_score(all_y_true, all_y_pred):.4f}")
print("\nDetailed Report:\n")
print(classification_report(all_y_true, all_y_pred, target_names=["Normal", "Violence"]))
print("=====================================")