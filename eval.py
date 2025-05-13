import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 📁 폴더 경로
gt_dir = "violence"
pred_dir = "output"

# 📊 평가 결과 저장용 리스트
all_metrics = []

# 📦 violence 폴더 내 정답 파일 순회
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".csv")])

for gt_file in gt_files:
    video_id = os.path.splitext(gt_file)[0]  # ex: A
    pred_file = f"{video_id}_predict.csv"
    pred_path = os.path.join(pred_dir, pred_file)
    gt_path = os.path.join(gt_dir, gt_file)

    # ✅ 예측 결과가 없으면 건너뜀
    if not os.path.exists(pred_path):
        print(f"⏭️  스킵: 예측 결과 없음 → {pred_file}")
        continue

    try:
        # ✅ CSV 불러오기 및 병합
        pred_df = pd.read_csv(pred_path)
        gt_df = pd.read_csv(gt_path)

        merged = pd.merge(gt_df, pred_df, on="frame", suffixes=('_gt', '_pred'))
        y_true = merged["violence_gt"]
        y_pred = merged["violence_pred"]

        # ✅ 평가 지표 계산
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        all_metrics.append({
            "video": video_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc
        })

        print(f"✅ {video_id}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} Acc={acc:.3f}")

    except Exception as e:
        print(f"❌ {video_id} 평가 실패: {e}")

# ✅ 전체 평균 계산
if all_metrics:
    result_df = pd.DataFrame(all_metrics)
    avg = result_df[["precision", "recall", "f1", "accuracy"]].mean()
    print("\n📊 전체 평균 평가 결과:")
    print(avg)

    # ✅ CSV 저장
    result_df.to_csv("evaluation_metrics_per_video.csv", index=False)
    print("📄 저장 완료: evaluation_metrics_per_video.csv")
else:
    print("🛑 평가 가능한 데이터가 없습니다.")