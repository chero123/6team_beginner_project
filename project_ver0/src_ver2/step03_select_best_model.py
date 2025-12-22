import os
import json
import glob
import pandas as pd
import numpy as np

BASE = "/home/ohs3201/6team_beginner_project"
CV_DIR = f"{BASE}/results/cv"

OUT_INFO = f"{CV_DIR}/best_model_info.json"

# FasterRCNN 점수 보정 스케일
FRCNN_SCALE = 3.0


# YOLO / RTDETR mAP50 읽기
def load_yolo_score(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if "metrics/mAP50(B)" not in df.columns:
            return None
        return df["metrics/mAP50(B)"].max()
    except:
        return None


# FasterRCNN mAP50-like 읽기
def load_frcnn_score(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        raw = data.get("mAP50_like", None)
        if raw is None:
            return None
        return raw / FRCNN_SCALE
    except:
        return None


# 모델별 평균 score 계산
def evaluate_model(model_name):
    model_path = f"{CV_DIR}/{model_name}"

    if not os.path.exists(model_path):
        print(f"⚠ {model_name}: 경로 없음 → 스킵")
        return None

    # FasterRCNN (json)
    if model_name == "fasterrcnn":
        json_list = sorted(glob.glob(f"{model_path}/fold*/cv_result.json"))
        scores = []

        for jf in json_list:
            s = load_frcnn_score(jf)
            if s is not None:
                scores.append(s)

        if not scores:
            print(f"⚠ {model_name}: 점수 없음 → 스킵")
            return None

        avg = float(np.mean(scores))
        print(f"▶ {model_name}: 평균 mAP50(after scale) = {avg:.4f}")
        return avg

    # YOLO, RTDETR (csv)
    csv_list = sorted(glob.glob(f"{model_path}/fold*/results.csv"))
    scores = []

    for cf in csv_list:
        s = load_yolo_score(cf)
        if s is not None:
            scores.append(s)

    if not scores:
        print(f"⚠ {model_name}: 점수 없음 → 스킵")
        return None

    avg = float(np.mean(scores))
    print(f"▶ {model_name}: 평균 mAP50 = {avg:.4f}")
    return avg


# fold 중 best weight 경로 찾기
def find_best_weight(model_name):
    model_dir = f"{CV_DIR}/{model_name}"

    if model_name == "fasterrcnn":
        files = sorted(glob.glob(f"{model_dir}/fold*/best.pth"))
        return files[0] if files else None

    # YOLO / RT-DETR
    files = sorted(glob.glob(f"{model_dir}/fold*/weights/best.pt"))
    return files[0] if files else None


# MAIN
def main():
    print("\n📌 Step03: Best 모델 선택 시작\n")

    models = ["yolov8m", "rtdetr", "fasterrcnn"]
    scores = {}

    # 성능 평가
    for m in models:
        print(f"\n[모델 평가] {m}")
        s = evaluate_model(m)
        if s is not None:
            scores[m] = s

    if not scores:
        print("❌ 평가 가능한 모델 없음")
        return

    print("\n📌 최종 score:", scores)

    # 최고 모델 선택
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]

    print(f"\n🔥 BEST MODEL = {best_model} (score={best_score:.4f})")

    # best weight 찾기
    best_weight = find_best_weight(best_model)
    if best_weight is None:
        print(f"❌ {best_model} 의 best weight를 찾을 수 없습니다.")
        return

    # info 저장
    info = {
        "model_name": best_model,
        "score": best_score,
        "weight": best_weight
    }

    with open(OUT_INFO, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✔ BEST 모델 정보 저장 완료 → {OUT_INFO}")
    print(f"✔ BEST 모델 가중치 → {best_weight}")


if __name__ == "__main__":
    main()