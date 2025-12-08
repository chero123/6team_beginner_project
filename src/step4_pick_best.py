import os
import json
import numpy as np

HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

CV_PATH = os.path.join(BASE_PROJECT, "results", "cv")

MODELS = {
    "yolov8m": "yolov8m.json",
    "rtdetr-l": "rtdetr-l.json",
    "fasterrcnn": "fasterrcnn_map.json"
}


def load_json(path):
    if not os.path.exists(path):
        print(f"⚠ 파일 없음: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def pick_best_model():
    print("\n===============================")
    print("   📊 Step4: 모델 성능 비교")
    print("===============================")

    results = []

    for model_name, filename in MODELS.items():
        file_path = os.path.join(CV_PATH, filename)
        data = load_json(file_path)

        if data is None:
            print(f" - {model_name}: 결과 없음")
            continue

        avg = data.get("avg_score") or data.get("avg_mAP50")
        fold_scores = data.get("fold_scores", [])

        print(f"\n📌 {model_name}")
        print(" - Fold Scores:", fold_scores)
        print(f" - 평균 mAP50: {avg:.5f}")

        results.append((model_name, avg))

    if not results:
        print("\n❌ 사용 가능한 CV 결과가 없습니다.")
        return None

    # mAP50 최고 모델 선택
    best_model = max(results, key=lambda x: x[1])

    print("\n====================================")
    print(f"   🥇 베스트 모델: {best_model[0]}")
    print(f"   🔥 최고 평균 mAP50: {best_model[1]:.5f}")
    print("====================================")

    # 선택된 모델 기록 저장
    best_path = os.path.join(CV_PATH, "best_model.json")
    with open(best_path, "w") as f:
        json.dump({
            "best_model": best_model[0],
            "best_score": best_model[1]
        }, f, indent=2)

    print(f"\n📁 저장됨: {best_path}")

    return best_model[0]


if __name__ == "__main__":
    pick_best_model()