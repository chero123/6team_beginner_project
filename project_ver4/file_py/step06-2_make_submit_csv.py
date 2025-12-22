import os
import re
import json
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
MODEL = f"{PROJECT_ROOT}/runs/detect/ver4_finetune_1152_final/weights/best.pt"

TEST_DIR = "/home/ohs3201/6team_beginner_project/data/test_images"
OUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MAP_T2DL = os.path.join(PROJECT_ROOT, "mappings", "trainid_to_dlidx.json")

SUBMIT_CSV = os.path.join(OUT_DIR, "submission.csv")

IMGSZ = 1152
CONF = 0.6
IOU = 0.7
MAX_DET_PER_IMG = 4

with open(MAP_T2DL, "r", encoding="utf-8") as f:
    t2dl = {int(k): int(v) for k, v in json.load(f).items()}

def image_id_from_name(fn: str) -> int:
    # 파일명에서 숫자만 뽑는 방식 (예: 123.png -> 123)
    stem = os.path.splitext(os.path.basename(fn))[0]
    m = re.findall(r"\d+", stem)
    if not m:
        # 숫자 없는 경우는 그대로 0 처리(원하면 에러로 바꿔도 됨)
        return 0
    return int(m[0])

model = YOLO(MODEL)

test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".png")]
test_files = sorted(test_files, key=lambda x: image_id_from_name(x))
test_paths = [os.path.join(TEST_DIR, f) for f in test_files]

# stream=True: 메모리 덜 씀
results = model.predict(
    source=test_paths,
    imgsz=IMGSZ,
    conf=CONF,
    iou=IOU,
    max_det=300,
    device=0,
    stream=True,
    verbose=False,
)

rows = []
ann_id = 1

for path, r in zip(test_paths, results):
    img_id = image_id_from_name(path)

    # boxes: xyxy, conf, cls(train_id)
    if r.boxes is None or len(r.boxes) == 0:
        continue

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)

    # score desc 정렬
    order = confs.argsort()[::-1]
    order = order[:MAX_DET_PER_IMG]

    for i in order:
        x1, y1, x2, y2 = xyxy[i]
        score = float(confs[i])
        tid = int(clses[i])
        dl_idx = t2dl.get(tid, None)
        if dl_idx is None:
            continue

        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        if w <= 0 or h <= 0:
            continue

        rows.append({
            "annotation_id": ann_id,
            "image_id": img_id,
            "category_id": dl_idx,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "score": score,
        })
        ann_id += 1

df = pd.DataFrame(rows)

# image_id 오름차순, score 내림차순 (이미 그 순서로 들어가긴 하지만 확정)
if not df.empty:
    df = df.sort_values(["image_id", "score"], ascending=[True, False]).reset_index(drop=True)
    df["annotation_id"] = range(1, len(df) + 1)

df.to_csv(SUBMIT_CSV, index=False)
print("[DONE] submission saved:", SUBMIT_CSV)
print(df.head(10))