import os
import json
import torch
import torchvision
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

BASE = "/home/ohs3201/6team_beginner_project"
DATA_DIR = f"{BASE}/data/test_images"

YOLO_WEIGHT = f"{BASE}/results_v3/yolov8l_v3/train/weights/best.pt"
FRCNN_WEIGHT = f"{BASE}/results_v3/fasterrcnn/best.pth"
CATEGORY_MAP = f"{BASE}/category_mapping.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load mapping
with open(CATEGORY_MAP, "r") as f:
    mp = json.load(f)
yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}
NUM_CLASSES = len(yolo2cat)


# Build FRCNN
def build_frcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feat, num_classes + 1
    )
    return model


print("🔄 Load YOLO...")
y_model = YOLO(YOLO_WEIGHT)

print("🔄 Load FRCNN...")
f_model = build_frcnn(NUM_CLASSES).to(DEVICE)
f_model.load_state_dict(torch.load(FRCNN_WEIGHT, map_location=DEVICE))
f_model.eval()


# YOLO inference
def infer_yolo(img_path):
    results = y_model.predict(img_path, conf=0.25, iou=0.5, verbose=False)
    boxes = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls[0].item())
            score = float(b.conf[0].item())
            cat = yolo2cat.get(cls, -1)
            boxes.append([cat, x1, y1, x2, y2, score])
    return boxes


# FRCNN inference
def infer_frcnn(img_path):
    img = Image.open(img_path).convert("RGB")
    t = torchvision.transforms.functional.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        out = f_model([t])[0]

    boxes = []
    for (x1, y1, x2, y2), lbl, score in zip(out["boxes"], out["labels"], out["scores"]):
        if score < 0.25:
            continue
        cat = yolo2cat.get(int(lbl.item()) - 1, -1)
        boxes.append([cat, float(x1), float(y1), float(x2), float(y2), float(score)])
    return boxes


# Soft-NMS
def soft_nms(boxes, sigma=0.5, thresh=0.001):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = boxes[:, 5]
    kept = []

    while len(boxes) > 0:
        max_idx = np.argmax(scores)
        best = boxes[max_idx]
        kept.append(best)

        boxes = np.delete(boxes, max_idx, axis=0)
        scores = np.delete(scores, max_idx, axis=0)

        for i in range(len(boxes)):
            x1 = max(best[1], boxes[i][1])
            y1 = max(best[2], boxes[i][2])
            x2 = min(best[3], boxes[i][3])
            y2 = min(best[4], boxes[i][4])
            inter = max(0, x2 - x1) * max(0, y2 - y1)

            area1 = (best[3] - best[1]) * (best[4] - best[2])
            area2 = (boxes[i][3] - boxes[i][1]) * (boxes[i][4] - boxes[i][2])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0

            scores[i] *= np.exp(-(iou * iou) / sigma)

        keep = scores > thresh
        boxes = boxes[keep]
        scores = scores[keep]

    return kept


# Weighted Box Fusion (simple)
def wbf(boxes):
    if len(boxes) <= 1:
        return boxes

    fused = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue

        cat_i, x1_i, y1_i, x2_i, y2_i, s_i = boxes[i]
        group = [(x1_i, y1_i, x2_i, y2_i, s_i)]
        used.add(i)

        for j in range(i + 1, len(boxes)):
            if j in used:
                continue

            cat_j, x1_j, y1_j, x2_j, y2_j, s_j = boxes[j]
            if cat_i != cat_j:
                continue

            # IoU
            xx1 = max(x1_i, x1_j)
            yy1 = max(y1_i, y1_j)
            xx2 = min(x2_i, x2_j)
            yy2 = min(y2_i, y2_j)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            area_j = (x2_j - x1_j) * (y2_j - y1_j)
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0

            if iou > 0.5:
                group.append((x1_j, y1_j, x2_j, y2_j, s_j))
                used.add(j)

        if len(group) == 1:
            fused.append([cat_i, x1_i, y1_i, x2_i, y2_i, s_i])
        else:
            g = np.array(group)
            w = g[:, 4]
            x1 = np.sum(g[:, 0] * w) / np.sum(w)
            y1 = np.sum(g[:, 1] * w) / np.sum(w)
            x2 = np.sum(g[:, 2] * w) / np.sum(w)
            y2 = np.sum(g[:, 3] * w) / np.sum(w)
            score = np.max(w)
            fused.append([cat_i, x1, y1, x2, y2, score])

    return fused


# Main submission create
submit_rows = []
test_imgs = sorted(os.listdir(DATA_DIR))
ann_id = 1

print(f"📸 총 테스트 이미지: {len(test_imgs)}")

for img_name in tqdm(test_imgs):
    image_id = int(img_name.replace(".jpg", "").replace(".png", ""))
    img_path = os.path.join(DATA_DIR, img_name)

    y = infer_yolo(img_path)
    f = infer_frcnn(img_path)

    merged = y + f
    snms = soft_nms(merged)
    final = wbf(snms)

    for cat, x1, y1, x2, y2, score in final:
        submit_rows.append([
            ann_id, image_id, cat,
            x1, y1, x2 - x1, y2 - y1, score
        ])
        ann_id += 1

# Save CSV
df = pd.DataFrame(submit_rows, columns=[
    "annotation_id", "image_id", "category_id",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
])

SAVE_PATH = f"{BASE}/results/submission/ver3/ensemble_submit_v3.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
df.to_csv(SAVE_PATH, index=False)

print(f"🎉 Ensemble CSV 저장 완료 → {SAVE_PATH}")