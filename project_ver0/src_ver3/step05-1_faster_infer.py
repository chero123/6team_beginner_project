import os
import json
import torch
import torchvision
from PIL import Image
import pandas as pd
from tqdm import tqdm

BASE = "/home/ohs3201/6team_beginner_project"
DATA_DIR = f"{BASE}/data/test_images"
FRCNN_WEIGHT = f"{BASE}/results_v3/fasterrcnn/best.pth"
CATEGORY_MAP = f"{BASE}/category_mapping.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load mapping
with open(CATEGORY_MAP, "r") as f:
    mp = json.load(f)
yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}
NUM_CLASSES = len(yolo2cat)


# Build FRCNN model
def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feat, num_classes + 1
    )
    return model


print("🔄 FasterRCNN 로드 중...")
model = build_model(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(FRCNN_WEIGHT, map_location=DEVICE))
model.eval()


submit_rows = []
test_imgs = sorted(os.listdir(DATA_DIR))
print(f"📸 총 테스트 이미지: {len(test_imgs)}")

ann_id = 1

for img_name in tqdm(test_imgs):
    image_id = int(img_name.replace(".jpg", "").replace(".png", ""))
    img_path = os.path.join(DATA_DIR, img_name)

    img = Image.open(img_path).convert("RGB")
    tensor = torchvision.transforms.functional.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        out = model([tensor])[0]

    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()

    for (x1, y1, x2, y2), lbl, score in zip(boxes, labels, scores):
        if score < 0.25:
            continue

        cls = lbl - 1     # FRCNN outputs 1~28 → 0~27
        cat = yolo2cat.get(cls, -1)

        submit_rows.append([
            ann_id, image_id, cat,
            x1, y1, x2 - x1, y2 - y1, float(score)
        ])
        ann_id += 1


# Save CSV
df = pd.DataFrame(submit_rows, columns=[
    "annotation_id", "image_id", "category_id",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
])

SAVE_PATH = f"{BASE}/results/submission/ver3/fasterrcnn_submit_v3.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
df.to_csv(SAVE_PATH, index=False)

print(f"🎉 FasterRCNN 제출 CSV 저장 완료 → {SAVE_PATH}")