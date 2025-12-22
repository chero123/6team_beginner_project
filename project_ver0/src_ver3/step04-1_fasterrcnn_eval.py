# step04-1_fasterrcnn_eval.py
import os
import json
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from tqdm import tqdm

BASE = "/home/ohs3201/6team_beginner_project"
FRCNN_MODEL_PATH = f"{BASE}/results_v3/fasterrcnn/best.pth"
FRCNN_COCO_DIR = f"{BASE}/frcnn_dataset_v3"
VAL_JSON = f"{FRCNN_COCO_DIR}/val.json"

OUT_DIR = f"{BASE}/results_v3/fasterrcnn_val"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load category mapping (yolo2cat)
with open(f"{BASE}/category_mapping.json") as f:
    mp = json.load(f)

yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}
NUM_CLASSES = len(yolo2cat)


# Build FRCNN model (28 classes)
def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feat, num_classes + 1   # background 포함
    )
    return model


def main():
    print("\n========== STEP04-1: FasterRCNN Validation ==========")
    print(f"📌 Load model: {FRCNN_MODEL_PATH}")

    # Load model
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(FRCNN_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    coco = COCO(VAL_JSON)
    img_ids = coco.getImgIds()

    results = []

    for img_id in tqdm(img_ids, desc="FRCNN infer"):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]

        img_path = os.path.join(BASE, "yolo_dataset_v3/images/val", file_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        for box, score, lbl in zip(boxes, scores, labels):
            if score < 0.05:
                continue

            # lbl: 1 ~ 28 → our category id: 0 ~ 27
            category_id = int(lbl - 1)

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

    save_path = os.path.join(OUT_DIR, "frcnn_val_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f)

    print(f"\n📌 Saved: {save_path}")


if __name__ == "__main__":
    main()